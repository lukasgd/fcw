"""Container build, deployment, and patching commands.

This module provides commands for building container images locally, deploying
them to remote HPC clusters via FirecREST, and a streamlined workflow for
iterating on code without rebuilding the entire image.

Key workflows:

1. **Initial Build & Deploy** (slow, one-time setup):

   Build the download stage locally, push it, then build-offline on the cluster:

       fcw container build --stage download -t my-fcw-app:download .
       fcw container push my-fcw-app:download
       fcw container build-remote my-fcw-app:download \\
           -f env/Dockerfile.prod-multistage -t my-fcw-app:latest \\
           --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \\
           --enroot --wait

2. **Extract Code for Editing**:

   Extract code from a container stage to edit locally. Writes a sidecar
   ``./code.meta.json`` recording stage + container_path for later use:

       fcw container extract my-fcw-app /workspace/BrainBERT ./code

3. **Quick Iteration (bind-mount, no rebuild)**:

   Upload patched code and add bind-mount entries to the container's TOML.
   Mount target defaults to the sidecar's ``container_path``:

       fcw container patch --container my-fcw-app ./code
       # Then use: srun --environment env/container.toml ...

4. **Bake Changes (rebuild)**:

   Bake accumulated patches from the container TOML into a new image:

       fcw container rebuild my-fcw-app --tag my-fcw-app:v2 --enroot --wait
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import subprocess
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List, Optional, Tuple

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from fcw.core import (
    SLURM_FAILED_STATES,
    ContainerConfig,
    extract_job_id,
    format_sbatch_lines,
    get_async_client,
    get_client,
    get_error_console,
    get_global_sbatch_options,
    get_output_console,
    get_system,
    load_config,
    resolve_context,
)

app = typer.Typer(no_args_is_help=True)
_error = get_error_console
_output = get_output_console


def _wait_and_check(client, system: str, job_id: str, label: str = "Job") -> None:
    """Wait for a job to complete and raise Exit(1) if it failed."""
    job_info = client.wait_for_job(system_name=system, job_id=job_id)
    state = job_info[0]["status"]["state"]
    if isinstance(state, list):
        state = ",".join(state)
    if any(fs in state for fs in SLURM_FAILED_STATES):
        _error().print(f"[red]{label} {job_id} finished with state: {state}[/red]")
        _error().print(f"[dim]Hint: Run `fcw job logs {job_id}` to see output[/dim]")
        raise typer.Exit(1)


def _merge_build_args(
    config_args: Optional[dict[str, str]],
    cli_args: Optional[List[str]],
) -> List[str]:
    """Merge config-level build_args with CLI --build-arg flags.

    CLI args override config args for the same key.
    """
    merged = dict(config_args or {})
    for arg in (cli_args or []):
        if "=" in arg:
            k, v = arg.split("=", 1)
            merged[k] = v
        else:
            # value-less arg (KEY): inherit value from the build environment
            merged[arg] = ""
    return [f"{k}={v}" if v else k for k, v in merged.items()]


def _find_container_config(config, image_tag: str):
    """Find a container config matching an image tag.

    Matches by exact tag first, then by image name prefix (the part before ':').
    This handles multi-stage workflows where intermediate tags like
    'my-fcw-app:24.04-download' should match a config with tag 'my-fcw-app:24.04'.
    """
    image_name = image_tag.split(":")[0] if ":" in image_tag else image_tag
    # Exact match first
    for _name, cont_config in config.containers.items():
        if cont_config.tag == image_tag:
            return cont_config
    # Prefix match (same image name, different tag)
    for _name, cont_config in config.containers.items():
        conf_name = cont_config.tag.split(":")[0] if ":" in cont_config.tag else cont_config.tag
        if conf_name == image_name:
            return cont_config
    return None


def _podman_setup_block() -> str:  # TODO: simplify these to strictly necessary ones on lys/clariden
    """Generate the common shell setup block for podman on HPC nodes.

    Handles: HOME fallback, systemd wait, podman state cleanup, storage.conf,
    and XDG_RUNTIME_DIR creation.
    """
    return """\
export HOME="${HOME:-/users/$USER}"

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

# Clean up previous podman state (non-fatal if no prior state exists)
podman system reset -f || true
rm -Rf /dev/shm/$USER/*
rm -Rf /tmp/xdg-run-$(id -u)*

# Configure podman storage on local filesystem if not already present
mkdir -p $HOME/.config/containers
if [ ! -f $HOME/.config/containers/storage.conf ]; then
    cat > $HOME/.config/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF
fi

export XDG_RUNTIME_DIR="$(mktemp -d -p "${TMPDIR:-/tmp}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"
"""


# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------


KNOWN_SYSTEMS = {
    "daint": "linux/arm64",
    "clariden": "linux/arm64",
    "santis": "linux/arm64",
    "lys": "linux/arm64",
    "beverin": "linux/amd64",
}


def _detect_remote_platform(client, system: str) -> Optional[str]:
    """Detect the remote system's platform via /proc/cpuinfo, or None (with a warning).

    Returns a platform string like ``linux/arm64`` or ``linux/amd64``. On failure
    (e.g. the FirecREST head endpoint can't read the /proc/cpuinfo pseudo-file) it
    warns and returns None so the caller falls back to --platform/config; the remote
    build verifies architecture regardless.
    """
    if system in KNOWN_SYSTEMS:
        return KNOWN_SYSTEMS[system]

    detail = ""
    try:
        result = client.head(system_name=system, path="/proc/cpuinfo", num_lines=20)
        content = result if isinstance(result, str) else (
            result.get("content") or result.get("output") or "")
        content_lower = content.lower()
        if "aarch64" in content_lower or "arm" in content_lower:
            return "linux/arm64"
        if "x86_64" in content_lower or "genuineintel" in content_lower or "authenticamd" in content_lower:
            return "linux/amd64"
    except Exception as e:
        detail = f" ({e})"

    _error().print(
        f"[yellow]Warning: could not auto-detect remote platform for '{system}'{detail}. "
        f"Pass --platform linux/arm64|amd64 or set 'platform' in fcw.yaml "
        f"(the remote build will verify architecture).[/yellow]"
    )
    return None


def _detect_container_runtime() -> str:
    """Detect available container runtime (podman or docker)."""
    for runtime in ["podman", "docker"]:
        try:
            subprocess.run([runtime, "--version"], capture_output=True, check=True)
            return runtime
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("No container runtime found. Install podman or docker.")


def _parse_patch_mounts(toml_content: str) -> list[tuple[str, str]]:
    """Extract .patches/ bind-mounts from TOML content.

    Returns list of (host_path, container_path) tuples for mount entries
    where the host side contains ``/.patches/``.
    """
    pattern = re.compile(r'"([^"]*/.patches/[^"]*):([^"]*)"')
    return pattern.findall(toml_content)


def _create_rebuilt_toml(
    original_toml_path: str,
    new_toml_path: str,
    new_image_path: Optional[str] = None,
) -> None:
    """Copy a container TOML, removing .patches/ bind-mounts and retargeting image.

    The original file is not modified. The new file has:
    - all mount entries containing ``/.patches/`` removed (with trailing-comma
      cleanup);
    - its ``image = "..."`` line rewritten to *new_image_path* if given. Any
      existing value (including an empty string) is replaced. If there is no
      ``image =`` line, nothing is inserted.
    """
    content = Path(original_toml_path).read_text()
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        if re.search(r'"[^"]*/.patches/[^"]*:[^"]*"', line):
            continue
        filtered.append(line)
    result = "".join(filtered)
    result = re.sub(r',(\s*\])', r'\1', result)
    result = re.sub(r'mounts\s*=\s*\[\s*\]', 'mounts = []', result)
    if new_image_path is not None:
        result = re.sub(
            r'^(\s*image\s*=\s*)"[^"]*"',
            lambda m: f'{m.group(1)}"{new_image_path}"',
            result,
            count=1,
            flags=re.MULTILINE,
        )
    Path(new_toml_path).write_text(result)


def _sanitize_tag_suffix(tag: str) -> str:
    """Sanitize a tag's suffix (post-colon part) into an identifier fragment."""
    suffix = tag.split(":")[-1] if ":" in tag else tag
    return re.sub(r'[^a-zA-Z0-9]', '-', suffix).strip('-')


def _derive_container_name(original_name: str, original_tag: str, new_tag: str) -> str:
    """Derive a new container config name from the original entry and a new tag.

    Idempotent across version chains: if *original_name* already ends with the
    sanitized suffix of *original_tag*, that trailing suffix is replaced rather
    than re-appended. This keeps v1 -> v2 -> v3 chains stable::

        _derive_container_name("app", "my-app:v1", "my-app:v2") -> "app-v2"
        _derive_container_name("app-v2", "my-app:v2", "my-app:v3") -> "app-v3"
        _derive_container_name("app", "my-app:latest", "my-app:24.04") -> "app-24-04"
    """
    parent_suffix = _sanitize_tag_suffix(original_tag)
    stem = original_name
    if parent_suffix and stem.endswith(f"-{parent_suffix}"):
        stem = stem[: -(len(parent_suffix) + 1)]
    new_suffix = _sanitize_tag_suffix(new_tag)
    return f"{stem}-{new_suffix}"


def _isolated_staging_dir(config, base: str, key: str) -> str:
    """Return a unique remote staging subdir for a build/rebuild job.

    Two concurrent jobs must never share a staging directory: the Dockerfile
    and any stage tars uploaded there are named deterministically (e.g.
    ``Dockerfile``), so a shared parent leads to one job clobbering the
    other's inputs mid-flight.

    The returned path has the form ``.fcw/<base>/<safe_key>-<YYYYMMDDTHHMMSS>``.
    The embedded timestamp makes stale dirs easy to identify for later GC.
    """
    safe_key = re.sub(r'[^a-zA-Z0-9]', '-', key).strip('-') or "build"
    timestamp = time.strftime("%Y%m%dT%H%M%S")
    return config.resolve_path(f".fcw/{base}/{safe_key}-{timestamp}", remote=True)


_STAGING_TS_RE = re.compile(r"-(\d{8}T\d{6})$")


async def _scan_staging_dirs(
    client: Any, system: str, base_path: str
) -> list[tuple[str, Optional[datetime]]]:
    """List ``.fcw/<base>/*`` entries, return (name, parsed_timestamp_or_None)."""
    try:
        entries = await client.list_files(
            system_name=system, path=base_path, recursive=False
        )
    except Exception:
        return []
    out: list[tuple[str, Optional[datetime]]] = []
    for entry in entries:
        name = (
            entry.get("name") if isinstance(entry, dict)
            else getattr(entry, "name", None)
        )
        if not name:
            continue
        m = _STAGING_TS_RE.search(name)
        ts: Optional[datetime] = None
        if m:
            try:
                ts = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
            except ValueError:
                ts = None
        out.append((name, ts))
    return out


def _staging_cleanup_block(staging_dir: str) -> str:
    """Bash snippet that removes *staging_dir* on successful SLURM-script exit.

    Registered via ``trap`` so it runs on the normal exit path but not when
    the script aborts under ``set -e``, preserving the staging dir for
    post-mortem inspection when a job fails.
    """
    q = shlex.quote(staging_dir)
    return f"""
# Clean up isolated staging dir on successful exit (preserved on failure for debugging)
_fcw_staging_cleanup() {{ rm -rf {q} || true; }}
trap '[ $? -eq 0 ] && _fcw_staging_cleanup' EXIT
"""


def _derive_rebuilt_toml_path(original: Path, original_tag: str, new_tag: str) -> Path:
    """Derive the sibling TOML path for a rebuilt container version.

    Mirrors ``_derive_container_name`` but operates on the original TOML path's
    stem so the new file lives next to the original with a version-suffixed
    name::

        env/container.toml,    app:v1 -> app:v2  ->  env/container-v2.toml
        env/container-v2.toml, app:v2 -> app:v3  ->  env/container-v3.toml
        env/node-burn.toml,    app:v1 -> app:v2  ->  env/node-burn-v2.toml
    """
    parent_suffix = _sanitize_tag_suffix(original_tag)
    stem = original.stem
    if parent_suffix and stem.endswith(f"-{parent_suffix}"):
        stem = stem[: -(len(parent_suffix) + 1)]
    new_suffix = _sanitize_tag_suffix(new_tag)
    return original.with_name(f"{stem}-{new_suffix}{original.suffix}")


# -----------------------------------------------------------------------------
# Sidecar metadata for extracted code dumps
# -----------------------------------------------------------------------------
# An extracted dump at <local_dest> has a sidecar JSON at <local_dest>.meta.json
# describing which stage/container path/image the code came from. `extract`
# writes it; `patch` reads it to default the bind-mount target; `rebuild` uses
# it to group patches by stage. `patch` never writes the sidecar.

SIDECAR_SUFFIX = ".meta.json"


def _sidecar_path(local_dest: str) -> Path:
    p = Path(local_dest).resolve()
    return p.with_name(p.name + SIDECAR_SUFFIX)


def _read_sidecar(local_dest: str) -> Optional[dict]:
    sp = _sidecar_path(local_dest)
    if not sp.exists():
        return None
    try:
        data = json.loads(sp.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    return data if isinstance(data, dict) else None


def _write_sidecar(local_dest: str, *, stage: str, container_path: str, source_image: str) -> None:
    sp = _sidecar_path(local_dest)
    sp.write_text(json.dumps({
        "stage": stage,
        "container_path": container_path,
        "source_image": source_image,
        "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2) + "\n")


# -----------------------------------------------------------------------------
# Patch index: local map of remote patch-name -> local dump path
# -----------------------------------------------------------------------------
# `patch` uploads a local dump to `.patches/<basename>/` on the remote and
# records the bind-mount in the container's TOML. On later `job submit` /
# `container rebuild`, we want to re-sync any changes to that local dump
# before the job runs. The bind-mount only stores the *remote* path, so we
# keep a local index mapping patch-name -> absolute local path per container.
# Survives renames/moves of the local dump because `patch` refreshes it.

def _patches_index_path(container_name: str) -> Path:
    return Path(".fcw") / "patches" / f"{container_name}.json"


def _read_patches_index(container_name: str) -> dict:
    p = _patches_index_path(container_name)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return data if isinstance(data, dict) else {}


def _record_patch_in_index(container_name: str, patch_name: str, local_path: str) -> None:
    idx = _read_patches_index(container_name)
    idx[patch_name] = os.path.abspath(local_path)
    p = _patches_index_path(container_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(idx, indent=2, sort_keys=True) + "\n")


def _resync_container_patches(config: Any, container_name: str, system: str, account: str) -> None:
    """Incrementally re-upload patch dumps for a container before a job runs.

    Looks up the container's TOML, parses ``.patches/`` bind-mount entries,
    and for each one finds the local source via the patch index and re-uploads
    only files changed since the last sync. No-op if the container has no
    TOML, no patch mounts, or no index entry for a given mount.

    Also refreshes the sidecar JSON next to each patch dir on the remote so
    ``rebuild`` can still group patches by stage correctly.

    Silent on missing index entries — a TOML mount without an index entry
    means the patch was added out-of-band (e.g. manually edited TOML) and
    we can't know where to re-sync from; the job still sees whatever is on
    the remote.
    """
    if container_name not in config.containers:
        return
    cont = config.containers[container_name]
    if not cont.toml:
        return
    toml_path = Path(cont.toml)
    if not toml_path.exists():
        return
    mounts = _parse_patch_mounts(toml_path.read_text())
    if not mounts:
        return
    idx = _read_patches_index(container_name)
    if not idx:
        return

    from fcw.commands.data import _upload_incremental

    async def sync_all() -> None:
        async_client = get_async_client()
        for remote_patch_dir, _container_path in mounts:
            patch_name = os.path.basename(remote_patch_dir.rstrip("/"))
            local_path = idx.get(patch_name)
            if not local_path or not os.path.isdir(local_path):
                continue
            try:
                count = await _upload_incremental(
                    async_client, system, account, local_path, remote_patch_dir
                )
                if count:
                    _error().print(
                        f"[dim]Re-synced {count} changed file(s) "
                        f"{local_path} -> {remote_patch_dir}[/dim]"
                    )
            except Exception as e:
                _error().print(
                    f"[yellow]Patch resync failed for {local_path}: {e}[/yellow]"
                )
                continue

            sidecar = _sidecar_path(local_path)
            if sidecar.exists():
                try:
                    await async_client.upload(
                        system_name=system,
                        local_file=str(sidecar),
                        directory=os.path.dirname(remote_patch_dir),
                        filename=sidecar.name,
                        account=account,
                        blocking=True,
                    )
                except Exception:
                    pass

    try:
        asyncio.run(sync_all())
    except Exception as e:
        _error().print(f"[yellow]Patch resync skipped: {e}[/yellow]")


def _resolve_container_config(config, name: str) -> ContainerConfig:
    """Look up a container by config name, error with helpful listing otherwise."""
    if name in config.containers:
        return config.containers[name]
    known = ", ".join(sorted(config.containers.keys())) or "(none)"
    _error().print(
        f"[red]Unknown container name: {name!r}[/red]\n"
        f"[dim]Known containers in fcw.yaml: {known}[/dim]"
    )
    raise typer.Exit(1)


def _resolve_extract_stage(container_cfg: ContainerConfig, requested: Optional[str]) -> str:
    """Default stage for extract: requested > 'download' if present > first local stage."""
    stages = container_cfg.get_local_stages()
    if requested:
        if requested not in stages and requested != container_cfg.get_remote_stage():
            _error().print(
                f"[yellow]Warning: stage {requested!r} not in configured stages "
                f"{stages + [container_cfg.get_remote_stage()]}[/yellow]"
            )
        return requested
    if "download" in stages:
        return "download"
    return stages[0]


def _parse_patch_arg(arg: str, sidecar_dir: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Parse ``<local-path>[:<container-path>]`` into (local, container_path_or_None).

    Splits on the first ':' not inside the local path. Windows-style paths aren't
    supported here (fcw is Linux-only at the CLI boundary).
    """
    if ":" in arg:
        local, container_path = arg.split(":", 1)
        return local, container_path
    return arg, None


def _build_one_stage(
    runtime: str,
    *,
    dockerfile: str,
    tag: str,
    stage: Optional[str],
    platform: Optional[str],
    build_args: List[str],
    context: str,
) -> None:
    """Run a single container build (podman/docker build).

    Raises typer.Exit(1) on failure.
    """
    cmd = [runtime, "build", "-f", dockerfile, "-t", tag]
    if stage:
        cmd.extend(["--target", stage])
    if platform:
        cmd.extend(["--platform", platform])
    for arg in build_args:
        cmd.extend(["--build-arg", arg])
    cmd.append(context)

    _error().print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _error().print(f"[red]Build failed: {tag}[/red]")
        raise typer.Exit(1)
    _error().print(f"[green]Built image: {tag}[/green]")


@app.command("build")
def build_image(
    ctx: typer.Context,
    name_or_context: str = typer.Argument(".", help="Container name from config, or build context directory"), # FIXME: this should be just the name, not build context dir (no overlap between these two). It should be optional and default to None. Requires adapting the code below.
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Dockerfile path"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Image tag"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Build specific stage (or all local_stages if omitted)"),
    platform: Optional[str] = typer.Option(None, "--platform", help="Target platform (e.g., linux/arm64)"),
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Set build-time variables (KEY=VALUE)"),
    context: Optional[str] = typer.Option(None, "--context", "-C", help="Build context directory (default: .)"),
    save: Optional[str] = typer.Option(None, "--save", "-o", help="Save image to tar file"),
):
    """Build a container image locally.

    When NAME_OR_CONTEXT matches a container name from fcw.yaml, all build
    parameters (Dockerfile, tag, platform, build_args) are resolved from
    config.  CLI flags override config values.

    Without --stage, builds all local_stages defined in config (default
    is just the download stage).  With --stage, builds only that stage.

    When NAME_OR_CONTEXT is a directory, it is used as the build context
    (backward-compatible).

    Examples::

        # Build all local stages for a config container
        fcw container build ngc-brainbert

        # Build only the download stage
        fcw container build ngc-brainbert --stage download

        # Legacy: build context directory with explicit flags
        fcw container build . -f Dockerfile -t my-app:latest --stage download
    """
    config = load_config((ctx.obj or {}).get("config_file"))

    # Resolve whether name_or_context is a config name or a directory
    cont_config = config.containers.get(name_or_context) if config.containers else None  # FIXME: throw error if argument was not none, but not found in containers
    if cont_config:
        resolved_file = file or cont_config.file
        resolved_tag = tag or cont_config.tag
        resolved_platform = platform or cont_config.platform
        build_context = context or "."  # FIXME: this should default to cont_config.context, i.e. build context needs to be part of the config.
    else:
        # Treat as build context directory (backward-compatible)
        build_context = context or name_or_context  # FIXME: should be just context, not name_or_context
        resolved_file = file
        resolved_tag = tag
        resolved_platform = platform

    # If no tag yet, try first container from config  # FIXME: rather than just using the first containers tag, throw an error that tag is required.
    if not resolved_tag and config.containers:
        first_container = next(iter(config.containers.values()))
        resolved_tag = first_container.tag

    if not resolved_file:
        _error().print("[red]No Dockerfile specified. Use --file or configure in fcw.yaml.[/red]")
        raise typer.Exit(1)

    runtime = _detect_container_runtime()
    _error().print(f"[dim]Using container runtime: {runtime}[/dim]")

    # Resolve platform: CLI/config > auto-detect from remote
    if not resolved_platform:
        try:
            system = get_system((ctx.obj or {}).get("system"))
            client = get_client()
            detected = _detect_remote_platform(client, system)
            if detected:
                resolved_platform = detected
                _error().print(f"[dim]Detected remote platform: {resolved_platform}[/dim]")
        except Exception:
            pass

    # Merge config-level build_args with CLI --build-arg flags
    all_build_args = _merge_build_args(
        cont_config.build_args if cont_config else None, build_arg
    )

    # Determine which stages to build
    if stage:
        # Single stage: derive stage-specific tag
        if cont_config and not tag:
            stage_tag = cont_config.stage_tag(stage)
        else:
            stage_tag = resolved_tag
        _build_one_stage(
            runtime,
            dockerfile=resolved_file,
            tag=stage_tag,
            stage=stage,
            platform=resolved_platform,
            build_args=all_build_args,
            context=build_context,
        )
        # Save if requested
        if save:
            _save_image(runtime, [stage_tag], save)
    elif cont_config and not tag:
        # Config name without --stage: build all local_stages
        local_stages = cont_config.get_local_stages()
        for s in local_stages:
            stage_tag = cont_config.stage_tag(s)
            _error().print(f"[bold]Building stage '{s}' → {stage_tag}[/bold]")
            _build_one_stage(
                runtime,
                dockerfile=resolved_file,
                tag=stage_tag,
                stage=s,
                platform=resolved_platform,
                build_args=all_build_args,
                context=build_context,
            )
        if save:
            _save_image(runtime, [cont_config.stage_tag(s) for s in local_stages], save)
    else:
        # No config or explicit --tag: single build (legacy behavior)
        _build_one_stage(
            runtime,
            dockerfile=resolved_file,
            tag=resolved_tag,
            stage=None,
            platform=resolved_platform,
            build_args=all_build_args,
            context=build_context,
        )
        if save:
            _save_image(runtime, [resolved_tag], save)


def _save_image(runtime: str, tags: list[str], path: str) -> None:
    """Export one or more container images to a single tar archive."""
    cmd = [runtime, "save"]
    if runtime == "podman" and len(tags) > 1:
        cmd.append("--multi-image-archive")
    cmd += ["-o", path, *tags]
    _error().print(f"[dim]Saving image to {path}...[/dim]")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _error().print("[red]Failed to save image[/red]")
        raise typer.Exit(1)
    _error().print(f"[green]Saved image to {path}[/green]")


def _push_one_image(
    image_tag: str,
    *,
    config,
    system: str,
    account: str,
    remote_dir: str,
    platform: Optional[str] = None,
) -> None:
    """Export and upload a single image tag to remote storage.

    Exports the image to a temporary tar, uploads via FirecREST, then cleans up.
    """
    runtime = _detect_container_runtime()  # FIXME: do the following lines do the same thing like _save_image above?  
    remote_filename = image_tag.replace(":", "+").replace("/", "+") + ".tar"
    tar_path = os.path.join(tempfile.gettempdir(), remote_filename)

    _error().print(f"[dim]Exporting {image_tag}...[/dim]")

    save_cmd = [runtime, "save", "-o", tar_path]
    if platform:
        help_output = subprocess.run(
            [runtime, "save", "--help"], capture_output=True, text=True
        ).stdout
        if "--platform" in help_output:
            save_cmd.extend(["--platform", platform])
        else:
            _error().print(
                f"[yellow]Warning: {runtime} save does not support --platform. "
                "Ignoring platform argument.[/yellow]"
            )
    save_cmd.append(image_tag)

    result = subprocess.run(save_cmd)
    if result.returncode != 0:
        _error().print(f"[red]Failed to export image: {image_tag}[/red]")
        raise typer.Exit(1)

    try:
        async def do_upload():
            client = get_async_client()
            try:
                await client.mkdir(
                    system_name=system, path=remote_dir, create_parents=True
                )
            except Exception:
                pass

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=_error(),
            ) as progress:
                progress.add_task(
                    f"Uploading {remote_filename} to {remote_dir}...", total=None
                )
                await client.upload(
                    system_name=system,
                    local_file=tar_path,
                    directory=remote_dir,
                    filename=remote_filename,
                    account=account,
                    blocking=True,
                )

        asyncio.run(do_upload())
        _error().print(f"[green]Uploaded {remote_filename} to {remote_dir}[/green]")
    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)


@app.command("push")
def push_image(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Config name, image tag, or tar file path"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Push specific stage (e.g., download)"),
    remote_path: Optional[str] = typer.Option(None, "--to", help="Remote path (default: from config)"),
    platform: Optional[str] = typer.Option(None, "--platform", help="Target platform (e.g., linux/arm64)"),
    do_import: bool = typer.Option(False, "--import", help="Import to squashfs after push"),
):
    """Upload a container image to remote storage.

    When IMAGE matches a container name from fcw.yaml, pushes all local_stages
    (or a specific stage with --stage).

    When IMAGE is a tar file, uploads directly.  When IMAGE is a tag, exports
    and uploads.

    Examples::

        # Push all local stages for a config container
        fcw container push ngc-brainbert

        # Push only the download stage
        fcw container push ngc-brainbert --stage download

        # Legacy: push a specific tag
        fcw container push my-app:v1-download
    """
    config, system, account = resolve_context(ctx)

    # Check if image is a config name
    cont_config = config.containers.get(image) if config.containers else None

    if cont_config:
        # Resolve remote directory
        images_dir = remote_path or cont_config.remote_path or "ce-images/"
        images_dir = config.resolve_path(images_dir, remote=True)

        if stage:
            # Push one specific stage
            stage_tag = cont_config.stage_tag(stage)
            _error().print(f"[bold]Pushing stage '{stage}' → {stage_tag}[/bold]")
            _push_one_image(
                stage_tag,
                config=config,
                system=system,
                account=account,
                remote_dir=images_dir,
                platform=platform,
            )
        else:
            # Push all local stages
            local_stages = cont_config.get_local_stages()
            for s in local_stages:
                stage_tag = cont_config.stage_tag(s)
                _error().print(f"[bold]Pushing stage '{s}' → {stage_tag}[/bold]")
                _push_one_image(
                    stage_tag,
                    config=config,
                    system=system,
                    account=account,
                    remote_dir=images_dir,
                    platform=platform,
                )

        if do_import:
            ctx.invoke(import_image, image=cont_config.tag)
        return

    # --- Legacy behavior: image is a tag or tar path ---

    # Determine remote path
    if remote_path is None:
        matched = _find_container_config(config, image)
        if matched and matched.remote_path:
            remote_path = matched.remote_path
        else:
            remote_path = "ce-images/"
    remote_path = config.resolve_path(remote_path, remote=True)

    # Check if image is a file or a tag
    if os.path.isfile(image):
        tar_path = image
        remote_filename = os.path.basename(image)
        cleanup_tar = False
    else:
        # Export image to tar
        runtime = _detect_container_runtime()
        remote_filename = image.replace(":", "+").replace("/", "+") + ".tar"
        tar_path = os.path.join(tempfile.gettempdir(), remote_filename)
        cleanup_tar = True

        _error().print(f"[dim]Exporting {image}...[/dim]")

        save_cmd = [runtime, "save", "-o", tar_path]
        if platform:
            help_output = subprocess.run(
                [runtime, "save", "--help"], capture_output=True, text=True
            ).stdout
            if "--platform" in help_output:
                save_cmd.extend(["--platform", platform])
            else:
                _error().print(
                    f"[yellow]Warning: {runtime} save does not support --platform. "
                    "Ignoring platform argument.[/yellow]"
                )
        save_cmd.append(image)

        result = subprocess.run(save_cmd)
        if result.returncode != 0:
            _error().print("[red]Failed to export image[/red]")
            raise typer.Exit(1)

    try:  # FIXME: is this the same do_upload function as the one in _push_one_image?
        async def do_upload():
            client = get_async_client()
            target_dir = os.path.dirname(remote_path)
            try:
                await client.mkdir(
                    system_name=system, path=target_dir, create_parents=True
                )
            except Exception:
                pass

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=_error(),
            ) as progress:
                progress.add_task(f"Uploading to {remote_path}...", total=None)
                await client.upload(
                    system_name=system,
                    local_file=tar_path,
                    directory=target_dir,
                    filename=remote_filename,
                    account=account,
                    blocking=True,
                )

        asyncio.run(do_upload())
        _error().print(f"[green]Uploaded to {remote_path}[/green]")

    finally:
        if cleanup_tar and os.path.exists(tar_path):
            os.unlink(tar_path)

    if do_import:
        ctx.invoke(import_image, image=image)


def _resolve_remote_tar(image_or_path: str, config) -> str:
    """Resolve an image tag or tar path to a remote tar path.

    If the argument looks like a tar path (ends with .tar or contains /),
    resolve it as a path. Otherwise treat it as an image tag and convert
    to the canonical tar filename under the container's configured remote_path
    (falling back to ce-images/).
    """
    if image_or_path.endswith(".tar") or "/" in image_or_path:
        return config.resolve_path(image_or_path, remote=True)
    tar_name = image_or_path.replace(":", "+").replace("/", "+") + ".tar"
    cont_config = _find_container_config(config, image_or_path)
    if cont_config:
        images_dir = config.resolve_container_images_dir(cont_config)
        return os.path.join(images_dir, tar_name)
    return config.resolve_path(f"ce-images/{tar_name}", remote=True)


@app.command("import")
def import_image(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Image tag or remote tar file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for squashfs"),
):
    """Import a container image with enroot on the remote cluster.

    Submits a job that loads the tar with podman and converts it to
    enroot squashfs format.

    Accepts an image tag (e.g., my-fcw-app:latest) or a remote tar path
    (e.g., ce-images/my-fcw-app-latest.tar).
    """
    config, system, account = resolve_context(ctx)

    remote_tar = _resolve_remote_tar(image, config)

    output_path = output or remote_tar.replace(".tar", ".sqsh")
    output_path = config.resolve_path(output_path, remote=True)

    q_remote_tar = shlex.quote(remote_tar)
    q_output_path = shlex.quote(output_path)

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-import
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-import-%j.out
{format_sbatch_lines(get_global_sbatch_options())}
set -e

{_podman_setup_block()}

# Load image
echo "Loading image from {q_remote_tar}..."
podman load -i {q_remote_tar}

# Get the loaded image name
IMAGE_ID=$(podman images --format "{{{{.ID}}}}" | head -1)
echo "Loaded image: $IMAGE_ID"

# Convert to enroot squashfs
echo "Converting to enroot squashfs: {q_output_path}..."
rm -f {q_output_path}
enroot import -x mount -o {q_output_path} podman://$IMAGE_ID || true
if [ ! -f {q_output_path} ]; then
    echo "ERROR: enroot import failed - output not found: {q_output_path}"
    exit 1
fi

echo "Done: {q_output_path}"
exit 0
"""

    # Submit job
    client = get_client()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=config.workdir.remote,
            script_local_path=script_path,
        )

        job_id = extract_job_id(result)
        print(job_id)
        _error().print(f"[green]Submitted import job: {job_id}[/green]")
        _error().print(f"[dim]Output will be: {output_path}[/dim]")

    finally:
        os.unlink(script_path)


def _stage_to_build_arg_name(stage_name: str) -> str:
    """Derive build-arg name from a stage name.

    Convention: ``DOWNLOAD_IMAGE`` for the ``download`` stage (backward-compatible),
    ``<STAGE>_IMAGE`` (uppercase, hyphens to underscores) for others.

    Examples::

        _stage_to_build_arg_name("download")          -> "DOWNLOAD_IMAGE"
        _stage_to_build_arg_name("runtime-download")  -> "RUNTIME_DOWNLOAD_IMAGE"
    """
    return stage_name.upper().replace("-", "_") + "_IMAGE"


def _generate_load_and_resolve_block(
    stage_tags: list[tuple[str, str]],
    images_dir: str,
) -> tuple[str, str]:
    """Generate bash for loading tars and resolving image IDs.

    Args:
        stage_tags: list of ``(stage_name, image_tag)`` pairs.
        images_dir: quoted remote directory containing tars.

    Returns:
        ``(load_block, build_arg_lines)`` where *load_block* is the bash to
        load all tars and verify arch, and *build_arg_lines* are the
        ``--build-arg`` entries for the podman build command.
    """
    load_lines = []
    build_arg_parts = []

    for stage_name, image_tag in stage_tags:
        var_name = _stage_to_build_arg_name(stage_name).replace("-", "_")
        tar_name = image_tag.replace(":", "+").replace("/", "+") + ".tar"
        q_tar = shlex.quote(f"{images_dir}/{tar_name}")
        q_tag = shlex.quote(image_tag)

        load_lines.append(f"""
# Load {stage_name} stage from tar
if [ -f {q_tar} ]; then
    echo "Loading {image_tag} from {tar_name}..."
    podman load -i {q_tar}
else
    echo "Error: tar not found: {q_tar}"
    exit 1
fi
{var_name}_ID=$(podman image inspect --format '{{{{.Id}}}}' docker.io/library/{image_tag} 2>/dev/null || podman image inspect --format '{{{{.Id}}}}' {q_tag})
echo "Resolved {stage_name} image ID: ${var_name}_ID"
""")
        build_arg_parts.append(f"    --build-arg {var_name}=${var_name}_ID \\")

    # Architecture check on the first loaded image
    if stage_tags:
        first_var = _stage_to_build_arg_name(stage_tags[0][0]).replace("-", "_")
        load_lines.append(f"""
# Verify image architecture matches compute node (never emulate on HPC)
IMAGE_ARCH=$(podman image inspect --format '{{{{.Architecture}}}}' "${first_var}_ID")
NODE_ARCH=$(uname -m)
case "$NODE_ARCH" in x86_64) NODE_ARCH=amd64;; aarch64) NODE_ARCH=arm64;; esac
if [ "$IMAGE_ARCH" != "$NODE_ARCH" ]; then
    echo "ERROR: Image architecture ($IMAGE_ARCH) does not match node ($NODE_ARCH)."
    echo "Rebuild the base image with --platform linux/$NODE_ARCH"
    exit 1
fi
""")

    return "\n".join(load_lines), "\n".join(build_arg_parts)


@app.command("build-remote")
def build_remote(  # FIXME: ce-images/ is used repeatedly as default remote dir - should this be configured in one place?
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Config name, or base image tag (must be pushed)"),
    dockerfile: Optional[str] = typer.Option(None, "--file", "-f", help="Local Dockerfile path"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Tag for the built image"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Target stage to build"),
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Build-time variables (KEY=VALUE)"),
    enroot: bool = typer.Option(False, "--enroot", help="Convert final image to enroot squashfs"), # FIXME: additionally enable pushing to registry
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for enroot squashfs"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for job completion"),
):
    """Build a container image on the remote cluster via a SLURM job.

    When IMAGE matches a container name from fcw.yaml, resolves the
    Dockerfile, output tag, stage, and images to load from config.  The
    --file and --tag options become optional.

    Otherwise, IMAGE is treated as a base image tag (legacy behavior).

    Examples::

        # Config-aware: resolves everything from fcw.yaml
        fcw container build-remote ngc-brainbert --enroot --wait

        # Legacy: explicit flags
        fcw container build-remote my-app:download \\
            -f env/Dockerfile -t my-app:latest \\
            --stage build-offline --enroot --wait
    """
    config, system, account = resolve_context(ctx)

    # Check if image is a config name
    cont_config = config.containers.get(image) if config.containers else None

    if cont_config:
        # Resolve from config
        resolved_dockerfile = dockerfile or cont_config.file
        resolved_tag = tag or cont_config.tag
        resolved_stage = stage or cont_config.get_remote_stage()
        images_dir = config.resolve_container_images_dir(cont_config)

        # Build list of (stage_name, tag) for all local stages to load
        local_stages = cont_config.get_local_stages()
        stage_tags = [(s, cont_config.stage_tag(s)) for s in local_stages]

        all_build_args = _merge_build_args(cont_config.build_args, build_arg)
    else:
        # Legacy: image is a raw tag
        if not dockerfile:
            _error().print("[red]--file/-f is required when not using a config name.[/red]")
            raise typer.Exit(1)
        if not tag:
            _error().print("[red]--tag/-t is required when not using a config name.[/red]")
            raise typer.Exit(1)
        resolved_dockerfile = dockerfile
        resolved_tag = tag
        resolved_stage = stage

        # Legacy: single image, resolve from tag matching
        matched = _find_container_config(config, tag) or _find_container_config(config, image)
        images_dir = config.resolve_container_images_dir(matched) if matched else config.resolve_path("ce-images/", remote=True)

        # Single stage: use "download" convention for the loaded image
        stage_tags = [("download", image)]

        all_build_args = _merge_build_args(
            matched.build_args if matched else None, build_arg
        )

    if not os.path.isfile(resolved_dockerfile):
        _error().print(f"[red]Dockerfile not found: {resolved_dockerfile}[/red]")
        raise typer.Exit(1)

    staging_dir = _isolated_staging_dir(
        config, "build-remote", f"{image}-{resolved_tag}"
    )

    if enroot:
        output_path = output or os.path.join(
            images_dir, f"{resolved_tag.replace(':', '+')}.sqsh"
        )

    # Step 1: Upload Dockerfile
    _error().print("[bold]Step 1: Uploading Dockerfile...[/bold]")

    async def do_upload():
        async_client = get_async_client()
        try:
            await async_client.mkdir(
                system_name=system, path=staging_dir, create_parents=True
            )
        except Exception:
            pass
        await async_client.upload(
            system_name=system,
            local_file=resolved_dockerfile,
            directory=staging_dir,
            filename="Dockerfile",
            account=account,
            blocking=True,
        )

    asyncio.run(do_upload())

    # Step 2: Build job script
    _error().print("[bold]Step 2: Submitting build job...[/bold]")

    # Generate image loading and build-arg blocks
    load_block, image_build_args = _generate_load_and_resolve_block(
        stage_tags, images_dir
    )

    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in all_build_args)
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""
    target_line = f"    --target {shlex.quote(resolved_stage)} \\\n" if resolved_stage else ""

    q_tag = shlex.quote(resolved_tag)
    q_staging_dir = shlex.quote(staging_dir)
    q_images_dir = shlex.quote(images_dir)

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-build-remote
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-build-remote-%j.out
#SBATCH --error=fcw-container-build-remote-%j.out
{format_sbatch_lines(get_global_sbatch_options())}
set -euxo pipefail
{_staging_cleanup_block(staging_dir)}
{_podman_setup_block()}
export CE_IMAGES_DIR={q_images_dir}

{load_block}

echo "=== Building {q_tag} ==="
cd {q_staging_dir}
podman build \\
{target_line}{image_build_args}
{extra_build_args_line}    -t {q_tag} \\
    -f Dockerfile .

echo "Built image: {resolved_tag}"
"""

    if enroot:
        q_output_path = shlex.quote(output_path)
        script += f"""
echo "=== Exporting to enroot ==="
mkdir -p $(dirname {q_output_path})
rm -f {q_output_path}
enroot import -x mount -o {q_output_path} podman://{resolved_tag} || true
if [ ! -f {q_output_path} ]; then
    echo "ERROR: enroot import failed - output not found: {q_output_path}"
    exit 1
fi
echo "Exported to: {q_output_path}"
ls -lh {q_output_path}
"""

    script += "\nexit 0\n"

    client = get_client()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=config.workdir.remote,
            script_local_path=script_path,
        )

        job_id = extract_job_id(result)
        print(job_id)
        _error().print(f"[green]Submitted build job: {job_id}[/green]")

        if wait:
            _error().print("[dim]Waiting for build to complete...[/dim]")
            _wait_and_check(client, system, job_id, "Build job")
            _error().print(f"[green]Build complete: {resolved_tag}[/green]")
            if enroot:
                _error().print(f"[green]Enroot image: {output_path}[/green]")
    finally:
        os.unlink(script_path)


@app.command("deploy")
def deploy_image(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Container name from config"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Override final image tag"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Override Dockerfile"),
    platform: Optional[str] = typer.Option(
        None, "--platform", help="Target platform (e.g., linux/amd64)"
    ),
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Build-time variables (KEY=VALUE)"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for remote build"),
):
    """Build, push, and deploy a container using the standard multistage pipeline.

    Orchestrates the full deployment workflow:

    1. Build all local stages (default: download) with network access
    2. Export and push each local stage image to the remote cluster
    3. Submit a SLURM job that builds the remote stage (default: build-offline)
       and exports to enroot squashfs

    Examples::

        fcw container deploy app --wait
        fcw container deploy app --tag my-app:v2 --build-arg BASE_IMAGE=python:3.12
    """
    config, system, account = resolve_context(ctx)

    # Resolve container config
    cont_config = None
    if name and name in config.containers:
        cont_config = config.containers[name]
        dockerfile = file or cont_config.file
        final_tag = tag or cont_config.tag
        platform = platform or cont_config.platform
        images_dir = config.resolve_container_images_dir(cont_config)
    elif tag:
        dockerfile = file or "Dockerfile"
        final_tag = tag
        images_dir = config.resolve_path("ce-images", remote=True)
    else:
        _error().print("[red]Specify container name from config or --tag[/red]")
        raise typer.Exit(1)

    if not dockerfile or not os.path.isfile(dockerfile):
        _error().print(f"[red]Dockerfile not found: {dockerfile}[/red]")
        raise typer.Exit(1)

    # Determine local stages and remote stage (from config, else defaults).
    # deploy intentionally uses the configured stages; if build/build-remote
    # ever gain richer stage flags, consider matching them here for consistency.
    if cont_config:
        local_stages = cont_config.get_local_stages()
        remote_stage = cont_config.get_remote_stage()
    else:
        local_stages = ["download"]
        remote_stage = "build-offline"

    sqsh_path = os.path.join(images_dir, f"{final_tag.replace(':', '+')}.sqsh")
    runtime = _detect_container_runtime()

    # Auto-detect remote platform if not specified
    if not platform:
        client = get_client()
        detected = _detect_remote_platform(client, system)
        if detected:
            platform = detected
            _error().print(f"[dim]Detected remote platform: {platform}[/dim]")

    # Merge config-level build_args with CLI --build-arg flags
    all_build_args = _merge_build_args(
        cont_config.build_args if cont_config else None,
        build_arg,
    )

    # Step 1: Build all local stages
    stage_tags = []
    for i, stage_name in enumerate(local_stages, 1):
        if cont_config:
            stage_tag = cont_config.stage_tag(stage_name)
        elif ":" in final_tag:
            stage_tag = f"{final_tag}-{stage_name}"
        else:
            stage_tag = f"{final_tag}:{stage_name}"

        _error().print(
            f"[bold]Step 1.{i}: Building '{stage_name}' stage locally ({stage_tag})...[/bold]"
        )
        _build_one_stage(
            runtime,
            dockerfile=dockerfile,
            tag=stage_tag,
            stage=stage_name,
            platform=platform,
            build_args=all_build_args,
            context=".",  # FIXME: this should default to cont_config.context, i.e. build context needs to be part of the config. And it should also be possible to override the build context from the CLI, e.g. with a --context flag.
        )
        stage_tags.append((stage_name, stage_tag))

    # Step 2: Export and push all local stage images
    for i, (stage_name, stage_tag) in enumerate(stage_tags, 1):
        _error().print(
            f"[bold]Step 2.{i}: Uploading '{stage_name}' image ({stage_tag})...[/bold]"
        )
        _push_one_image(
            stage_tag,
            config=config,
            system=system,
            account=account,
            remote_dir=images_dir,
        )

    # Step 3: Upload Dockerfile and submit remote build job
    _error().print(
        f"[bold]Step 3: Building '{remote_stage}' stage on cluster ({final_tag})...[/bold]"
    )

    staging_dir = _isolated_staging_dir(
        config, "deploy", f"{name or 'deploy'}-{final_tag}"
    )

    async def do_upload_dockerfile():
        async_client = get_async_client()
        try:
            await async_client.mkdir(
                system_name=system, path=staging_dir, create_parents=True
            )
        except Exception:
            pass
        await async_client.upload(
            system_name=system,
            local_file=dockerfile,
            directory=staging_dir,
            filename="Dockerfile",
            account=account,
            blocking=True,
        )

    asyncio.run(do_upload_dockerfile())

    # Generate image loading and build-arg blocks
    load_block, image_build_args = _generate_load_and_resolve_block(
        stage_tags, images_dir
    )

    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in all_build_args)
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""

    q_final_tag = shlex.quote(final_tag)
    q_staging_dir = shlex.quote(staging_dir)
    q_images_dir = shlex.quote(images_dir)
    q_sqsh_path = shlex.quote(sqsh_path)
    global_sbatch = format_sbatch_lines(get_global_sbatch_options())
    setup_block = _podman_setup_block()

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-deploy
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-deploy-%j.out
#SBATCH --error=fcw-container-deploy-%j.out
{global_sbatch}
set -euxo pipefail
{_staging_cleanup_block(staging_dir)}
{setup_block}
export CE_IMAGES_DIR={q_images_dir}

{load_block}

echo "=== Building {q_final_tag} ({remote_stage} stage) ==="
cd {q_staging_dir}
podman build \\
    --target {shlex.quote(remote_stage)} \\
{image_build_args}
{extra_build_args_line}    -t {q_final_tag} \\
    -f Dockerfile .

echo "Built image: {q_final_tag}"

echo "=== Exporting to enroot ==="
mkdir -p $(dirname {q_sqsh_path})
rm -f {q_sqsh_path}
enroot import -x mount -o {q_sqsh_path} podman://{final_tag} || true
if [ ! -f {q_sqsh_path} ]; then
    echo "ERROR: enroot import failed - output not found: {q_sqsh_path}"
    exit 1
fi
echo "Exported to: {q_sqsh_path}"
ls -lh {q_sqsh_path}

exit 0
"""

    client = get_client()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=config.workdir.remote,
            script_local_path=script_path,
        )

        job_id = extract_job_id(result)
        print(job_id)
        _error().print(f"[green]Submitted deploy job: {job_id}[/green]")

        if wait:
            _error().print("[dim]Waiting for deploy to complete...[/dim]")
            _wait_and_check(client, system, job_id, "Deploy job")
            _error().print(f"[green]Deployed: {sqsh_path}[/green]")
        else:
            _error().print(f"[dim]Expected output: {sqsh_path}[/dim]")

    finally:
        os.unlink(script_path)


# -----------------------------------------------------------------------------
# Extract / Patch / Update Commands (Code Iteration Workflow)
# -----------------------------------------------------------------------------

@app.command("extract")
def extract_from_image(
    ctx: typer.Context,
    container_name: str = typer.Argument(..., help="Container config name (from fcw.yaml)"),
    container_path: str = typer.Argument(..., help="Path inside container (e.g., /workspace/BrainBERT)"),
    local_dest: str = typer.Argument(..., help="Local destination directory"),
    stage: Optional[str] = typer.Option(
        None, "--stage", help="Stage to extract from (default: 'download' if available, else first local stage)"
    ),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for extraction to complete"),
):
    """Extract files from a container stage's image tarball on the remote cluster.

    Resolves the container and stage from ``fcw.yaml`` (low-level raw image
    tags are no longer accepted — use a container config name). After a
    successful local extraction, writes a sidecar ``<local_dest>.meta.json``
    recording the source stage, container path, and image. Later ``patch``
    and ``rebuild`` use this sidecar to default the bind-mount target and
    to group patches by stage.

    The extraction runs as a remote job that loads the stage tar, creates a
    container, ``podman cp``'s the requested path, and tars it up. The
    archive is then downloaded and unpacked locally.

    Examples:
        fcw container extract app /workspace/BrainBERT ./code
        fcw container extract app /workspace/BrainBERT ./code --stage runtime-download
    """
    config, system, account = resolve_context(ctx)

    container_cfg = _resolve_container_config(config, container_name)
    resolved_stage = _resolve_extract_stage(container_cfg, stage)
    stage_image = container_cfg.stage_tag(resolved_stage)

    # Staging path for extraction (isolated per container/stage/timestamp so
    # parallel extracts on the same remote workdir don't clobber each other).
    staging_dir = _isolated_staging_dir(config, "extract", f"{container_name}-{resolved_stage}")
    archive_name = f"{os.path.basename(container_path.rstrip('/'))}.tar.gz"
    remote_archive = f"{staging_dir}/{archive_name}"

    remote_tar = _resolve_remote_tar(stage_image, config)

    q_stage_image = shlex.quote(stage_image)
    q_remote_tar = shlex.quote(remote_tar)
    q_container_path = shlex.quote(container_path)
    q_staging_dir = shlex.quote(staging_dir)
    q_remote_archive = shlex.quote(remote_archive)
    q_basename = shlex.quote(os.path.basename(container_path.rstrip("/")))

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-extract
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-extract-%j.out
{format_sbatch_lines(get_global_sbatch_options())}
set -euxo pipefail

{_podman_setup_block()}

# Load the stage image from its pushed tar if not already in local storage.
if podman image exists {q_stage_image} 2>/dev/null; then
    EXTRACT_IMAGE={q_stage_image}
elif [ -f {q_remote_tar} ]; then
    echo "Loading image from {q_remote_tar}..."
    podman load -i {q_remote_tar}
    EXTRACT_IMAGE={q_stage_image}
else
    echo "Error: image {q_stage_image} not in storage and no tar at {q_remote_tar}"
    exit 1
fi

echo "Extracting {q_container_path} from $EXTRACT_IMAGE..."

CID=$(podman create "$EXTRACT_IMAGE" /bin/true)
echo "Created container: $CID"

mkdir -p {q_staging_dir}

EXTRACT_TMP=$(mktemp -d)
podman cp "$CID:{container_path}" "$EXTRACT_TMP/"
podman rm "$CID"

cd "$EXTRACT_TMP"
# Archive WITHOUT the basename wrapper podman cp creates: a directory's contents go
# at the archive root (so the local dump root maps directly onto the container path,
# which patch/rebuild require); a single file is kept as-is.
if [ -d {q_basename} ]; then
    tar czf {q_remote_archive} -C {q_basename} .
else
    tar czf {q_remote_archive} {q_basename}
fi
rm -rf "$EXTRACT_TMP"

echo "Extracted to: {q_remote_archive}"
ls -lh {q_remote_archive}
exit 0
"""

    client = get_client()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=config.workdir.remote,
            script_local_path=script_path,
        )

        job_id = extract_job_id(result)
        _error().print(f"[green]Submitted extract job: {job_id}[/green]")

        if wait:
            _error().print("[dim]Waiting for extraction...[/dim]")
            _wait_and_check(client, system, job_id, "Extract job")
            _error().print("[green]Extraction complete[/green]")

            # Download the archive
            _error().print(f"[dim]Downloading {remote_archive}...[/dim]")
            local_archive = os.path.join(local_dest, archive_name)
            os.makedirs(local_dest, exist_ok=True)

            async def do_download():
                async_client = get_async_client()
                await async_client.download(
                    system_name=system,
                    source_path=remote_archive,
                    target_path=local_archive,
                    account=account,
                    blocking=True,
                )

            asyncio.run(do_download())

            # Extract locally
            _error().print(f"[dim]Extracting to {local_dest}...[/dim]")
            subprocess.run(["tar", "xzf", local_archive, "-C", local_dest], check=True)
            os.unlink(local_archive)

            _write_sidecar(
                local_dest,
                stage=resolved_stage,
                container_path=container_path,
                source_image=stage_image,
            )

            _error().print(f"[green]Extracted to {local_dest}[/green]")
            _error().print(f"[dim]Sidecar: {_sidecar_path(local_dest)}[/dim]")
        else:
            print(job_id)
            _error().print("[dim]After job completes, download with:[/dim]")
            _error().print(f"  fcw data download {remote_archive} {local_dest}")
            _error().print(
                "[yellow]Note: sidecar is only written after local extraction; "
                "run `fcw container extract` with --wait to produce it.[/yellow]"
            )

    finally:
        os.unlink(script_path)


def _update_toml_bind_mount(toml_path: Path, bind_mount: str, container_path: str) -> None:
    """Add/replace a bind-mount entry in an existing TOML file. Idempotent per container_path."""
    content = toml_path.read_text()
    old_mount_pattern = rf'"[^"]*:{re.escape(container_path)}"'
    if re.search(old_mount_pattern, content):
        content = re.sub(old_mount_pattern, f'"{bind_mount}"', content)
    elif re.search(r'^mounts\s*=\s*\[', content, re.MULTILINE):
        content = re.sub(r'(mounts\s*=\s*\[)', rf'\1\n    "{bind_mount}",', content)
    else:
        content += f'\nmounts = [\n    "{bind_mount}",\n]\n'
    toml_path.write_text(content)


@app.command("patch")
def patch_container(
    ctx: typer.Context,
    paths: List[str] = typer.Argument(
        ...,
        help="One or more local paths, each optionally with mount target: "
             "'<local>' (reads sidecar for target) or '<local>:<container-path>' (override).",
    ),
    container_name: str = typer.Option(
        ..., "--container", "-c", help="Container config name (from fcw.yaml). Its TOML is updated."
    ),
):
    """Upload patched code dumps and add bind-mount entries to the container's TOML.

    Primary purpose: enable quick iteration by mounting local code dumps over
    directories in a deployed enroot image — no rebuild needed to test.

    For each ``<local-path>``, if the mount target isn't given via
    ``<local>:<container-path>``, it is read from the sidecar
    ``<local>.meta.json`` written by ``fcw container extract``. The TOML file
    resolved from ``containers.<container_name>.toml`` is updated in place
    (bind-mount added/replaced per container_path) and uploaded to the remote.

    ``patch`` never writes or modifies sidecars.

    Examples:
        # Target inferred from sidecar written by `extract`
        fcw container patch --container app ./code

        # Multiple dumps
        fcw container patch -c app ./code ./configs

        # Explicit override of the in-container target
        fcw container patch -c app ./code:/opt/alt/path
    """
    config, system, account = resolve_context(ctx)

    container_cfg = _resolve_container_config(config, container_name)
    if not container_cfg.toml:
        _error().print(
            f"[red]Container {container_name!r} has no `toml:` set in fcw.yaml[/red]\n"
            "[dim]`patch` requires a TOML file to add bind-mount entries to.[/dim]"
        )
        raise typer.Exit(1)

    toml_path = Path(container_cfg.toml)
    if not toml_path.exists():
        _error().print(f"[red]TOML file not found: {toml_path}[/red]")
        raise typer.Exit(1)

    # Resolve each (local_path, container_path) pair up-front; fail fast before uploading.
    resolved: list[tuple[str, str]] = []
    for arg in paths:
        local_path, explicit_target = _parse_patch_arg(arg)
        local_path = os.path.abspath(local_path)
        if not os.path.isdir(local_path):
            _error().print(f"[red]Not a directory: {local_path}[/red]")
            raise typer.Exit(1)

        if explicit_target:
            container_path = explicit_target
        else:
            meta = _read_sidecar(local_path)
            if not meta or "container_path" not in meta:
                _error().print(
                    f"[red]No sidecar at {_sidecar_path(local_path)} and no explicit mount "
                    f"target for {local_path!r}.[/red]\n"
                    "[dim]Use '<local>:<container-path>' to override, or run "
                    "`fcw container extract` to produce a sidecar.[/dim]"
                )
                raise typer.Exit(1)
            container_path = meta["container_path"]

        resolved.append((local_path, container_path))

    # Upload each dump and update the TOML.
    from fcw.commands.data import _upload_directory

    async def upload_all() -> list[str]:
        async_client = get_async_client()
        remote_dirs: list[str] = []
        for local_path, _ in resolved:
            patch_name = os.path.basename(local_path.rstrip("/"))
            remote_patch_dir = config.resolve_path(f".patches/{patch_name}", remote=True)
            _error().print(f"[dim]Uploading {local_path} -> {remote_patch_dir}...[/dim]")
            await _upload_directory(async_client, system, account, local_path, remote_patch_dir)
            # Also mirror the sidecar to remote so `rebuild` can group patches by stage.
            local_sidecar = _sidecar_path(local_path)
            if local_sidecar.exists():
                remote_patch_parent = os.path.dirname(remote_patch_dir)
                await async_client.upload(
                    system_name=system,
                    local_file=str(local_sidecar),
                    directory=remote_patch_parent,
                    filename=local_sidecar.name,
                    account=account,
                    blocking=True,
                )
            remote_dirs.append(remote_patch_dir)
        return remote_dirs

    remote_patch_dirs = asyncio.run(upload_all())

    for (local_path, container_path), remote_patch_dir in zip(resolved, remote_patch_dirs):
        bind_mount = f"{remote_patch_dir}:{container_path}"
        _update_toml_bind_mount(toml_path, bind_mount, container_path)
        patch_name = os.path.basename(remote_patch_dir.rstrip("/"))
        _record_patch_in_index(container_name, patch_name, local_path)
        _error().print(f"[green]+ mount {remote_patch_dir} -> {container_path}[/green]")

    # Upload the updated TOML to remote.
    remote_toml = config.resolve_path(str(toml_path), remote=True)

    async def do_upload_toml():
        async_client = get_async_client()
        remote_toml_dir = os.path.dirname(remote_toml)
        try:
            await async_client.mkdir(
                system_name=system, path=remote_toml_dir, create_parents=True
            )
        except Exception:
            pass
        await async_client.upload(
            system_name=system,
            local_file=str(toml_path),
            directory=remote_toml_dir,
            filename=os.path.basename(remote_toml),
            account=account,
            blocking=True,
        )

    asyncio.run(do_upload_toml())
    _error().print(f"[green]Updated {toml_path} (local + remote)[/green]")
    _error().print(f"[dim]Run with: srun --environment {remote_toml} ...[/dim]")


# -----------------------------------------------------------------------------
# Rebuild (bake patches into new image)
# -----------------------------------------------------------------------------


def _stage_tar_persistence_block(
    cont: ContainerConfig,
    new_tag: str,
    patches_by_stage: dict,
    images_dir: str,
) -> str:
    """Bash: re-tag each local stage under <new_tag>-<stage> and `podman save` it."""
    lines: list[str] = []
    for stage in cont.get_local_stages():
        var = _stage_to_build_arg_name(stage).replace("-", "_")
        # new stage tag uses the new container tag as prefix so future extracts/rebuilds
        # can find the tar at ce-images/<new_tag>+<stage>.tar (mirrors `push` convention).
        if ":" in new_tag:
            new_stage_tag = f"{new_tag}-{stage}"
        else:
            new_stage_tag = f"{new_tag}:{stage}"
        tar_name = new_stage_tag.replace(":", "+").replace("/", "+") + ".tar"
        q_new_tag = shlex.quote(new_stage_tag)
        q_tar = shlex.quote(f"{images_dir}/{tar_name}")
        lines.append(f"""
echo "Persisting stage '{stage}' as {new_stage_tag}..."
podman tag "${var}_ID" {q_new_tag}
podman save -o {q_tar} {q_new_tag}
ls -lh {q_tar}
""")
    return "\n".join(lines)


@app.command("rebuild")
def rebuild_container(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Container name from fcw.yaml (e.g., 'app')"),
    tag: str = typer.Option(..., "--tag", "-t", help="Tag for rebuilt image (e.g., my-app:v2)"),
    build_arg: Optional[List[str]] = typer.Option(
        None, "--build-arg", help="Build-time variables (KEY=VALUE)"
    ),
    default_stage: Optional[str] = typer.Option(
        None, "--default-stage",
        help="Stage assumed for patches missing a sidecar (otherwise an error).",
    ),
    dump: Optional[List[str]] = typer.Option(
        None, "--dump",
        help="Mode B: rebuild from explicit local dump(s) '<local>[:<container>]'. "
             "Bypasses the container TOML. Mutually exclusive with the default (Mode A).",
    ),
    # Future: --target-stage <s> to stop rebuild at a non-final stage for users who
    # want to skip the final build-offline. Deferred — correctness (always produce
    # the final image) is preferred over performance by default.
    enroot: bool = typer.Option(False, "--enroot", help="Convert final image to enroot squashfs"),
    output: Optional[str] = typer.Option(
        None, "--output", "-o", help="Output path for enroot squashfs"
    ),
    cleanup: bool = typer.Option(
        True, "--cleanup/--no-cleanup", help="Remove remote .patches/ after rebuild"
    ),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for job completion"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print SLURM script without submitting"),
) -> None:
    """Bake accumulated patches into a new container image.

    Reads the container's TOML to discover ``.patches/`` bind-mounts added by
    ``container patch``, then submits a SLURM job that applies all patches to
    the target stage image (default: first local stage) and rebuilds the
    remote stage.

    On success, creates a new container entry in fcw.yaml and a new TOML file
    with patch mounts removed. The original container entry is preserved.

    Examples::

        # Dry run: inspect the generated SLURM script
        fcw container rebuild app --tag my-app:v2 --dry-run

        # Full rebuild with enroot export
        fcw container rebuild app --tag my-app:v2 --enroot --wait
    """
    from fcw.core import add_container_to_config

    config, system, account = resolve_context(ctx)

    # 1. Look up container config
    if name not in config.containers:
        _error().print(f"[red]Unknown container: {name}[/red]")
        raise typer.Exit(1)
    cont = config.containers[name]

    if not cont.file:
        _error().print(f"[red]Container '{name}' has no Dockerfile configured[/red]")
        raise typer.Exit(1)

    mode_b = bool(dump)
    toml_path: Optional[Path] = Path(cont.toml) if cont.toml else None
    if toml_path is not None and not toml_path.exists():
        # TOML referenced in config but missing on disk — only fatal in Mode A.
        if not mode_b:
            _error().print(f"[red]TOML file not found: {cont.toml}[/red]")
            raise typer.Exit(1)
        toml_path = None

    if mode_b:
        # Mode B: patches come from --dump args. No TOML required.
        patch_mounts: list[tuple[str, str]] = []
        stages_for_patches: list[Optional[str]] = []
        from fcw.commands.data import _upload_directory

        async def upload_dumps() -> None:
            async_client = get_async_client()
            for arg in dump or []:
                local_path, explicit_target = _parse_patch_arg(arg)
                local_path = os.path.abspath(local_path)
                if not os.path.isdir(local_path):
                    _error().print(f"[red]Not a directory: {local_path}[/red]")
                    raise typer.Exit(1)
                meta = _read_sidecar(local_path)
                target = explicit_target or (meta.get("container_path") if meta else None)
                if not target:
                    _error().print(
                        f"[red]No sidecar and no explicit target for {local_path!r}. "
                        "Use '<local>:<container-path>'.[/red]"
                    )
                    raise typer.Exit(1)
                patch_name = os.path.basename(local_path.rstrip("/"))
                remote_patch_dir = config.resolve_path(f".patches/{patch_name}", remote=True)
                _error().print(f"[dim]Uploading {local_path} -> {remote_patch_dir}...[/dim]")
                await _upload_directory(
                    async_client, system, account, local_path, remote_patch_dir
                )
                patch_mounts.append((remote_patch_dir, target))
                stages_for_patches.append(meta.get("stage") if meta else None)

        asyncio.run(upload_dumps())
    else:
        # Mode A: read patches from the container's TOML.
        if toml_path is None:
            _error().print(f"[red]Container '{name}' has no TOML file configured[/red]")
            raise typer.Exit(1)
        _resync_container_patches(config, name, system, account)
        patch_mounts = _parse_patch_mounts(toml_path.read_text())
        if not patch_mounts:
            _error().print(
                "[yellow]No .patches/ mounts found in TOML — nothing to rebuild[/yellow]"
            )
            raise typer.Exit(0)

        # _resync_container_patches just re-uploaded each local sidecar, so
        # reading stages locally via the patch index is authoritative and
        # skips N FirecREST round-trips.
        idx = _read_patches_index(name)
        stages_for_patches = []
        for remote_patch_dir, _ in patch_mounts:
            patch_name = os.path.basename(remote_patch_dir.rstrip("/"))
            local_path = idx.get(patch_name)
            if local_path and os.path.isdir(local_path):
                meta = _read_sidecar(local_path)
                stages_for_patches.append(meta.get("stage") if meta else None)
            else:
                stages_for_patches.append(None)

    _error().print(
        f"[bold]Resolved {len(patch_mounts)} patch mount(s) to bake:[/bold]"
    )
    for host_path, container_path in patch_mounts:
        _error().print(f"  {host_path} → {container_path}")

    local_stages = cont.get_local_stages()
    remote_stage = cont.get_remote_stage()

    patches_by_stage: dict[str, list[tuple[str, str]]] = {}
    for (host_path, container_path), stage in zip(patch_mounts, stages_for_patches):
        if stage is None:
            if not default_stage:
                _error().print(
                    f"[red]No remote sidecar for {host_path} and no --default-stage given.[/red]\n"
                    "[dim]Either re-run `fcw container patch` with a dump produced by "
                    "`extract` (which writes a sidecar), or pass --default-stage.[/dim]"
                )
                raise typer.Exit(1)
            stage = default_stage
        if stage not in local_stages:
            _error().print(
                f"[red]Patch stage {stage!r} is not a local stage of container {name!r} "
                f"({local_stages}).[/red]\n"
                "[dim]Only local stages can be patched on the remote (they get loaded from "
                "tars and committed). The remote stage is always rebuilt via podman build.[/dim]"
            )
            raise typer.Exit(1)
        patches_by_stage.setdefault(stage, []).append((host_path, container_path))

    _error().print(f"[bold]Patches grouped across {len(patches_by_stage)} stage(s):[/bold]")
    for stage, entries in patches_by_stage.items():
        _error().print(f"  [cyan]{stage}[/cyan]:")
        for h, c in entries:
            _error().print(f"    {h} -> {c}")

    # 4. Resolve paths
    images_dir = config.resolve_container_images_dir(cont)
    staging_dir = _isolated_staging_dir(config, "rebuild", f"{name}-{tag}")
    dockerfile = cont.file

    # 5. Generate load block for ALL local stages (correctness: final build needs all
    #    of them as build-args, patched or not).
    stage_tag_pairs = [(s, cont.stage_tag(s)) for s in local_stages]
    load_block, build_arg_lines = _generate_load_and_resolve_block(stage_tag_pairs, images_dir)

    # Per-stage patch-and-commit block. Overwrites <STAGE>_IMAGE_ID with the
    # patched image's ID so the final build-arg lines pick it up unchanged.
    patch_commit_lines: list[str] = []
    new_tag_suffix = _sanitize_tag_suffix(tag)
    for stage, entries in patches_by_stage.items():
        var = _stage_to_build_arg_name(stage).replace("-", "_")
        patched_tag = f"{cont.stage_tag(stage)}-patched-{new_tag_suffix}"
        q_patched_tag = shlex.quote(patched_tag)
        cp_lines = "\n".join(
            f'podman cp {shlex.quote(h)}/. "$CID:{c}"' for h, c in entries
        )
        patch_commit_lines.append(f"""
echo "=== Patching stage '{stage}' ({len(entries)} mount(s)) ==="
CID=$(podman create "${var}_ID" /bin/true)
{cp_lines}
podman commit "$CID" {q_patched_tag}
podman rm "$CID"
{var}_ID=$(podman image inspect --format '{{{{.Id}}}}' {q_patched_tag})
echo "Committed patched stage '{stage}' as {patched_tag} (ID: ${var}_ID)"
""")
    patch_commit_block = "\n".join(patch_commit_lines)

    all_build_args = _merge_build_args(cont.build_args, build_arg)
    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in all_build_args)
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""

    q_tag = shlex.quote(tag)
    q_staging_dir = shlex.quote(staging_dir)
    q_images_dir = shlex.quote(images_dir)
    global_sbatch = format_sbatch_lines(get_global_sbatch_options())
    setup_block = _podman_setup_block()

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-rebuild
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-rebuild-%j.out
#SBATCH --error=fcw-container-rebuild-%j.out
{global_sbatch}set -euxo pipefail
{_staging_cleanup_block(staging_dir)}
{setup_block}
export CE_IMAGES_DIR={q_images_dir}

# === Load all local stages ===
{load_block}

# === Apply patches per stage (overrides the corresponding *_IMAGE_ID var) ===
{patch_commit_block}

echo "=== Rebuilding {remote_stage} stage ==="
cd {q_staging_dir}
podman build --target {shlex.quote(remote_stage)} \\
{build_arg_lines}
{extra_build_args_line}    -t {q_tag} \\
    -f Dockerfile .

echo "Built image: {q_tag}"

# === Persist per-stage tars for the new version so future rebuilds can load them ===
# For chain-rebuilds (v1 -> v2 -> v3) to work, every local stage of the new version
# needs a tar at ce-images/<new-tag>+<stage>.tar. Patched stages save their committed
# image; unpatched stages are retagged and saved so the naming is uniform.
mkdir -p {q_images_dir}
{_stage_tar_persistence_block(cont, tag, patches_by_stage, images_dir)}
"""
    if enroot:
        output_path = output or os.path.join(
            images_dir, f"{tag.replace(':', '+')}.sqsh"
        )
        q_output_path = shlex.quote(output_path)
        script += f"""
echo "=== Exporting to enroot ==="
mkdir -p $(dirname {q_output_path})
rm -f {q_output_path}
enroot import -x mount -o {q_output_path} podman://{tag} || true
if [ ! -f {q_output_path} ]; then
    echo "ERROR: enroot import failed - output not found: {q_output_path}"
    exit 1
fi
echo "Exported to: {q_output_path}"
ls -lh {q_output_path}
"""
    script += "\nexit 0\n"

    # 6. Dry run: print and return (no remote operations)
    if dry_run:
        new_name = _derive_container_name(name, cont.tag, tag)
        _error().print("[bold]Generated SLURM script:[/bold]")
        _error().print(script)
        _error().print(f"\n[dim]Would create container entry '{new_name}' in fcw.yaml[/dim]")
        if toml_path is not None:
            new_toml_path = _derive_rebuilt_toml_path(toml_path, cont.tag, tag)
            _error().print(f"[dim]Would create TOML: {new_toml_path}[/dim]")
        else:
            _error().print("[dim]No TOML derivation (Mode B without source TOML).[/dim]")
        return

    # 7. Upload Dockerfile to staging dir
    _error().print(f"[bold]Uploading Dockerfile to {staging_dir}...[/bold]")

    async def do_upload() -> None:
        async_client = get_async_client()
        try:
            await async_client.mkdir(
                system_name=system, path=staging_dir, create_parents=True
            )
        except Exception:
            pass  # May already exist
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_error()) as p:
            p.add_task("Uploading Dockerfile...", total=None)
            await async_client.upload(
                system_name=system,
                local_file=dockerfile,
                directory=staging_dir,
                filename="Dockerfile",
                account=account,
                blocking=True,
            )

    asyncio.run(do_upload())
    _error().print("[green]Uploaded Dockerfile[/green]")

    # 8. Submit job
    client = get_client()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
        f.write(script)
        script_path = f.name

    try:
        result = client.submit(
            system_name=system,
            account=account,
            working_dir=config.workdir.remote,
            script_local_path=script_path,
        )

        job_id = extract_job_id(result)
        print(job_id)
        _error().print(f"[green]Submitted rebuild job: {job_id}[/green]")

        # 9. Post-rebuild flow (only if --wait)
        if wait:
            _error().print("[dim]Waiting for rebuild to complete...[/dim]")
            _wait_and_check(client, system, job_id, "Rebuild job")
            _error().print(f"[green]Rebuild complete: {tag}[/green]")
            if enroot:
                _error().print(f"[green]Enroot image: {output_path}[/green]")

            # Create new TOML (without patch mounts, image retargeted) when
            # there's one to derive from.
            new_name = _derive_container_name(name, cont.tag, tag)
            new_toml_str: Optional[str] = None
            new_cont = ContainerConfig(
                file=cont.file,
                tag=tag,
                remote_path=cont.remote_path,
                toml=None,
            )
            if toml_path is not None:
                new_toml_path = _derive_rebuilt_toml_path(toml_path, cont.tag, tag)
                _create_rebuilt_toml(
                    str(toml_path),
                    str(new_toml_path),
                    config.resolve_container_image(new_cont),
                )
                new_toml_str = str(new_toml_path)
                new_cont.toml = new_toml_str
                _error().print(f"[green]Created TOML: {new_toml_path}[/green]")
            if config._config_path is None:
                _error().print("[yellow]Warning: no config file path — skipping config update[/yellow]")
            else:
                add_container_to_config(config._config_path, new_name, new_cont)
                _error().print(f"[green]Added container '{new_name}' to fcw.yaml[/green]")

            # Cleanup remote .patches/ dirs
            if cleanup:
                _error().print("[dim]Cleaning up remote .patches/ directories...[/dim]")

                async def do_cleanup() -> None:
                    async_client = get_async_client()
                    for host_path, _ in patch_mounts:
                        try:
                            await async_client.rm(
                                system_name=system,
                                path=host_path,
                                account=account,
                                blocking=True,
                            )
                        except Exception as e:
                            _error().print(
                                f"[yellow]Warning: could not remove {host_path}: {e}[/yellow]"
                            )

                asyncio.run(do_cleanup())
                _error().print("[green]Remote patches cleaned up[/green]")

            _error().print(
                "\n[bold]To use the rebuilt container, update your job config:[/bold]"
            )
            _error().print(f"  container: {new_name}")
    finally:
        os.unlink(script_path)


# -----------------------------------------------------------------------------
# Listing
# -----------------------------------------------------------------------------

@app.command("gc")
def gc_staging(
    ctx: typer.Context,
    older_than: Optional[int] = typer.Option(
        None, "--older-than",
        help="Only consider dirs older than N days (parsed from dir name timestamp).",
    ),
    all_dirs: bool = typer.Option(
        False, "--all", help="Consider every staging dir regardless of age."
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Actually delete. Without this flag, dry-run only."
    ),
) -> None:
    """List or remove leftover ``.fcw/{rebuild,deploy,extract}`` staging dirs.

    The SLURM cleanup trap only fires on successful job exit; crashed or
    cancelled builds leave their staging dirs behind. This command lists
    them (default) or removes them (with ``--force``).
    """
    config, system, account = resolve_context(ctx)
    if older_than is None and not all_dirs:
        # Default: show everything, no filter.
        all_dirs = True
    cutoff: Optional[datetime] = None
    if older_than is not None:
        cutoff = datetime.now() - timedelta(days=older_than)

    bases = ["rebuild", "deploy", "extract"]

    async def run() -> None:
        client = get_async_client()
        any_candidate = False
        for base in bases:
            base_path = config.resolve_path(f".fcw/{base}", remote=True)
            entries = await _scan_staging_dirs(client, system, base_path)
            if not entries:
                continue
            header_printed = False
            for name, ts in entries:
                if cutoff is not None:
                    if ts is None or ts >= cutoff:
                        continue
                path = f"{base_path}/{name}"
                age_str = (
                    f"age {(datetime.now() - ts).days}d" if ts else "age unknown"
                )
                if not header_printed:
                    _error().print(f"[bold].fcw/{base}/[/bold]")
                    header_printed = True
                any_candidate = True
                if force:
                    try:
                        await client.rm(
                            system_name=system, path=path, account=account,
                            blocking=True,
                        )
                        _error().print(f"  [red]removed[/red] {name} ({age_str})")
                    except Exception as e:
                        _error().print(
                            f"  [yellow]could not remove[/yellow] {name}: {e}"
                        )
                else:
                    _error().print(f"  {name} ({age_str})")
        if not any_candidate:
            _error().print("[dim]No staging dirs matched.[/dim]")
        elif not force:
            _error().print(
                "\n[dim]Dry-run. Re-run with --force to delete.[/dim]"
            )

    asyncio.run(run())


@app.command("list")
def list_images(
    ctx: typer.Context,
    remote: bool = typer.Option(False, "--remote", "-r", help="List images on remote cluster"),
):
    """List container images (local or remote)."""
    if remote:
        config, system, account = resolve_context(ctx)

        # Collect image directories from container configs
        images_dirs: set[str] = set()
        for _name, cont_config in config.containers.items():
            images_dirs.add(config.resolve_container_images_dir(cont_config))
        if not images_dirs:
            images_dirs.add(config.resolve_path("images", remote=True))

        async def do_list():
            client = get_async_client()
            for images_path in sorted(images_dirs):
                try:
                    entries = await client.list_files(
                        system_name=system,
                        path=images_path,
                        recursive=False,
                    )

                    _output().print(f"[bold]Remote images in {images_path}:[/bold]")
                    for entry in entries:
                        name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
                        size = entry.get("size") if isinstance(entry, dict) else getattr(entry, "size", 0)
                        if name and name.endswith(".sqsh"):
                            _output().print(f"  {name}  ({size / 1024 / 1024:.1f} MB)")
                except Exception as e:
                    _error().print(f"[yellow]Could not list remote images in {images_path}: {e}[/yellow]")

        asyncio.run(do_list())
    else:
        # List local images
        runtime = _detect_container_runtime()
        subprocess.run([runtime, "images"])  # FIXME: should proabably offer an option to restrict display to containers in config 

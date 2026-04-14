"""Container build, deployment, and patching commands.

This module provides commands for building container images locally, deploying
them to remote HPC clusters via FirecREST, and a streamlined workflow for
iterating on code without rebuilding the entire image.

Key workflows:

1. **Initial Build & Deploy** (slow, one-time setup):

   Build the download stage locally, push it, then build-offline on the cluster:

       fcw container build --stage download -t my-fcw-app:download .
       fcw container push my-fcw-app:download  # FIXME: does this push the Dockerfile?
       fcw container build-remote my-fcw-app:download \\
           -f env/Dockerfile.prod-multistage -t my-fcw-app:latest \\
           --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \\
           --enroot --wait

2. **Extract Code for Editing**:
   
   Extract code from the download image to edit locally:
   
       fcw container extract my-fcw-app:download /workspace/BrainBERT ./code

3. **Quick Iteration (bind-mount, no rebuild)**:
   
   Upload patched code and generate TOML with bind-mount for srun:
   
       fcw container patch ./code /workspace/BrainBERT --toml env/container.toml
       # Then use: srun --environment env/container.toml ...

4. **Bake Changes (rebuild)**:

   Bake accumulated patches from the container TOML into a new image:

       fcw container rebuild my-fcw-app --tag my-fcw-app:v2 --enroot --wait
"""

from __future__ import annotations

import asyncio
import os
import re
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from fcw.core import (
    SLURM_FAILED_STATES,
    extract_job_id,
    format_sbatch_lines,
    get_async_client,
    get_client,
    get_console,
    get_global_sbatch_options,
    get_system,
    load_config,
    resolve_context,
)

app = typer.Typer(no_args_is_help=True)
_console = get_console  # FIXME: why do we need this indirection instead of just using get_console() directly?


def _wait_and_check(client, system: str, job_id: str, label: str = "Job") -> None:
    """Wait for a job to complete and raise Exit(1) if it failed."""
    job_info = client.wait_for_job(system_name=system, job_id=job_id)
    state = job_info[0]["status"]["state"]
    if isinstance(state, list):
        state = ",".join(state)
    if any(fs in state for fs in SLURM_FAILED_STATES):
        _console().print(f"[red]{label} {job_id} finished with state: {state}[/red]")
        _console().print(f"[dim]Hint: Run `fcw job logs {job_id}` to see output[/dim]")
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
        if "=" in arg:  # FIXME: does this handle both --build-arg KEY=VALUE and --build-arg KEY VALUE correctly?
            k, v = arg.split("=", 1)
            merged[k] = v
        else:
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


def _detect_remote_platform(client, system: str) -> Optional[str]:  # FIXME: This currently doesn't work (seems like a remote permission issue). Could it be alternatively solved with an fcw job run like approach? 
    """Detect the remote system's platform by reading /proc/cpuinfo.

    Returns a platform string like ``linux/arm64`` or ``linux/amd64``,
    or None if detection fails.
    """
    if system in KNOWN_SYSTEMS:
        return KNOWN_SYSTEMS[system]

    try:
        result = client.head(system_name=system, path="/proc/cpuinfo", num_lines=20)
        content = result.get("content") or result.get("output") or ""
        if isinstance(result, str):
            content = result
        content_lower = content.lower()
        if "aarch64" in content_lower or "arm" in content_lower:
            return "linux/arm64"
        if "x86_64" in content_lower or "genuineintel" in content_lower or "authenticamd" in content_lower:
            return "linux/amd64"
    except Exception:
        pass
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


def _create_rebuilt_toml(original_toml_path: str, new_toml_path: str) -> None:
    """Copy a container TOML, removing .patches/ bind-mount lines.

    The original file is not modified. The new file has all mount entries
    containing ``/.patches/`` removed, with trailing-comma cleanup.
    """
    content = Path(original_toml_path).read_text()
    # Remove lines that are .patches/ mount entries
    lines = content.splitlines(keepends=True)
    filtered: list[str] = []
    for line in lines:
        if re.search(r'"[^"]*/.patches/[^"]*:[^"]*"', line):
            continue
        filtered.append(line)
    # Clean up trailing commas before closing bracket: e.g., '    "x",\n]\n'
    result = "".join(filtered)
    result = re.sub(r',(\s*\])', r'\1', result)
    # Clean up empty mounts array (only whitespace/newlines between brackets)
    result = re.sub(r'mounts\s*=\s*\[\s*\]', 'mounts = []', result)
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

    _console().print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _console().print(f"[red]Build failed: {tag}[/red]")
        raise typer.Exit(1)
    _console().print(f"[green]Built image: {tag}[/green]")


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
        _console().print("[red]No Dockerfile specified. Use --file or configure in fcw.yaml.[/red]")
        raise typer.Exit(1)

    runtime = _detect_container_runtime()
    _console().print(f"[dim]Using container runtime: {runtime}[/dim]")

    # Resolve platform: CLI/config > auto-detect from remote
    if not resolved_platform:
        try:
            system = get_system((ctx.obj or {}).get("system"))
            client = get_client()
            detected = _detect_remote_platform(client, system)
            if detected:
                resolved_platform = detected
                _console().print(f"[dim]Detected remote platform: {resolved_platform}[/dim]")
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
            _save_image(runtime, stage_tag, save)
    elif cont_config and not tag:
        # Config name without --stage: build all local_stages
        local_stages = cont_config.get_local_stages()
        for s in local_stages:
            stage_tag = cont_config.stage_tag(s)
            _console().print(f"[bold]Building stage '{s}' → {stage_tag}[/bold]")
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
            # Save the last built stage
            _save_image(runtime, cont_config.stage_tag(local_stages[-1]), save)  # FIXME: need to save all of the built local_stages, not just the last one
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
            _save_image(runtime, resolved_tag, save)


def _save_image(runtime: str, tag: str, path: str) -> None:
    """Export a container image to a tar file."""
    cmd = [runtime, "save", "-o", path, tag]
    _console().print(f"[dim]Saving image to {path}...[/dim]")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _console().print("[red]Failed to save image[/red]")
        raise typer.Exit(1)
    _console().print(f"[green]Saved image to {path}[/green]")


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

    _console().print(f"[dim]Exporting {image_tag}...[/dim]")

    save_cmd = [runtime, "save", "-o", tar_path]
    if platform:
        help_output = subprocess.run(
            [runtime, "save", "--help"], capture_output=True, text=True
        ).stdout
        if "--platform" in help_output:
            save_cmd.extend(["--platform", platform])
        else:
            _console().print(
                f"[yellow]Warning: {runtime} save does not support --platform. "
                "Ignoring platform argument.[/yellow]"
            )
    save_cmd.append(image_tag)

    result = subprocess.run(save_cmd)
    if result.returncode != 0:
        _console().print(f"[red]Failed to export image: {image_tag}[/red]")
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
                console=_console(),
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
        _console().print(f"[green]Uploaded {remote_filename} to {remote_dir}[/green]")
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
            _console().print(f"[bold]Pushing stage '{stage}' → {stage_tag}[/bold]")
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
                _console().print(f"[bold]Pushing stage '{s}' → {stage_tag}[/bold]")
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

        _console().print(f"[dim]Exporting {image}...[/dim]")

        save_cmd = [runtime, "save", "-o", tar_path]
        if platform:
            help_output = subprocess.run(
                [runtime, "save", "--help"], capture_output=True, text=True
            ).stdout
            if "--platform" in help_output:
                save_cmd.extend(["--platform", platform])
            else:
                _console().print(
                    f"[yellow]Warning: {runtime} save does not support --platform. "
                    "Ignoring platform argument.[/yellow]"
                )
        save_cmd.append(image)

        result = subprocess.run(save_cmd)
        if result.returncode != 0:
            _console().print("[red]Failed to export image[/red]")
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
                console=_console(),
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
        _console().print(f"[green]Uploaded to {remote_path}[/green]")

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
        _console().print(f"[green]Submitted import job: {job_id}[/green]")
        _console().print(f"[dim]Output will be: {output_path}[/dim]")

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
            _console().print("[red]--file/-f is required when not using a config name.[/red]")
            raise typer.Exit(1)
        if not tag:
            _console().print("[red]--tag/-t is required when not using a config name.[/red]")
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
        _console().print(f"[red]Dockerfile not found: {resolved_dockerfile}[/red]")
        raise typer.Exit(1)

    staging_dir = config.resolve_path(".fcw/build-remote", remote=True)

    if enroot:
        output_path = output or os.path.join(
            images_dir, f"{resolved_tag.replace(':', '+')}.sqsh"
        )

    # Step 1: Upload Dockerfile
    _console().print("[bold]Step 1: Uploading Dockerfile...[/bold]")

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
    _console().print("[bold]Step 2: Submitting build job...[/bold]")

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
        _console().print(f"[green]Submitted build job: {job_id}[/green]")

        if wait:
            _console().print("[dim]Waiting for build to complete...[/dim]")
            _wait_and_check(client, system, job_id, "Build job")
            _console().print(f"[green]Build complete: {resolved_tag}[/green]")
            if enroot:
                _console().print(f"[green]Enroot image: {output_path}[/green]")
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
        _console().print("[red]Specify container name from config or --tag[/red]")
        raise typer.Exit(1)

    if not dockerfile or not os.path.isfile(dockerfile):
        _console().print(f"[red]Dockerfile not found: {dockerfile}[/red]")
        raise typer.Exit(1)

    # Determine local stages and remote stage # FIXME: make these overridable from CLI
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
            _console().print(f"[dim]Detected remote platform: {platform}[/dim]")

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

        _console().print(
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
        _console().print(
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
    _console().print(
        f"[bold]Step 3: Building '{remote_stage}' stage on cluster ({final_tag})...[/bold]"
    )

    staging_dir = config.resolve_path(".fcw/deploy", remote=True)  # FIXME: Don't all the Dockerfiles overwrite each other in the staging dir like this?

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
            filename="Dockerfile",  # FIXME: shouldn't this be dockerfile?
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
        _console().print(f"[green]Submitted deploy job: {job_id}[/green]")

        if wait:
            _console().print("[dim]Waiting for deploy to complete...[/dim]")
            _wait_and_check(client, system, job_id, "Deploy job")
            _console().print(f"[green]Deployed: {sqsh_path}[/green]")
        else:
            _console().print(f"[dim]Expected output: {sqsh_path}[/dim]")

    finally:
        os.unlink(script_path)


# -----------------------------------------------------------------------------
# Extract / Patch / Update Commands (Code Iteration Workflow)
# -----------------------------------------------------------------------------

@app.command("extract")
def extract_from_image(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Source image (e.g., my-fcw-app:download)"),
    container_path: str = typer.Argument(..., help="Path inside container (e.g., /workspace/BrainBERT)"),
    local_dest: str = typer.Argument(..., help="Local destination directory"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for extraction to complete"),
):
    """Extract files from a remote container image.
    
    This is the first step in the code iteration workflow. It extracts code
    from an existing container image so you can edit it locally.
    
    The extraction runs as a job on the remote cluster:
    1. Creates a container from the image (without running it)
    2. Copies the specified path to a staging area
    3. Creates a tar archive
    
    After the job completes, use ``fcw data download`` to fetch the archive.
    
    Examples:
        # Extract BrainBERT code from download stage
        fcw container extract my-fcw-app:download /workspace/BrainBERT ./code
        
        # Extract to current directory
        fcw container extract my-fcw-app:download /workspace/BrainBERT .
    """
    config, system, account = resolve_context(ctx)

    # Staging path for extraction
    staging_dir = config.resolve_path(".fcw/extract", remote=True)
    archive_name = f"{os.path.basename(container_path.rstrip('/'))}.tar.gz"
    remote_archive = f"{staging_dir}/{archive_name}"

    # Resolve the pushed tar path so the job can load it if needed.
    # With the multistage deploy workflow, the pushed tar is the download
    # stage (e.g., fcw-aux+latest-download.tar), not the final tag. Try both.
    remote_tar = _resolve_remote_tar(image, config)
    if ":" in image:  # FIXME: this applies download again even though the help message and examples already suggest to use the download tag. It would be simpler, if the user could just specify the stage and then the stage to tag mapping is done by fcw consistent with the deploy workflow. Also, it should be possible to refer to a container config and not need to specify the fully qualified image name (this is more sort of the legacy interface)
        download_tag = f"{image}-download"
    else:
        download_tag = f"{image}:download"
    remote_download_tar = _resolve_remote_tar(download_tag, config)

    q_image = shlex.quote(image)
    q_remote_tar = shlex.quote(remote_tar)
    q_remote_download_tar = shlex.quote(remote_download_tar)
    q_container_path = shlex.quote(container_path)
    q_staging_dir = shlex.quote(staging_dir)
    q_remote_archive = shlex.quote(remote_archive)

    q_download_tag = shlex.quote(download_tag)  # FIXME: extracting this directly can't work on the remote machine won't work as it has to first be loaded from a tar

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-extract
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-extract-%j.out
{format_sbatch_lines(get_global_sbatch_options())}
set -euxo pipefail

{_podman_setup_block()}

# Determine which image to use for extraction.
# Prefer the download stage since extract operates on the download stage,
# not the final build-offline image.
EXTRACT_IMAGE=""
if podman image exists {q_download_tag} 2>/dev/null; then
    EXTRACT_IMAGE={q_download_tag}
elif podman image exists {q_image} 2>/dev/null; then
    EXTRACT_IMAGE={q_image}
elif [ -f {q_remote_download_tar} ]; then
    echo "Loading download image from {q_remote_download_tar}..."
    podman load -i {q_remote_download_tar}
    EXTRACT_IMAGE={q_download_tag}
elif [ -f {q_remote_tar} ]; then
    echo "Loading image from {q_remote_tar}..."
    podman load -i {q_remote_tar}
    EXTRACT_IMAGE={q_image}
else
    echo "Error: image not found and no tar at {q_remote_download_tar} or {q_remote_tar}"
    exit 1
fi

echo "Extracting {q_container_path} from $EXTRACT_IMAGE..."

# Create container from the download stage (don't run it)
CID=$(podman create "$EXTRACT_IMAGE" /bin/true)
echo "Created container: $CID"

# Create staging directory
mkdir -p {q_staging_dir}

# Copy files out of container
EXTRACT_TMP=$(mktemp -d)
podman cp "$CID:{container_path}" "$EXTRACT_TMP/"
podman rm "$CID"

# Create archive
cd "$EXTRACT_TMP"
tar czf {q_remote_archive} *
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
        _console().print(f"[green]Submitted extract job: {job_id}[/green]")

        if wait:
            _console().print("[dim]Waiting for extraction...[/dim]")
            _wait_and_check(client, system, job_id, "Extract job")
            _console().print("[green]Extraction complete[/green]")

            # Download the archive
            _console().print(f"[dim]Downloading {remote_archive}...[/dim]")
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
            _console().print(f"[dim]Extracting to {local_dest}...[/dim]")
            subprocess.run(["tar", "xzf", local_archive, "-C", local_dest], check=True)
            os.unlink(local_archive)

            _console().print(f"[green]Extracted to {local_dest}[/green]")
        else:
            print(job_id)
            _console().print("[dim]After job completes, download with:[/dim]")
            _console().print(f"  fcw data download .fcw/extract/{archive_name} {local_dest}")

    finally:
        os.unlink(script_path)


@app.command("patch")
def patch_container(  # FIXME: this is still only for the low-level interface, can't refer to container by config name/stage
    ctx: typer.Context,
    local_path: str = typer.Argument(..., help="Local directory with patched code"),
    container_path: str = typer.Argument(..., help="Target path inside container"),
    toml: Optional[str] = typer.Option(None, "--toml", help="TOML file to update with bind-mount"),
    create_toml: bool = typer.Option(False, "--create", help="Create new TOML file if it doesn't exist"),  # FIXME why would this be needed?
):
    """Upload patched code and configure bind-mount for quick iteration.
    
    This command uploads your modified code to the remote cluster and
    optionally updates a container TOML file with a bind-mount configuration.
    This allows fast iteration without rebuilding the container image.
    
    The patched code is uploaded to ``$WORKDIR/.patches/<dirname>`` and the
    TOML file is updated to bind-mount this over the original container path.
    
    Examples:
        # Upload code and update TOML
        fcw container patch ./code /workspace/BrainBERT --toml env/container.toml
        
        # Then run with the patched container:
        # srun --environment env/container.toml python train.py
        
        # Just upload (no TOML update)
        fcw container patch ./code /workspace/BrainBERT
    """
    config, system, account = resolve_context(ctx)

    local_path = os.path.abspath(local_path)
    if not os.path.isdir(local_path):
        _console().print(f"[red]Not a directory: {local_path}[/red]")
        raise typer.Exit(1)

    # Determine remote patch directory
    patch_name = os.path.basename(local_path.rstrip("/"))
    remote_patch_dir = config.resolve_path(f".patches/{patch_name}", remote=True)

    # Upload the patched code
    _console().print(f"[dim]Uploading {local_path} to {remote_patch_dir}...[/dim]")

    async def do_upload():
        from fcw.commands.data import _upload_directory
        async_client = get_async_client()
        await _upload_directory(async_client, system, account, local_path, remote_patch_dir)

    asyncio.run(do_upload())
    _console().print(f"[green]Uploaded to {remote_patch_dir}[/green]")

    # Update TOML file if specified
    bind_mount = f"{remote_patch_dir}:{container_path}"
    if toml:
        toml_path = Path(toml)
        if not toml_path.exists():
            if create_toml:
                _console().print(f"[dim]Creating {toml}...[/dim]")
                toml_content = f'''\
# Container environment configuration
# Generated by fcw container patch

mounts = [
    "{bind_mount}",
]
'''
                toml_path.parent.mkdir(parents=True, exist_ok=True)
                toml_path.write_text(toml_content)
            else:
                _console().print(f"[red]TOML file not found: {toml}[/red]")
                _console().print("[dim]Use --create to create a new file[/dim]")
                raise typer.Exit(1)
        else:
            content = toml_path.read_text()

            # Check if a mount to this container_path already exists
            old_mount_pattern = rf'"[^"]*:{re.escape(container_path)}"'
            if re.search(old_mount_pattern, content):
                # Replace existing bind-mount entry
                content = re.sub(old_mount_pattern, f'"{bind_mount}"', content)
            elif re.search(r'^mounts\s*=\s*\[', content, re.MULTILINE):
                # Append to existing mounts array
                content = re.sub(
                    r'(mounts\s*=\s*\[)',
                    rf'\1\n    "{bind_mount}",',
                    content,
                )
            else:
                # No mounts array — add one
                content += f'\nmounts = [\n    "{bind_mount}",\n]\n'

            toml_path.write_text(content)

        # Upload TOML to remote
        remote_toml = config.resolve_path(toml, remote=True)

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
        _console().print(f"[green]Updated {toml} (local + remote)[/green]")
        _console().print(f"[dim]Run with: srun --environment {remote_toml} ...[/dim]")
    else:
        _console().print("[dim]To use, add to your container TOML:[/dim]")
        _console().print('  [mounts]')
        _console().print(f'  "{remote_patch_dir}" = "{container_path}"')


@app.command("update", hidden=True, deprecated=True)
def update_image(ctx: typer.Context) -> None:
    """Removed: use ``patch`` + ``rebuild`` instead."""
    _console().print(
        "[red]`fcw container update` has been removed.[/red]\n"
        "Use [bold]fcw container patch[/bold] to stage code changes and "
        "[bold]fcw container rebuild[/bold] to bake them into a new image."
    )
    raise typer.Exit(2)


# -----------------------------------------------------------------------------
# Rebuild (bake patches into new image)
# -----------------------------------------------------------------------------


@app.command("rebuild")
def rebuild_container(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Container name from fcw.yaml (e.g., 'app')"),
    tag: str = typer.Option(..., "--tag", "-t", help="Tag for rebuilt image (e.g., my-app:v2)"),
    build_arg: Optional[List[str]] = typer.Option(
        None, "--build-arg", help="Build-time variables (KEY=VALUE)"
    ),
    patch_stage: Optional[str] = typer.Option(
        None, "--patch-stage", help="Stage to apply patches to (default: first local stage)"
    ),
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
    from fcw.core import ContainerConfig, add_container_to_config

    config, system, account = resolve_context(ctx)

    # 1. Look up container config
    if name not in config.containers:
        _console().print(f"[red]Unknown container: {name}[/red]")
        raise typer.Exit(1)
    cont = config.containers[name]

    if not cont.toml:
        _console().print(f"[red]Container '{name}' has no TOML file configured[/red]")
        raise typer.Exit(1)
    if not cont.file:
        _console().print(f"[red]Container '{name}' has no Dockerfile configured[/red]")
        raise typer.Exit(1)

    toml_path = Path(cont.toml)
    if not toml_path.exists():
        _console().print(f"[red]TOML file not found: {cont.toml}[/red]")
        raise typer.Exit(1)

    # 2. Parse patch mounts
    toml_content = toml_path.read_text()
    patch_mounts = _parse_patch_mounts(toml_content)
    if not patch_mounts:
        _console().print("[yellow]No .patches/ mounts found in TOML — nothing to rebuild[/yellow]")
        raise typer.Exit(0)

    _console().print(
        f"[bold]Found {len(patch_mounts)} patch mount(s) to bake into image:[/bold]"
    )
    for host_path, container_path in patch_mounts:
        _console().print(f"  {host_path} → {container_path}")

    # 3. Derive stage tag for the image to patch
    target_stage = patch_stage or cont.get_local_stages()[0]  # FIXME: need a systematic way to map patches to the local stage they were extracted from, otherwise prefer to make patch_stage a required argument
    stage_tag = cont.stage_tag(target_stage)
    remote_stage = cont.get_remote_stage()

    # 4. Resolve paths
    images_dir = config.resolve_container_images_dir(cont)
    remote_stage_tar = _resolve_remote_tar(stage_tag, config)
    staging_dir = config.resolve_path(".fcw/rebuild", remote=True)
    dockerfile = cont.file

    # 5. Generate SLURM script
    all_build_args = _merge_build_args(cont.build_args, build_arg)
    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in all_build_args)
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""

    q_stage_tag = shlex.quote(stage_tag)
    q_remote_stage_tar = shlex.quote(remote_stage_tar)
    q_tag = shlex.quote(tag)
    q_staging_dir = shlex.quote(staging_dir)
    q_images_dir = shlex.quote(images_dir)
    global_sbatch = format_sbatch_lines(get_global_sbatch_options())
    setup_block = _podman_setup_block()

    # Build podman cp lines for each patch
    patch_cp_lines = []
    for host_path, container_path in patch_mounts:
        q_host = shlex.quote(host_path)
        patch_cp_lines.append(f'podman cp {q_host}/. "$CID:{container_path}"')
    patch_cp_block = "\n".join(patch_cp_lines)

    build_arg_name = _stage_to_build_arg_name(target_stage)

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-rebuild
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-rebuild-%j.out
#SBATCH --error=fcw-container-rebuild-%j.out
{global_sbatch}set -euxo pipefail

{setup_block}
export CE_IMAGES_DIR={q_images_dir}

# Load {target_stage} stage image from tar
if ! podman image exists {q_stage_tag} 2>/dev/null; then
    if [ -f {q_remote_stage_tar} ]; then
        echo "Loading image from {q_remote_stage_tar}..."
        podman load -i {q_remote_stage_tar}
    else
        echo "Error: image {q_stage_tag} not found and no tar at {q_remote_stage_tar}"
        exit 1
    fi
fi

IMAGE_ID=$(podman image inspect --format '{{{{.Id}}}}' docker.io/library/{stage_tag} 2>/dev/null || podman image inspect --format '{{{{.Id}}}}' {stage_tag})
echo "Resolved {target_stage} image ID: $IMAGE_ID"

echo "=== Baking {len(patch_mounts)} patch(es) into {target_stage} image ==="
CID=$(podman create $IMAGE_ID /bin/true)
{patch_cp_block}
PATCHED_IMAGE="{stage_tag}-patched"
podman commit "$CID" "$PATCHED_IMAGE"
podman rm "$CID"
PATCHED_ID=$(podman image inspect --format '{{{{.Id}}}}' "$PATCHED_IMAGE")
echo "Committed patched image: $PATCHED_IMAGE (ID: $PATCHED_ID)"

echo "=== Rebuilding {remote_stage} stage ==="
cd {q_staging_dir}
podman build --target {shlex.quote(remote_stage)} \\
    --build-arg {build_arg_name}=$PATCHED_ID \\
{extra_build_args_line}    -t {q_tag} \\
    -f Dockerfile .

echo "Built image: {q_tag}"
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
        new_toml_path = _derive_rebuilt_toml_path(toml_path, cont.tag, tag)
        _console().print("[bold]Generated SLURM script:[/bold]")
        _console().print(script)
        _console().print(f"\n[dim]Would create container entry '{new_name}' in fcw.yaml[/dim]")
        _console().print(f"[dim]Would create TOML: {new_toml_path}[/dim]")
        return

    # 7. Upload Dockerfile to staging dir
    _console().print(f"[bold]Uploading Dockerfile to {staging_dir}...[/bold]")

    async def do_upload() -> None:
        async_client = get_async_client()
        try:
            await async_client.mkdir(
                system_name=system, path=staging_dir, create_parents=True
            )
        except Exception:
            pass  # May already exist
        with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=_console()) as p:
            p.add_task("Uploading Dockerfile...", total=None)
            await async_client.upload(
                system_name=system,
                local_file=dockerfile,
                directory=staging_dir,
                filename="Dockerfile",  # FIXME: doesn't this cause conflicts if multiple containers are rebuilt at the same time?
                account=account,
                blocking=True,
            )

    asyncio.run(do_upload())
    _console().print("[green]Uploaded Dockerfile[/green]")

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
        _console().print(f"[green]Submitted rebuild job: {job_id}[/green]")

        # 9. Post-rebuild flow (only if --wait)
        if wait:
            _console().print("[dim]Waiting for rebuild to complete...[/dim]")
            _wait_and_check(client, system, job_id, "Rebuild job")
            _console().print(f"[green]Rebuild complete: {tag}[/green]")
            if enroot:
                _console().print(f"[green]Enroot image: {output_path}[/green]")

            # Create new TOML (without patch mounts)
            new_name = _derive_container_name(name, cont.tag, tag)
            new_toml_path = _derive_rebuilt_toml_path(toml_path, cont.tag, tag)
            _create_rebuilt_toml(str(toml_path), str(new_toml_path))
            _console().print(f"[green]Created TOML: {new_toml_path}[/green]")

            # Add new container entry to fcw.yaml
            new_cont = ContainerConfig(
                file=cont.file,
                tag=tag,
                remote_path=cont.remote_path,
                stage=cont.stage,
                toml=str(new_toml_path),
            )
            if config._config_path is None:
                _console().print("[yellow]Warning: no config file path — skipping config update[/yellow]")
            else:
                add_container_to_config(config._config_path, new_name, new_cont)
                _console().print(f"[green]Added container '{new_name}' to fcw.yaml[/green]")

            # Cleanup remote .patches/ dirs
            if cleanup:
                _console().print("[dim]Cleaning up remote .patches/ directories...[/dim]")

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
                            _console().print(
                                f"[yellow]Warning: could not remove {host_path}: {e}[/yellow]"
                            )

                asyncio.run(do_cleanup())
                _console().print("[green]Remote patches cleaned up[/green]")

            _console().print(
                "\n[bold]To use the rebuilt container, update your job config:[/bold]"
            )
            _console().print(f"  container: {new_name}")
    finally:
        os.unlink(script_path)


# -----------------------------------------------------------------------------
# Listing
# -----------------------------------------------------------------------------

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

                    _console().print(f"[bold]Remote images in {images_path}:[/bold]")
                    for entry in entries:
                        name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
                        size = entry.get("size") if isinstance(entry, dict) else getattr(entry, "size", 0)
                        if name and name.endswith(".sqsh"):
                            _console().print(f"  {name}  ({size / 1024 / 1024:.1f} MB)")
                except Exception as e:
                    _console().print(f"[yellow]Could not list remote images in {images_path}: {e}[/yellow]")

        asyncio.run(do_list())
    else:
        # List local images
        runtime = _detect_container_runtime()
        subprocess.run([runtime, "images"])  # FIXME: should proabably offer an option to restrict display to containers in config 

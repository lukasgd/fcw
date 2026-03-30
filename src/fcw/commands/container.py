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
   
   Extract code from the download image to edit locally:
   
       fcw container extract my-fcw-app:download /workspace/BrainBERT ./code

3. **Quick Iteration (bind-mount, no rebuild)**:
   
   Upload patched code and generate TOML with bind-mount for srun:
   
       fcw container patch ./code /workspace/BrainBERT --toml env/container.toml
       # Then use: srun --environment env/container.toml ...

4. **Bake Changes (rebuild)**:
   
   Patch the download image and rebuild build-offline:
   
       fcw container update ./code my-fcw-app:download /workspace/BrainBERT \\
           --tag my-fcw-app:v2 --rebuild --enroot --wait
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from fcw.core import (
    load_config,
    get_client,
    get_async_client,
    get_system,
    get_account,
    extract_job_id,
    resolve_context,
    get_console,
    get_global_sbatch_options,
    format_sbatch_lines,
    SLURM_FAILED_STATES,
)

app = typer.Typer(no_args_is_help=True)
_console = get_console


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


def _podman_setup_block() -> str:
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

def _detect_container_runtime() -> str:
    """Detect available container runtime (podman or docker)."""
    for runtime in ["podman", "docker"]:
        try:
            subprocess.run([runtime, "--version"], capture_output=True, check=True)
            return runtime
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    raise RuntimeError("No container runtime found. Install podman or docker.")


@app.command("build")
def build_image(
    ctx: typer.Context,
    context: str = typer.Argument(".", help="Build context directory"),
    file: Optional[str] = typer.Option(None, "--file", "-f", help="Dockerfile path"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Image tag"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Build specific stage only"),
    platform: Optional[str] = typer.Option(None, "--platform", help="Target platform (e.g., linux/arm64)"),
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Set build-time variables (KEY=VALUE)"),
    offline: bool = typer.Option(False, "--offline", help="Build with pre-downloaded dependencies"),
    save: Optional[str] = typer.Option(None, "--save", "-o", help="Save image to tar file"),
):
    """Build a container image locally.
    
    This builds the image using the local container runtime (podman or docker).
    Use --stage to build only a specific stage (e.g., for offline builds).
    Use --save to export the image to a tar file for transfer.
    """
    config = load_config((ctx.obj or {}).get("config_file"))

    runtime = _detect_container_runtime()
    _console().print(f"[dim]Using container runtime: {runtime}[/dim]")
    
    # Build command
    cmd = [runtime, "build"]
    
    if file:
        cmd.extend(["-f", file])
    
    if tag:
        cmd.extend(["-t", tag])
    elif config.containers:
        # Use first container from config if no tag specified
        first_container = next(iter(config.containers.values()))
        tag = first_container.tag
        cmd.extend(["-t", tag])
    
    if stage:
        cmd.extend(["--target", stage])
    
    if platform:
        cmd.extend(["--platform", platform])

    if build_arg:
        for arg in build_arg:
            cmd.extend(["--build-arg", arg])

    cmd.append(context)
    
    _console().print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        _console().print("[red]Build failed[/red]")
        raise typer.Exit(1)
    
    _console().print(f"[green]Built image: {tag}[/green]")
    
    # Save if requested
    if save:
        save_cmd = [runtime, "save", "-o", save, tag]
        _console().print(f"[dim]Saving image to {save}...[/dim]")
        result = subprocess.run(save_cmd)
        if result.returncode != 0:
            _console().print("[red]Failed to save image[/red]")
            raise typer.Exit(1)
        _console().print(f"[green]Saved image to {save}[/green]")


@app.command("push")
def push_image(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Image tar file or tag to push"),
    remote_path: Optional[str] = typer.Option(None, "--to", help="Remote path (default: from config)"),
    do_import: bool = typer.Option(False, "--import", help="Import to squashfs after push"),
):
    """Upload a container image to remote storage.

    If image is a tar file, uploads directly.
    If image is a tag, exports and uploads.
    Use --import to also submit an import job to convert to squashfs.
    """
    config, system, account = resolve_context(ctx)
    
    # Determine remote path
    if remote_path is None:
        cont_config = _find_container_config(config, image)
        if cont_config and cont_config.remote_path:
            remote_path = cont_config.remote_path
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
        result = subprocess.run([runtime, "save", "-o", tar_path, image])
        if result.returncode != 0:
            _console().print("[red]Failed to export image[/red]")
            raise typer.Exit(1)

    try:
        # Upload
        async def do_upload():
            client = get_async_client()

            # Ensure remote directory exists
            target_dir = os.path.dirname(remote_path)
            try:
                await client.mkdir(system_name=system, path=target_dir, create_parents=True)
            except Exception:
                pass  # May already exist

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
        # Invoke import using the tag (resolved to remote tar path)
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
    """Import a container image on the remote cluster.

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


@app.command("build-remote")
def build_remote(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Base image to build from (must be pushed)"),
    dockerfile: str = typer.Option(..., "--file", "-f", help="Local Dockerfile path"),
    tag: str = typer.Option(..., "--tag", "-t", help="Tag for the built image"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Target stage to build"),
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Build-time variables (KEY=VALUE)"),
    enroot: bool = typer.Option(False, "--enroot", help="Convert final image to enroot squashfs"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for enroot squashfs"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for job completion"),
):
    """Build a container image on the remote cluster via a SLURM job.

    Uploads the Dockerfile, then submits a job that loads the base image
    from its pushed tar, runs podman build, and optionally exports to
    enroot squashfs.

    Examples:
        # Build build-offline stage from pushed download image
        fcw container build-remote my-fcw-app:download \\
            -f env/Dockerfile.prod-multistage -t my-fcw-app:latest \\
            --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \\
            --enroot --wait
    """
    config, system, account = resolve_context(ctx)

    if not os.path.isfile(dockerfile):
        _console().print(f"[red]Dockerfile not found: {dockerfile}[/red]")
        raise typer.Exit(1)

    # Resolve paths
    remote_image_tar = _resolve_remote_tar(image, config)
    staging_dir = config.resolve_path(".fcw/build-remote", remote=True)
    remote_dockerfile = f"{staging_dir}/Dockerfile"

    # Resolve images directory from config if available
    cont_config = _find_container_config(config, tag) or _find_container_config(config, image)
    images_dir = config.resolve_container_images_dir(cont_config) if cont_config else config.resolve_path("ce-images/", remote=True)

    if enroot:
        output_path = output or os.path.join(images_dir, f"{tag.replace(':', '+')}.sqsh")

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
            local_file=dockerfile,
            directory=staging_dir,
            filename="Dockerfile",
            account=account,
            blocking=True,
        )

    asyncio.run(do_upload())

    # Step 2: Build job script
    _console().print("[bold]Step 2: Submitting build job...[/bold]")

    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in (build_arg or []))
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""
    target_line = f"    --target {shlex.quote(stage)} \\\n" if stage else ""

    q_image = shlex.quote(image)
    q_tag = shlex.quote(tag)
    q_remote_image_tar = shlex.quote(remote_image_tar)
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

# Load base image from tar
if ! podman image exists {q_image} 2>/dev/null; then
    if [ -f {q_remote_image_tar} ]; then
        echo "Loading image from {q_remote_image_tar}..."
        podman load -i {q_remote_image_tar}
    else
        echo "Error: image {q_image} not found and no tar at {q_remote_image_tar}"
        exit 1
    fi
fi

# Get the image ID for reliable reference in FROM directives
IMAGE_ID=$(podman image inspect --format '{{{{.Id}}}}' docker.io/library/{image} 2>/dev/null || podman image inspect --format '{{{{.Id}}}}' {image})
echo "Resolved image ID: $IMAGE_ID"

echo "=== Building {q_tag} ==="
cd {q_staging_dir}
podman build \\
{target_line}    --build-arg DOWNLOAD_IMAGE=$IMAGE_ID \\
{extra_build_args_line}    -t {q_tag} \\
    -f Dockerfile .

echo "Built image: {q_tag}"
"""

    if enroot:
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
            _console().print(f"[green]Build complete: {tag}[/green]")
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
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Build-time variables (KEY=VALUE)"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for remote build"),
):
    """Build, push, and deploy a container using the standard multistage pipeline.

    Orchestrates the full deployment workflow:

    1. Build the ``download`` stage locally (has network access)
    2. Export and push the download image to the remote cluster
    3. Submit a SLURM job that builds the ``build-offline`` stage and exports to enroot

    The Dockerfile must follow the multistage pattern with ``download`` and
    ``build-offline`` stages (see ``examples/basic/env/Dockerfile.app``).

    Examples:
        fcw container deploy app --wait
        fcw container deploy app --tag my-app:v2 --build-arg BASE_IMAGE=python:3.12
    """
    config, system, account = resolve_context(ctx)

    # Resolve container config
    if name and name in config.containers:
        cont_config = config.containers[name]
        dockerfile = file or cont_config.file
        final_tag = tag or cont_config.tag
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

    # Derive download tag: <image>:<version>-download
    if ":" in final_tag:
        download_tag = f"{final_tag}-download"
    else:
        download_tag = f"{final_tag}:download"

    sqsh_path = os.path.join(images_dir, f"{final_tag.replace(':', '+')}.sqsh")
    runtime = _detect_container_runtime()

    # Step 1: Build download stage locally
    _console().print(f"[bold]Step 1: Building download stage locally ({download_tag})...[/bold]")

    build_cmd = [runtime, "build", "--target", "download", "-t", download_tag, "-f", dockerfile]
    for arg in (build_arg or []):
        build_cmd.extend(["--build-arg", arg])
    build_cmd.append(".")

    _console().print(f"[dim]Running: {' '.join(build_cmd)}[/dim]")
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        _console().print("[red]Build failed[/red]")
        raise typer.Exit(1)
    _console().print(f"[green]Built download stage: {download_tag}[/green]")

    # Step 2: Export and push download image
    _console().print("[bold]Step 2: Uploading download image...[/bold]")

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
        tar_path = f.name

    try:
        result = subprocess.run([runtime, "save", "-o", tar_path, download_tag])
        if result.returncode != 0:
            _console().print("[red]Failed to export image[/red]")
            raise typer.Exit(1)

        remote_tar_filename = download_tag.replace(":", "+").replace("/", "+") + ".tar"
        remote_tar_dir = images_dir

        async def do_upload():
            client = get_async_client()
            try:
                await client.mkdir(
                    system_name=system, path=remote_tar_dir, create_parents=True
                )
            except Exception:
                pass

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=_console(),
            ) as progress:
                progress.add_task(f"Uploading {remote_tar_filename}...", total=None)
                await client.upload(
                    system_name=system,
                    local_file=tar_path,
                    directory=remote_tar_dir,
                    filename=remote_tar_filename,
                    account=account,
                    blocking=True,
                )

        asyncio.run(do_upload())
        remote_tar = os.path.join(remote_tar_dir, remote_tar_filename)
        _console().print(f"[green]Uploaded to {remote_tar}[/green]")

    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)

    # Step 3: Upload Dockerfile and submit remote build job
    _console().print(f"[bold]Step 3: Building offline stage on cluster ({final_tag})...[/bold]")

    staging_dir = config.resolve_path(".fcw/deploy", remote=True)

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

    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in (build_arg or []))
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""

    q_download_tag = shlex.quote(download_tag)
    q_final_tag = shlex.quote(final_tag)
    q_remote_tar = shlex.quote(remote_tar)
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

# Load download image from tar
if ! podman image exists {q_download_tag} 2>/dev/null; then
    if [ -f {q_remote_tar} ]; then
        echo "Loading image from {q_remote_tar}..."
        podman load -i {q_remote_tar}
    else
        echo "Error: image {q_download_tag} not found and no tar at {q_remote_tar}"
        exit 1
    fi
fi

# Get the image ID for reliable reference in FROM directives
IMAGE_ID=$(podman image inspect --format '{{{{.Id}}}}' docker.io/library/{download_tag} 2>/dev/null || podman image inspect --format '{{{{.Id}}}}' {download_tag})
echo "Resolved download image ID: $IMAGE_ID"

echo "=== Building {q_final_tag} (build-offline stage) ==="
cd {q_staging_dir}
podman build \\
    --target build-offline \\
    --build-arg DOWNLOAD_IMAGE=$IMAGE_ID \\
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
    if ":" in image:
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

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-extract
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-extract-%j.out
{format_sbatch_lines(get_global_sbatch_options())}
set -euxo pipefail

{_podman_setup_block()}

# Load image from tar if not already available
if ! podman image exists {q_image} 2>/dev/null; then
    if [ -f {q_remote_tar} ]; then
        echo "Loading image from {q_remote_tar}..."
        podman load -i {q_remote_tar}
    elif [ -f {q_remote_download_tar} ]; then
        echo "Loading download image from {q_remote_download_tar}..."
        podman load -i {q_remote_download_tar}
    else
        echo "Error: image {q_image} not found and no tar at {q_remote_tar} or {q_remote_download_tar}"
        exit 1
    fi
fi

echo "Extracting {q_container_path} from {q_image}..."

# Create container (don't run it)
CID=$(podman create {q_image} /bin/true)
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
            _console().print(f"[green]Extraction complete[/green]")
            
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
            _console().print(f"[dim]After job completes, download with:[/dim]")
            _console().print(f"  fcw data download .fcw/extract/{archive_name} {local_dest}")
            
    finally:
        os.unlink(script_path)


@app.command("patch")
def patch_container(
    ctx: typer.Context,
    local_path: str = typer.Argument(..., help="Local directory with patched code"),
    container_path: str = typer.Argument(..., help="Target path inside container"),
    toml: Optional[str] = typer.Option(None, "--toml", help="TOML file to update with bind-mount"),
    create_toml: bool = typer.Option(False, "--create", help="Create new TOML file if it doesn't exist"),
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
        import re

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
        _console().print(f"[dim]To use, add to your container TOML:[/dim]")
        _console().print(f'  [mounts]')
        _console().print(f'  "{remote_patch_dir}" = "{container_path}"')


@app.command("update")
def update_image(
    ctx: typer.Context,
    local_path: str = typer.Argument(..., help="Local directory with patched code"),
    image: str = typer.Argument(..., help="Base image to patch (e.g., my-fcw-app:download)"),
    container_path: str = typer.Argument(..., help="Path inside container to replace"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Tag for patched image"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild build-offline stage from patched image"),
    dockerfile: Optional[str] = typer.Option(None, "--file", "-f", help="Dockerfile for rebuild (required with --rebuild)"),
    enroot: bool = typer.Option(False, "--enroot", help="Convert final image to enroot squashfs"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for enroot squashfs"),
    build_arg: Optional[List[str]] = typer.Option(None, "--build-arg", help="Set build-time variables for rebuild (KEY=VALUE)"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for job completion"),
):
    """Patch a container image with updated code and optionally rebuild.
    
    This is the symmetric operation to ``extract``: it takes local code and
    bakes it into a container image. The workflow is:
    
    1. Upload patched code to staging directory
    2. Create container from base image, copy in patched code, commit as new image
    3. Optionally rebuild the build-offline stage using the patched image
    4. Optionally export to enroot squashfs
    
    Examples:
        # Just patch the download image
        fcw container update ./code my-fcw-app:download /workspace/BrainBERT --tag my-fcw-app:patched
        
        # Patch and rebuild build-offline stage
        fcw container update ./code my-fcw-app:download /workspace/BrainBERT \\
            --tag my-fcw-app:v2 --rebuild --file env/Dockerfile.prod-multistage
        
        # Full pipeline with enroot
        fcw container update ./code my-fcw-app:download /workspace/BrainBERT \\
            --tag my-fcw-app:v2 --rebuild -f env/Dockerfile.prod-multistage --enroot --wait
    """
    config, system, account = resolve_context(ctx)
    
    if rebuild and not dockerfile:
        _console().print("[red]--rebuild requires --file[/red]")
        raise typer.Exit(1)
    
    local_path = os.path.abspath(local_path)
    if not os.path.isdir(local_path):
        _console().print(f"[red]Not a directory: {local_path}[/red]")
        raise typer.Exit(1)
    
    # Generate tag if not provided
    patched_tag = tag or f"{image.split(':')[0]}:patched"
    final_tag = patched_tag

    # Resolve images directory from config if available
    images_dir = config.resolve_path("ce-images/", remote=True)
    for _name, cont_config in config.containers.items():
        if cont_config.tag in (tag, image, patched_tag, final_tag):
            images_dir = config.resolve_container_images_dir(cont_config)
            break

    # Staging paths
    patch_name = os.path.basename(local_path.rstrip("/"))
    staging_dir = config.resolve_path(".fcw/update", remote=True)
    remote_patch_path = f"{staging_dir}/{patch_name}"
    remote_dockerfile = f"{staging_dir}/Dockerfile" if dockerfile else None
    
    # Step 1: Upload patched code
    _console().print(f"[bold]Step 1: Uploading patched code...[/bold]")
    
    async def do_upload():
        from fcw.commands.data import _upload_directory
        async_client = get_async_client()

        # Create staging directory
        try:
            await async_client.mkdir(
                system_name=system, path=staging_dir, create_parents=True
            )
        except Exception:
            pass

        # Upload code (directory)
        await _upload_directory(async_client, system, account, local_path, remote_patch_path)

        # Upload Dockerfile if rebuilding
        if dockerfile:
            await async_client.upload(
                system_name=system,
                local_file=dockerfile,
                directory=staging_dir,
                filename="Dockerfile",
                account=account,
                blocking=True,
            )
    
    asyncio.run(do_upload())
    _console().print(f"[green]Uploaded to {staging_dir}[/green]")
    
    # Step 2: Build job script
    _console().print(f"[bold]Step 2: Submitting build job...[/bold]")
    
    # Resolve the pushed tar path so the job can load it if needed
    remote_image_tar = _resolve_remote_tar(image, config)

    # Extra build args for podman build
    extra_build_args = " ".join(f"--build-arg {shlex.quote(a)}" for a in (build_arg or []))
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""

    q_image = shlex.quote(image)
    q_remote_image_tar = shlex.quote(remote_image_tar)
    q_remote_patch_path = shlex.quote(remote_patch_path)
    q_container_path = shlex.quote(container_path)
    q_patched_tag = shlex.quote(patched_tag)
    q_final_tag = shlex.quote(final_tag)
    q_staging_dir = shlex.quote(staging_dir)
    global_sbatch = format_sbatch_lines(get_global_sbatch_options())
    setup_block = _podman_setup_block()

    # Common: load base image from tar
    load_image_block = f"""
# Load image from tar if not already available
if ! podman image exists {q_image} 2>/dev/null; then
    if [ -f {q_remote_image_tar} ]; then
        echo "Loading image from {q_remote_image_tar}..."
        podman load -i {q_remote_image_tar}
    else
        echo "Error: image {q_image} not found and no tar at {q_remote_image_tar}"
        exit 1
    fi
fi
"""

    # Build the job script
    if rebuild:
        # Full rebuild: patch + build-offline
        if enroot:
            output_path = output or os.path.join(images_dir, f"{final_tag.replace(':', '+')}.sqsh")

        script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-update
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-update-%j.out
#SBATCH --error=fcw-container-update-%j.out
{global_sbatch}
set -euxo pipefail

{setup_block}
{load_image_block}

echo "=== Step 1: Patching {q_image} ==="

# Create container from base image
CID=$(podman create {q_image} /bin/true)
echo "Created container: $CID"

# Copy patched code into container
podman cp {q_remote_patch_path}/. "$CID:{container_path}"

# Commit as patched image
PATCHED_IMAGE="{patched_tag}-base"
podman commit "$CID" "$PATCHED_IMAGE"
podman rm "$CID"
echo "Committed patched image: $PATCHED_IMAGE"

echo "=== Step 2: Rebuilding build-offline stage ==="

# Build build-offline stage from patched download image
cd {q_staging_dir}
podman build --target build-offline \\
    --build-arg DOWNLOAD_IMAGE=$PATCHED_IMAGE \\
{extra_build_args_line}    -t {q_final_tag} \\
    -f Dockerfile .

echo "Built final image: {q_final_tag}"
"""
        if enroot:
            q_output_path = shlex.quote(output_path)
            script += f"""
echo "=== Step 3: Exporting to enroot ==="
mkdir -p $(dirname {q_output_path})
rm -f {q_output_path}
enroot import -x mount -o {q_output_path} podman://{final_tag} || true
if [ ! -f {q_output_path} ]; then
    echo "ERROR: enroot import failed - output not found: {q_output_path}"
    exit 1
fi
echo "Exported to: {q_output_path}"
ls -lh {q_output_path}
"""
        script += "\nexit 0\n"
    else:
        # Just patch, no rebuild
        script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-update
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-update-%j.out
#SBATCH --error=fcw-container-update-%j.out
{global_sbatch}
set -euxo pipefail

{setup_block}
{load_image_block}

echo "=== Patching {q_image} ==="

# Create container from base image
CID=$(podman create {q_image} /bin/true)
echo "Created container: $CID"

# Copy patched code into container
podman cp {q_remote_patch_path}/. "$CID:{container_path}"

# Commit as patched image
podman commit "$CID" {q_patched_tag}
podman rm "$CID"

echo "Committed patched image: {q_patched_tag}"
podman images | grep {shlex.quote(patched_tag.split(':')[0])}
"""
        if enroot:
            output_path = output or os.path.join(images_dir, f"{patched_tag.replace(':', '+')}.sqsh")
            q_output_path = shlex.quote(output_path)
            script += f"""
echo "=== Exporting to enroot ==="
mkdir -p $(dirname {q_output_path})
rm -f {q_output_path}
enroot import -x mount -o {q_output_path} podman://{patched_tag} || true
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
            _console().print(f"[green]Build complete: {final_tag}[/green]")
            if enroot:
                _console().print(f"[green]Enroot image: {output_path}[/green]")
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
        subprocess.run([runtime, "images"])

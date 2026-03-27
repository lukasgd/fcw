"""Container build, deployment, and patching commands.

This module provides commands for building container images locally, deploying
them to remote HPC clusters via FirecREST, and a streamlined workflow for
iterating on code without rebuilding the entire image.

Key workflows:

1. **Initial Build & Deploy** (slow, one-time setup):

   Build the download stage locally, push it, then build-offline on the cluster:

       fcw container build --stage download -t myapp:download .
       fcw container push myapp:download
       fcw container build-remote myapp:download \\
           -f env/Dockerfile.prod-multistage -t myapp:latest \\
           --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \\
           --enroot --wait

2. **Extract Code for Editing**:
   
   Extract code from the download image to edit locally:
   
       fcw container extract myapp:download /workspace/BrainBERT ./code

3. **Quick Iteration (bind-mount, no rebuild)**:
   
   Upload patched code and generate TOML with bind-mount for srun:
   
       fcw container patch ./code /workspace/BrainBERT --toml env/container.toml
       # Then use: srun --environment env/container.toml ...

4. **Bake Changes (rebuild)**:
   
   Patch the download image and rebuild build-offline:
   
       fcw container update ./code myapp:download /workspace/BrainBERT \\
           --tag myapp:v2 --rebuild --enroot --wait
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from fcw.core import load_config, get_client, get_async_client, get_system, get_account

app = typer.Typer(no_args_is_help=True)
console = Console()


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
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    runtime = _detect_container_runtime()
    console.print(f"[dim]Using container runtime: {runtime}[/dim]")
    
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
    
    console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")
    
    result = subprocess.run(cmd)
    if result.returncode != 0:
        console.print("[red]Build failed[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Built image: {tag}[/green]")
    
    # Save if requested
    if save:
        save_cmd = [runtime, "save", "-o", save, tag]
        console.print(f"[dim]Saving image to {save}...[/dim]")
        result = subprocess.run(save_cmd)
        if result.returncode != 0:
            console.print("[red]Failed to save image[/red]")
            raise typer.Exit(1)
        console.print(f"[green]Saved image to {save}[/green]")


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
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    # Determine remote path
    if remote_path is None:
        # Try to find in config
        for name, cont_config in config.containers.items():
            if cont_config.tag == image or (cont_config.remote_path and name == image):
                remote_path = cont_config.remote_path
                break
        
        if remote_path is None:
            remote_path = "images/"
    
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

        console.print(f"[dim]Exporting {image}...[/dim]")
        result = subprocess.run([runtime, "save", "-o", tar_path, image])
        if result.returncode != 0:
            console.print("[red]Failed to export image[/red]")
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
                console=console,
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
        console.print(f"[green]Uploaded to {remote_path}[/green]")

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
    to the canonical tar filename under images/.
    """
    if image_or_path.endswith(".tar") or "/" in image_or_path:
        return config.resolve_path(image_or_path, remote=True)
    # Image tag -> images/<tag-as-filename>.tar
    tar_name = image_or_path.replace(":", "+").replace("/", "+") + ".tar"
    return config.resolve_path(f"images/{tar_name}", remote=True)


@app.command("import")
def import_image(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Image tag or remote tar file path"),
    output: Optional[str] = typer.Option(None, "--output", "-o", help="Output path for squashfs"),
):
    """Import a container image on the remote cluster.

    Submits a job that loads the tar with podman and converts it to
    enroot squashfs format.

    Accepts an image tag (e.g., myapp:latest) or a remote tar path
    (e.g., images/myapp-latest.tar).
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)

    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)

    remote_tar = _resolve_remote_tar(image, config)

    output_path = output or remote_tar.replace(".tar", ".sqsh")
    output_path = config.resolve_path(output_path, remote=True)

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-import
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-import-%j.out

set -e

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

# Clean up previous podman state
podman system reset -f
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

export XDG_RUNTIME_DIR="$(mktemp -d -p "${{TMPDIR:-/tmp}}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

# Load image
echo "Loading image from {remote_tar}..."
podman load -i {remote_tar}

# Get the loaded image name
IMAGE_ID=$(podman images --format "{{{{.ID}}}}" | head -1)
echo "Loaded image: $IMAGE_ID"

# Convert to enroot squashfs
echo "Converting to enroot squashfs: {output_path}..."
enroot import -x mount -o {output_path} podman://$IMAGE_ID

echo "Done: {output_path}"
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

        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        print(job_id)
        console.print(f"[green]Submitted import job: {job_id}[/green]")
        console.print(f"[dim]Output will be: {output_path}[/dim]")

    finally:
        os.unlink(script_path)


@app.command("build-remote")
def build_remote(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Base image to build from (must be pushed)"),
    dockerfile: str = typer.Option(..., "--dockerfile", "-f", help="Local Dockerfile path"),
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
        fcw container build-remote myapp:download \\
            -f env/Dockerfile.prod-multistage -t myapp:latest \\
            --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \\
            --enroot --wait
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)

    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)

    if not os.path.isfile(dockerfile):
        console.print(f"[red]Dockerfile not found: {dockerfile}[/red]")
        raise typer.Exit(1)

    # Resolve paths
    remote_image_tar = _resolve_remote_tar(image, config)
    staging_dir = config.resolve_path(".fcw/build-remote", remote=True)
    remote_dockerfile = f"{staging_dir}/Dockerfile"

    # Resolve images directory from config if available
    images_dir = config.resolve_path("images/", remote=True)
    for _name, cont_config in config.containers.items():
        if cont_config.tag == tag or cont_config.tag == image:
            images_dir = config.resolve_container_images_dir(cont_config)
            break

    if enroot:
        output_path = output or config.resolve_path(
            f"images/{tag.replace(':', '+')}.sqsh", remote=True
        )

    # Step 1: Upload Dockerfile
    console.print("[bold]Step 1: Uploading Dockerfile...[/bold]")

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
    console.print("[bold]Step 2: Submitting build job...[/bold]")

    extra_build_args = " ".join(f"--build-arg {a}" for a in (build_arg or []))
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""
    target_line = f"    --target {stage} \\\n" if stage else ""

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-build-remote
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-build-remote-%j.out
#SBATCH --error=fcw-container-build-remote-%j.out

set -euxo pipefail

export HOME="${{HOME:-/users/$USER}}"
export CONTAINER_IMAGES_DIR={images_dir}

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

# Set up XDG_RUNTIME_DIR before any podman commands
export XDG_RUNTIME_DIR="$(mktemp -d -p "${{TMPDIR:-/tmp}}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

# Clean up previous podman state (non-fatal if no prior state exists)
podman system reset -f || true
rm -Rf /dev/shm/$USER/*
rm -Rf /tmp/xdg-run-$(id -u)*

# Re-create XDG_RUNTIME_DIR after cleanup
export XDG_RUNTIME_DIR="$(mktemp -d -p "${{TMPDIR:-/tmp}}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

# Load base image from tar
if ! podman image exists {image} 2>/dev/null; then
    if [ -f {remote_image_tar} ]; then
        echo "Loading image from {remote_image_tar}..."
        podman load -i {remote_image_tar}
    else
        echo "Error: image {image} not found and no tar at {remote_image_tar}"
        exit 1
    fi
fi

# Get the image ID for reliable reference in FROM directives
IMAGE_ID=$(podman image inspect --format '{{{{.Id}}}}' docker.io/library/{image} 2>/dev/null || podman image inspect --format '{{{{.Id}}}}' {image})
echo "Resolved image ID: $IMAGE_ID"

echo "=== Building {tag} ==="
cd {staging_dir}
podman build \\
{target_line}    --build-arg DOWNLOAD_IMAGE=$IMAGE_ID \\
{extra_build_args_line}    -t {tag} \\
    -f Dockerfile .

echo "Built image: {tag}"
"""

    if enroot:
        script += f"""
echo "=== Exporting to enroot ==="
mkdir -p $(dirname {output_path})
enroot import -x mount -o {output_path} podman://{tag}
echo "Exported to: {output_path}"
ls -lh {output_path}
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

        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        print(job_id)
        console.print(f"[green]Submitted build job: {job_id}[/green]")

        if wait:
            console.print("[dim]Waiting for build to complete...[/dim]")
            client.wait_for_job(system_name=system, job_id=job_id)
            console.print(f"[green]Build complete: {tag}[/green]")
            if enroot:
                console.print(f"[green]Enroot image: {output_path}[/green]")
    finally:
        os.unlink(script_path)


@app.command("deploy")
def deploy_image(
    ctx: typer.Context,
    name: Optional[str] = typer.Argument(None, help="Container name from config"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Image tag (if not using config)"),
    wait: bool = typer.Option(True, "--wait/--no-wait", help="Wait for import to complete"),
):
    """Build locally, push, and import on remote (all in one).
    
    This is a convenience command that combines build, push, and import.
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    # Find container config
    if name and name in config.containers:
        cont_config = config.containers[name]
        dockerfile = cont_config.file
        image_tag = cont_config.tag
        remote_path = config.resolve_container_image(cont_config)
        images_dir = config.resolve_container_images_dir(cont_config)
    elif tag:
        dockerfile = "Dockerfile"
        image_tag = tag
        remote_path = config.resolve_path(f"images/{tag.replace(':', '+')}.sqsh", remote=True)
        images_dir = config.resolve_path("images/", remote=True)
    else:
        console.print("[red]Specify container name from config or --tag[/red]")
        raise typer.Exit(1)
    
    # 1. Build locally
    console.print("[bold]Step 1: Building image locally...[/bold]")
    runtime = _detect_container_runtime()
    
    build_cmd = [runtime, "build", "-t", image_tag]
    if dockerfile:
        build_cmd.extend(["-f", dockerfile])
    build_cmd.append(".")
    
    result = subprocess.run(build_cmd)
    if result.returncode != 0:
        console.print("[red]Build failed[/red]")
        raise typer.Exit(1)
    
    # 2. Export and upload
    console.print("[bold]Step 2: Uploading image...[/bold]")
    
    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as f:
        tar_path = f.name
    
    try:
        # Export
        result = subprocess.run([runtime, "save", "-o", tar_path, image_tag])
        if result.returncode != 0:
            console.print("[red]Failed to export image[/red]")
            raise typer.Exit(1)
        
        # Upload
        async def do_upload():
            client = get_async_client()
            remote_tar = remote_path.replace(".sqsh", ".tar")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Uploading...", total=None)
                
                await client.upload(
                    system_name=system,
                    local_file=tar_path,
                    directory=os.path.dirname(remote_tar),
                    filename=os.path.basename(remote_tar),
                    account=account,
                    blocking=True,
                )
            
            return remote_tar
        
        remote_tar = asyncio.run(do_upload())
        
    finally:
        if os.path.exists(tar_path):
            os.unlink(tar_path)
    
    # 3. Import on remote
    console.print("[bold]Step 3: Importing on remote cluster...[/bold]")
    
    # Use import command logic
    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-import
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-import-%j.out

set -e

export CONTAINER_IMAGES_DIR={images_dir}

mkdir -p $HOME/.config/containers
if [ ! -f $HOME/.config/containers/storage.conf ]; then
    cat > $HOME/.config/containers/storage.conf << 'EOF'
[storage]
driver = "overlay"
runroot = "/dev/shm/$USER/runroot"
graphroot = "/dev/shm/$USER/root"
EOF
fi

export XDG_RUNTIME_DIR=/tmp/$USER/containers/run
mkdir -p $XDG_RUNTIME_DIR

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

echo "Loading image from {remote_tar}..."
podman load -i {remote_tar}

IMAGE_ID=$(podman images --format "{{{{.ID}}}}" | head -1)
echo "Loaded image: $IMAGE_ID"

echo "Converting to enroot: {remote_path}..."
enroot import -x mount -o {remote_path} podman://$IMAGE_ID

echo "Done: {remote_path}"
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
        
        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        console.print(f"[green]Submitted import job: {job_id}[/green]")
        
        if wait:
            console.print(f"[dim]Waiting for import to complete...[/dim]")
            client.wait_for_job(system_name=system, job_id=job_id)
            console.print(f"[green]Image deployed: {remote_path}[/green]")
        else:
            print(job_id)
            
    finally:
        os.unlink(script_path)


# -----------------------------------------------------------------------------
# Extract / Patch / Update Commands (Code Iteration Workflow)
# -----------------------------------------------------------------------------

@app.command("extract")
def extract_from_image(
    ctx: typer.Context,
    image: str = typer.Argument(..., help="Source image (e.g., myapp:download)"),
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
        fcw container extract myapp:download /workspace/BrainBERT ./code
        
        # Extract to current directory
        fcw container extract myapp:download /workspace/BrainBERT .
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    # Staging path for extraction
    staging_dir = config.resolve_path(".fcw/extract", remote=True)
    archive_name = f"{os.path.basename(container_path.rstrip('/'))}.tar.gz"
    remote_archive = f"{staging_dir}/{archive_name}"

    # Resolve the pushed tar path so the job can load it if needed
    remote_tar = _resolve_remote_tar(image, config)

    script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-extract
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-extract-%j.out

set -euxo pipefail

export HOME="${{HOME:-/users/$USER}}"

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

# Clean up previous podman state
podman system reset -f
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

export XDG_RUNTIME_DIR="$(mktemp -d -p "${{TMPDIR:-/tmp}}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

# Load image from tar if not already available
if ! podman image exists {image} 2>/dev/null; then
    if [ -f {remote_tar} ]; then
        echo "Loading image from {remote_tar}..."
        podman load -i {remote_tar}
    else
        echo "Error: image {image} not found and no tar at {remote_tar}"
        exit 1
    fi
fi

echo "Extracting {container_path} from {image}..."

# Create container (don't run it)
CID=$(podman create {image} /bin/true)
echo "Created container: $CID"

# Create staging directory
mkdir -p {staging_dir}

# Copy files out of container
TMPDIR=$(mktemp -d)
podman cp "$CID:{container_path}" "$TMPDIR/"
podman rm "$CID"

# Create archive
cd "$TMPDIR"
tar czf {remote_archive} *
rm -rf "$TMPDIR"

echo "Extracted to: {remote_archive}"
ls -lh {remote_archive}
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
        
        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        console.print(f"[green]Submitted extract job: {job_id}[/green]")
        
        if wait:
            console.print("[dim]Waiting for extraction...[/dim]")
            client.wait_for_job(system_name=system, job_id=job_id)
            console.print(f"[green]Extraction complete[/green]")
            
            # Download the archive
            console.print(f"[dim]Downloading {remote_archive}...[/dim]")
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
            console.print(f"[dim]Extracting to {local_dest}...[/dim]")
            subprocess.run(["tar", "xzf", local_archive, "-C", local_dest], check=True)
            os.unlink(local_archive)
            
            console.print(f"[green]Extracted to {local_dest}[/green]")
        else:
            print(job_id)
            console.print(f"[dim]After job completes, download with:[/dim]")
            console.print(f"  fcw data download .fcw/extract/{archive_name} {local_dest}")
            
    finally:
        os.unlink(script_path)


@app.command("patch")
def patch_container(
    ctx: typer.Context,
    local_path: str = typer.Argument(..., help="Local directory with patched code"),
    container_path: str = typer.Argument(..., help="Target path inside container"),
    toml: Optional[str] = typer.Option(None, "--toml", "-t", help="TOML file to update with bind-mount"),
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
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    local_path = os.path.abspath(local_path)
    if not os.path.isdir(local_path):
        console.print(f"[red]Not a directory: {local_path}[/red]")
        raise typer.Exit(1)
    
    # Determine remote patch directory
    patch_name = os.path.basename(local_path.rstrip("/"))
    remote_patch_dir = config.resolve_path(f".patches/{patch_name}", remote=True)
    
    # Upload the patched code
    console.print(f"[dim]Uploading {local_path} to {remote_patch_dir}...[/dim]")
    
    async def do_upload():
        from fcw.commands.data import _upload_directory
        async_client = get_async_client()
        await _upload_directory(async_client, system, account, local_path, remote_patch_dir)

    asyncio.run(do_upload())
    console.print(f"[green]Uploaded to {remote_patch_dir}[/green]")
    
    # Update TOML file if specified
    bind_mount = f"{remote_patch_dir}:{container_path}"
    if toml:
        toml_path = Path(toml)
        import re

        if not toml_path.exists():
            if create_toml:
                console.print(f"[dim]Creating {toml}...[/dim]")
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
                console.print(f"[red]TOML file not found: {toml}[/red]")
                console.print("[dim]Use --create to create a new file[/dim]")
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
        console.print(f"[green]Updated {toml} (local + remote)[/green]")
        console.print(f"[dim]Run with: srun --environment {remote_toml} ...[/dim]")
    else:
        console.print(f"[dim]To use, add to your container TOML:[/dim]")
        console.print(f'  [mounts]')
        console.print(f'  "{remote_patch_dir}" = "{container_path}"')


@app.command("update")
def update_image(
    ctx: typer.Context,
    local_path: str = typer.Argument(..., help="Local directory with patched code"),
    image: str = typer.Argument(..., help="Base image to patch (e.g., myapp:download)"),
    container_path: str = typer.Argument(..., help="Path inside container to replace"),
    tag: Optional[str] = typer.Option(None, "--tag", "-t", help="Tag for patched image"),
    rebuild: bool = typer.Option(False, "--rebuild", help="Rebuild build-offline stage from patched image"),
    dockerfile: Optional[str] = typer.Option(None, "--dockerfile", "-f", help="Dockerfile for rebuild (required with --rebuild)"),
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
        fcw container update ./code myapp:download /workspace/BrainBERT --tag myapp:patched
        
        # Patch and rebuild build-offline stage
        fcw container update ./code myapp:download /workspace/BrainBERT \\
            --tag myapp:v2 --rebuild --dockerfile env/Dockerfile.prod-multistage
        
        # Full pipeline with enroot
        fcw container update ./code myapp:download /workspace/BrainBERT \\
            --tag myapp:v2 --rebuild -f env/Dockerfile.prod-multistage --enroot --wait
    """
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    if rebuild and not dockerfile:
        console.print("[red]--rebuild requires --dockerfile[/red]")
        raise typer.Exit(1)
    
    local_path = os.path.abspath(local_path)
    if not os.path.isdir(local_path):
        console.print(f"[red]Not a directory: {local_path}[/red]")
        raise typer.Exit(1)
    
    # Generate tag if not provided
    patched_tag = tag or f"{image.split(':')[0]}:patched"
    final_tag = patched_tag
    
    # Staging paths
    patch_name = os.path.basename(local_path.rstrip("/"))
    staging_dir = config.resolve_path(".fcw/update", remote=True)
    remote_patch_path = f"{staging_dir}/{patch_name}"
    remote_dockerfile = f"{staging_dir}/Dockerfile" if dockerfile else None
    
    # Step 1: Upload patched code
    console.print(f"[bold]Step 1: Uploading patched code...[/bold]")
    
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
    console.print(f"[green]Uploaded to {staging_dir}[/green]")
    
    # Step 2: Build job script
    console.print(f"[bold]Step 2: Submitting build job...[/bold]")
    
    # Resolve the pushed tar path so the job can load it if needed
    remote_image_tar = _resolve_remote_tar(image, config)

    # Extra build args for podman build
    extra_build_args = " ".join(f"--build-arg {a}" for a in (build_arg or []))
    extra_build_args_line = f"    {extra_build_args} \\\n" if extra_build_args else ""

    # Build the job script
    if rebuild:
        # Full rebuild: patch + build-offline
        if enroot:
            output_path = output or config.resolve_path(f"images/{final_tag.replace(':', '+')}.sqsh", remote=True)

        script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-update
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-update-%j.out
#SBATCH --error=fcw-container-update-%j.out

set -euxo pipefail

export HOME="${{HOME:-/users/$USER}}"

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

# Clean up previous podman state
podman system reset -f
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

export XDG_RUNTIME_DIR="$(mktemp -d -p "${{TMPDIR:-/tmp}}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

# Load image from tar if not already available
if ! podman image exists {image} 2>/dev/null; then
    if [ -f {remote_image_tar} ]; then
        echo "Loading image from {remote_image_tar}..."
        podman load -i {remote_image_tar}
    else
        echo "Error: image {image} not found and no tar at {remote_image_tar}"
        exit 1
    fi
fi

echo "=== Step 1: Patching {image} ==="

# Create container from base image
CID=$(podman create {image} /bin/true)
echo "Created container: $CID"

# Copy patched code into container
podman cp {remote_patch_path}/. $CID:{container_path}

# Commit as patched image
PATCHED_IMAGE="{patched_tag}-base"
podman commit $CID $PATCHED_IMAGE
podman rm $CID
echo "Committed patched image: $PATCHED_IMAGE"

echo "=== Step 2: Rebuilding build-offline stage ==="

# Build build-offline stage from patched download image
cd {staging_dir}
podman build --target build-offline \\
    --build-arg DOWNLOAD_IMAGE=$PATCHED_IMAGE \\
{extra_build_args_line}    -t {final_tag} \\
    -f Dockerfile .

echo "Built final image: {final_tag}"
"""
        if enroot:
            script += f"""
echo "=== Step 3: Exporting to enroot ==="
mkdir -p $(dirname {output_path})
enroot import -x mount -o {output_path} podman://{final_tag}
echo "Exported to: {output_path}"
ls -lh {output_path}
"""
    else:
        # Just patch, no rebuild
        script = f"""#!/bin/bash -l
#SBATCH --job-name=fcw-container-update
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --output=fcw-container-update-%j.out
#SBATCH --error=fcw-container-update-%j.out

set -euxo pipefail

export HOME="${{HOME:-/users/$USER}}"

# Wait for systemd user session to settle before podman
while pgrep -U $(id -u) systemd ; do sleep 0.2 ; done

# Clean up previous podman state
podman system reset -f
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

export XDG_RUNTIME_DIR="$(mktemp -d -p "${{TMPDIR:-/tmp}}" xdg-run-$UID.XXXXXX)"
chmod 700 "$XDG_RUNTIME_DIR"

# Load image from tar if not already available
if ! podman image exists {image} 2>/dev/null; then
    if [ -f {remote_image_tar} ]; then
        echo "Loading image from {remote_image_tar}..."
        podman load -i {remote_image_tar}
    else
        echo "Error: image {image} not found and no tar at {remote_image_tar}"
        exit 1
    fi
fi

echo "=== Patching {image} ==="

# Create container from base image
CID=$(podman create {image} /bin/true)
echo "Created container: $CID"

# Copy patched code into container
podman cp {remote_patch_path}/. $CID:{container_path}

# Commit as patched image
podman commit $CID {patched_tag}
podman rm $CID

echo "Committed patched image: {patched_tag}"
podman images | grep {patched_tag.split(':')[0]}
"""
        if enroot:
            output_path = output or config.resolve_path(f"images/{patched_tag.replace(':', '+')}.sqsh", remote=True)
            script += f"""
echo "=== Exporting to enroot ==="
mkdir -p $(dirname {output_path})
enroot import -x mount -o {output_path} podman://{patched_tag}
echo "Exported to: {output_path}"
ls -lh {output_path}
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
        
        job_id = result.get("jobId") or result.get("jobid") or result.get("job_id")
        print(job_id)
        console.print(f"[green]Submitted build job: {job_id}[/green]")
        
        if wait:
            console.print("[dim]Waiting for build to complete...[/dim]")
            client.wait_for_job(system_name=system, job_id=job_id)
            console.print(f"[green]Build complete: {final_tag}[/green]")
            if enroot:
                console.print(f"[green]Enroot image: {output_path}[/green]")
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
        config_file = ctx.obj.get("config_file") if ctx.obj else None
        config = load_config(config_file)
        
        system = get_system(ctx.obj.get("system") if ctx.obj else None)
        account = get_account(ctx.obj.get("account") if ctx.obj else None)
        
        # List squashfs files in images directory
        images_path = config.resolve_path("images", remote=True)
        
        async def do_list():
            client = get_async_client()
            try:
                entries = await client.list_files(
                    system_name=system,
                    path=images_path,
                    recursive=False,
                )
                
                console.print(f"[bold]Remote images in {images_path}:[/bold]")
                for entry in entries:
                    name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
                    size = entry.get("size") if isinstance(entry, dict) else getattr(entry, "size", 0)
                    if name and name.endswith(".sqsh"):
                        console.print(f"  {name}  ({size / 1024 / 1024:.1f} MB)")
            except Exception as e:
                console.print(f"[yellow]Could not list remote images: {e}[/yellow]")
        
        asyncio.run(do_list())
    else:
        # List local images
        runtime = _detect_container_runtime()
        subprocess.run([runtime, "images"])

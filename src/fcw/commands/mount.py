"""FUSE mount command group.

This module provides commands for mounting remote FirecREST storage
as a local filesystem using pyfuse3.
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import typer
from rich.console import Console

from fcw.core import load_config, get_system, get_account

app = typer.Typer(no_args_is_help=True)
console = Console()

# Check if pyfuse3 is available
FUSE_AVAILABLE = False
try:
    import pyfuse3
    import trio
    FUSE_AVAILABLE = True
except ImportError:
    pass


def _check_fuse_available():
    """Check if FUSE dependencies are available."""
    if not FUSE_AVAILABLE:
        console.print("[red]FUSE support not available.[/red]")
        console.print("Install with: pip install fcw[fuse]")
        console.print("Also requires libfuse3-dev system package.")
        raise typer.Exit(1)


@app.command("start")
def mount_filesystem(
    ctx: typer.Context,
    remote_path: str = typer.Argument(..., help="Remote path to mount"),
    mountpoint: str = typer.Argument(..., help="Local mount point"),
    read_only: bool = typer.Option(False, "--read-only", "-r", help="Mount read-only"),
    cache_ttl: int = typer.Option(5, "--cache-ttl", help="Cache TTL in seconds"),
    foreground: bool = typer.Option(False, "--foreground", "-f", help="Run in foreground"),
    allow_other: bool = typer.Option(False, "--allow-other", help="Allow other users to access"),
    debug: bool = typer.Option(False, "--debug", help="Enable debug logging"),
):
    """Mount remote FirecREST storage as local filesystem.
    
    This uses FUSE (Filesystem in Userspace) to provide transparent
    access to remote files. Requires pyfuse3 and libfuse3.
    
    Example:
        fcw mount start /scratch/user/project ./remote-files
        ls ./remote-files
        fcw mount stop ./remote-files
    """
    _check_fuse_available()
    
    config_file = ctx.obj.get("config_file") if ctx.obj else None
    config = load_config(config_file)
    
    system = get_system(ctx.obj.get("system") if ctx.obj else None)
    account = get_account(ctx.obj.get("account") if ctx.obj else None)
    
    # Resolve remote path
    if not remote_path.startswith("/"):
        remote_path = config.resolve_path(remote_path, remote=True)
    
    # Create mountpoint if needed
    if not os.path.exists(mountpoint):
        os.makedirs(mountpoint)
    
    # Import FUSE filesystem implementation
    from fcw.fuse.filesystem import FirecrestFS, run_filesystem
    
    console.print(f"Mounting {system}:{remote_path} at {mountpoint}")
    console.print(f"[dim]Cache TTL: {cache_ttl}s, Read-only: {read_only}[/dim]")
    
    if not foreground:
        console.print("[dim]Running in background. Use 'fcw mount stop' to unmount.[/dim]")
    
    # Run the filesystem
    run_filesystem(
        mountpoint=mountpoint,
        remote_root=remote_path,
        system=system,
        account=account,
        cache_ttl=cache_ttl,
        read_only=read_only,
        allow_other=allow_other,
        foreground=foreground,
        debug=debug,
    )


@app.command("stop")
def unmount_filesystem(
    mountpoint: str = typer.Argument(..., help="Mount point to unmount"),
    force: bool = typer.Option(False, "--force", "-f", help="Force unmount"),
):
    """Unmount a FUSE filesystem.
    
    Example:
        fcw mount stop ./remote-files
    """
    import subprocess
    
    cmd = ["fusermount", "-u"]
    if force:
        cmd.append("-z")  # Lazy unmount
    cmd.append(mountpoint)
    
    result = subprocess.run(cmd)
    if result.returncode == 0:
        console.print(f"[green]Unmounted {mountpoint}[/green]")
    else:
        console.print(f"[red]Failed to unmount {mountpoint}[/red]")
        if not force:
            console.print("Try with --force for lazy unmount")
        raise typer.Exit(1)


@app.command("list")
def list_mounts():
    """List active fcw FUSE mounts."""
    import subprocess
    
    result = subprocess.run(["mount", "-t", "fuse.fcw"], capture_output=True, text=True)
    
    if result.stdout.strip():
        console.print("[bold]Active fcw mounts:[/bold]")
        for line in result.stdout.strip().split("\n"):
            console.print(f"  {line}")
    else:
        # Try generic fuse mounts and filter
        result = subprocess.run(["mount", "-t", "fuse"], capture_output=True, text=True)
        firecrest_mounts = [l for l in result.stdout.split("\n") if "firecrest" in l.lower()]
        
        if firecrest_mounts:
            console.print("[bold]Active FirecREST mounts:[/bold]")
            for line in firecrest_mounts:
                console.print(f"  {line}")
        else:
            console.print("[dim]No active fcw mounts found[/dim]")

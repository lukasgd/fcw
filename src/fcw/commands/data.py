"""Data transfer command group."""

from __future__ import annotations

import asyncio
import os
import re
import tarfile
import tempfile
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

import firecrest

from fcw.core import load_config, DirectoryType, get_async_client, get_system, get_account, resolve_context, get_console

app = typer.Typer(no_args_is_help=True)
_console = get_console

# Sync state directory
SYNC_STATE_DIR = ".fcw/sync"


def _spinner(message: str):
    """Create a progress spinner with a message."""
    p = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=_console())
    p.add_task(message, total=None)
    return p


def _get_sync_marker_path(local_dir: str, direction: str) -> Path:
    """Get path to sync marker file."""
    marker_dir = Path(SYNC_STATE_DIR)
    marker_dir.mkdir(parents=True, exist_ok=True)
    # Create a safe filename from the directory path
    safe_name = local_dir.replace("/", "_").replace("\\", "_").strip("_")
    return marker_dir / f"{safe_name}.{direction}.marker"


def _read_last_sync_timestamp(local_dir: str, direction: str) -> float:
    """Read the last sync timestamp from marker file."""
    marker_path = _get_sync_marker_path(local_dir, direction)
    try:
        return float(marker_path.read_text().strip())
    except (FileNotFoundError, ValueError):
        return 0.0


def _write_last_sync_timestamp(local_dir: str, direction: str, ts: float | None = None):
    """Write the sync timestamp to marker file."""
    if ts is None:
        ts = time.time()
    marker_path = _get_sync_marker_path(local_dir, direction)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(str(ts))


def _collect_local_files_since(
    local_dir: str, since_ts: float, follow_symlinks: bool = False
) -> list[tuple[str, str]]:
    """Collect local files modified since timestamp.

    Returns list of (absolute_path, relative_path) tuples.
    """
    local_dir = os.path.abspath(local_dir)
    files = []

    for root, _dirs, filenames in os.walk(local_dir, followlinks=follow_symlinks):
        for name in filenames:
            # Skip sync markers and hidden files
            if name.startswith(".fcw") or name.startswith(".firecrest"):
                continue
            
            abs_path = os.path.join(root, name)
            try:
                mtime = os.path.getmtime(abs_path)
            except FileNotFoundError:
                continue
            
            if mtime > since_ts:
                rel_path = os.path.relpath(abs_path, local_dir)
                files.append((abs_path, rel_path))
    
    return files


async def _collect_remote_files_since(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    remote_dir: str,
    since_ts: float,
) -> list[str]:
    """Collect remote files modified since timestamp.
    
    Returns list of relative paths.
    """
    entries = await client.list_files(
        system_name=system,
        path=remote_dir,
        recursive=True,
        show_hidden=True,
    )
    
    files = []
    for entry in entries:
        name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
        entry_type = entry.get("type") if isinstance(entry, dict) else getattr(entry, "type", None)
        
        # Skip directories and marker files
        if entry_type == "d":
            continue
        if name and (name.startswith(".fcw") or name.startswith(".firecrest")):
            continue
        
        last_modified = entry.get("lastModified") if isinstance(entry, dict) else getattr(entry, "lastModified", None)
        if not last_modified:
            continue
        
        dt = datetime.fromisoformat(last_modified)
        mtime = dt.timestamp()
        
        if mtime > since_ts:
            files.append(name)
    
    return files


def _build_emacs_match_pattern(paths: list[str], source_path: str) -> str:
    """Build emacs-style regex pattern for tar's --match option."""
    root = os.path.basename(source_path.rstrip("/"))
    patterns = []
    
    for p in paths:
        p = p.replace(os.sep, "/").lstrip("./")
        full = f"{root}/{p}"
        escaped = re.escape(full)
        fragment = "./" + escaped
        patterns.append(fragment)
    
    if not patterns:
        return "^$"
    if len(patterns) == 1:
        return f"^{patterns[0]}$"
    inner = r"\|".join(patterns)
    return rf"^\({inner}\)$"


async def _upload_directory(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    local_dir: str,
    remote_dir: str,
    follow_symlinks: bool = False,
) -> None:
    """Upload a local directory by tar → upload → extract on remote."""
    local_dir = os.path.abspath(local_dir)
    extract_target = os.path.dirname(remote_dir.rstrip("/"))

    # Ensure the parent directory exists on remote
    try:
        await client.mkdir(system_name=system, path=extract_target, create_parents=True)
    except Exception:
        pass  # May already exist

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "fcw_upload.tar.gz")

        with tarfile.open(archive_path, "w:gz", dereference=follow_symlinks) as tar:
            tar.add(local_dir, arcname=os.path.basename(local_dir.rstrip(os.sep)))

        remote_archive = extract_target.rstrip("/") + "/.fcw_upload.tar.gz"
        await client.upload(
            system_name=system,
            local_file=archive_path,
            directory=extract_target,
            filename=".fcw_upload.tar.gz",
            account=account,
            blocking=True,
        )

        await client.extract(
            system_name=system,
            source_path=remote_archive,
            target_path=extract_target,
            account=account,
            blocking=True,
        )

        try:
            await client.rm(system_name=system, path=remote_archive, account=account, blocking=True)
        except Exception:
            pass


def _extract_dir_archive(local_archive: str, local_dir: str) -> None:
    """Unpack a directory archive (rooted at the dir basename) so its contents
    land in local_dir, not a nested local_dir/<basename>/ subdir."""
    with tarfile.open(local_archive, "r:gz") as tar:
        tar.extractall(path=os.path.dirname(local_dir))


async def _download_directory(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    remote_dir: str,
    local_dir: str,
) -> None:
    """Download a remote directory by compress → download → extract locally."""
    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    extract_target = os.path.dirname(remote_dir.rstrip("/"))
    remote_archive = extract_target.rstrip("/") + "/.fcw_download.tar.gz"

    await client.compress(
        system_name=system,
        source_path=remote_dir,
        target_path=remote_archive,
        account=account,
        blocking=True,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        local_archive = os.path.join(tmpdir, "fcw_download.tar.gz")

        await client.download(
            system_name=system,
            source_path=remote_archive,
            target_path=local_archive,
            account=account,
            blocking=True,
        )

        _extract_dir_archive(local_archive, local_dir)

    try:
        await client.rm(system_name=system, path=remote_archive, account=account, blocking=True)
    except Exception:
        pass


async def _upload_incremental(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    local_dir: str,
    remote_dir: str,
    follow_symlinks: bool = False,
) -> int:
    """Upload only files changed since last sync.

    Returns number of files uploaded.
    """
    local_dir = os.path.abspath(local_dir)
    last_sync = _read_last_sync_timestamp(local_dir, "push")
    files = _collect_local_files_since(local_dir, last_sync, follow_symlinks)
    
    if not files:
        return 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "fcw_sync.tar.gz")

        # Create archive with relative paths (no root directory wrapper)
        with tarfile.open(archive_path, "w:gz", dereference=follow_symlinks) as tar:
            for abs_path, rel_path in files:
                tar.add(abs_path, arcname=rel_path)

        # Upload archive to remote_dir and extract in place
        remote_archive = remote_dir.rstrip("/") + "/.fcw_sync_upload.tar.gz"
        await client.upload(
            system_name=system,
            local_file=archive_path,
            directory=remote_dir,
            filename=".fcw_sync_upload.tar.gz",
            account=account,
            blocking=True,
        )

        # Extract on remote
        await client.extract(
            system_name=system,
            source_path=remote_archive,
            target_path=remote_dir,
            account=account,
            blocking=True,
        )
        
        # Cleanup remote archive
        try:
            await client.rm(system_name=system, path=remote_archive, account=account, blocking=True)
        except Exception:
            pass
    
    _write_last_sync_timestamp(local_dir, "push")
    return len(files)


async def _download_incremental(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    remote_dir: str,
    local_dir: str,
) -> int:
    """Download only files changed since last sync.
    
    Returns number of files downloaded.
    """
    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    
    last_sync = _read_last_sync_timestamp(local_dir, "pull")
    files = await _collect_remote_files_since(client, system, remote_dir, last_sync)
    
    if not files:
        return 0
    
    # Build match pattern for selective compression
    match_pattern = _build_emacs_match_pattern(files, remote_dir)
    remote_archive = f"{remote_dir.rstrip('/')}/../.fcw_sync_download.tar.gz"
    
    # Compress matching files on remote
    num_attempts = 3
    for attempt in range(num_attempts):
        try:
            await client.compress(
                system_name=system,
                source_path=remote_dir,
                target_path=remote_archive,
                match_pattern=match_pattern,
                account=account,
                blocking=True,
            )
            break
        except firecrest.FirecrestException as e:
            if attempt == num_attempts - 1:
                raise
            await asyncio.sleep(2)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        local_archive = os.path.join(tmpdir, "fcw_sync_download.tar.gz")
        
        # Download archive
        await client.download(
            system_name=system,
            source_path=remote_archive,
            target_path=local_archive,
            account=account,
            blocking=True,
        )
        
        # Extract locally
        _extract_dir_archive(local_archive, local_dir)
    
    # Cleanup remote archive
    try:
        await client.rm(system_name=system, path=remote_archive, account=account, blocking=True)
    except Exception:
        pass
    
    _write_last_sync_timestamp(local_dir, "pull")
    return len(files)


@app.command()
def upload(
    ctx: typer.Context,
    paths: List[str] = typer.Argument(..., help="Local paths to upload"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Upload only changed files"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuously watch for changes"),
    interval: int = typer.Option(10, "--interval", help="Watch interval in seconds"),
    force: bool = typer.Option(False, "--force", "-f", help="Override directory type restrictions"),
    follow_symlinks: bool = typer.Option(
        False, "--follow-symlinks", "-L",
        help="Upload the real files behind symlinks instead of the links themselves"),
):
    """Upload local files/directories to remote storage."""
    config, system, account = resolve_context(ctx)

    # Validate directory types
    for path in paths:
        rel_path = os.path.relpath(path, config.workdir.local)
        if not config.can_upload(rel_path) and not force:
            dir_type = config.get_directory_type(rel_path)
            _console().print(
                f"[red]Error:[/red] '{rel_path}' is declared as '{dir_type.value}' (download-only).\n"
                "Use --force to override or change type in fcw.yaml."
            )
            raise typer.Exit(1)
    
    async def do_upload():
        client = get_async_client()
        
        while True:
            for local_path in paths:
                rel_path = os.path.relpath(local_path, config.workdir.local)
                remote_path = config.resolve_path(rel_path, remote=True)
                
                if incremental and os.path.isdir(local_path):
                    with _spinner(f"Syncing {local_path}..."):
                        count = await _upload_incremental(
                            client, system, account, local_path, remote_path,
                            follow_symlinks,
                        )
                    if count > 0:
                        _console().print(f"[green]Uploaded {count} files to {remote_path}[/green]")
                    else:
                        _console().print(f"[dim]No changes in {local_path}[/dim]")
                elif os.path.isdir(local_path):
                    # Directory upload via tar
                    with _spinner(f"Uploading {local_path}..."):
                        await _upload_directory(
                            client, system, account, local_path, remote_path,
                            follow_symlinks,
                        )
                    _console().print(f"[green]Uploaded {local_path} to {remote_path}[/green]")
                else:
                    # Direct file upload — ensure the remote parent dir exists first
                    try:
                        await client.mkdir(
                            system_name=system,
                            path=os.path.dirname(remote_path),
                            create_parents=True,
                        )
                    except Exception:
                        pass  # May already exist
                    with _spinner(f"Uploading {local_path}..."):
                        await client.upload(
                            system_name=system,
                            local_file=local_path,
                            directory=os.path.dirname(remote_path),
                            filename=os.path.basename(remote_path),
                            account=account,
                            blocking=True,
                        )
                    _console().print(f"[green]Uploaded {local_path} to {remote_path}[/green]")
            
            if not watch:
                break
            
            _console().print(f"[dim]Waiting {interval}s...[/dim]")
            await asyncio.sleep(interval)
    
    asyncio.run(do_upload())


@app.command()
def download(
    ctx: typer.Context,
    paths: List[str] = typer.Argument(..., help="Remote paths to download"),
    incremental: bool = typer.Option(False, "--incremental", "-i", help="Download only changed files"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Continuously watch for changes"),
    interval: int = typer.Option(10, "--interval", help="Watch interval in seconds"),
    force: bool = typer.Option(False, "--force", "-f", help="Override directory type restrictions"),
):
    """Download remote files/directories to local storage."""
    config, system, account = resolve_context(ctx)

    # Validate directory types
    for path in paths:
        if not config.can_download(path) and not force:
            dir_type = config.get_directory_type(path)
            _console().print(
                f"[red]Error:[/red] '{path}' is declared as '{dir_type.value}' (upload-only).\n"
                "Use --force to override or change type in fcw.yaml."
            )
            raise typer.Exit(1)
    
    async def do_download():
        client = get_async_client()
        
        while True:
            for rel_path in paths:
                remote_path = config.resolve_path(rel_path, remote=True)
                local_path = config.resolve_path(rel_path, remote=False)
                
                if incremental:
                    with _spinner(f"Syncing {rel_path}..."):
                        count = await _download_incremental(
                            client, system, account, remote_path, local_path
                        )
                    if count > 0:
                        _console().print(f"[green]Downloaded {count} files from {remote_path}[/green]")
                    else:
                        _console().print(f"[dim]No changes in {rel_path}[/dim]")
                else:
                    # Check if remote path is a directory
                    is_dir = False
                    try:
                        await client.list_files(
                            system_name=system,
                            path=remote_path,
                            recursive=False,
                        )
                        is_dir = True
                    except Exception:
                        pass

                    if is_dir:
                        # Directory download via compress
                        with _spinner(f"Downloading {rel_path}..."):
                            await _download_directory(
                                client, system, account, remote_path, local_path
                            )
                        _console().print(
                            f"[green]Downloaded {remote_path} to {local_path}[/green]"
                        )
                    else:
                        # Direct file download
                        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
                        with _spinner(f"Downloading {rel_path}..."):
                            await client.download(
                                system_name=system,
                                source_path=remote_path,
                                target_path=local_path,
                                account=account,
                                blocking=True,
                            )
                        _console().print(
                            f"[green]Downloaded {remote_path} to {local_path}[/green]"
                        )
            
            if not watch:
                break
            
            _console().print(f"[dim]Waiting {interval}s...[/dim]")
            await asyncio.sleep(interval)
    
    asyncio.run(do_download())


@app.command("ls")
def list_files(
    ctx: typer.Context,
    path: str = typer.Argument(..., help="Remote path to list"),
    recursive: bool = typer.Option(False, "-R", "--recursive", help="List recursively"),
):
    """List remote directory contents."""
    config = load_config((ctx.obj or {}).get("config_file"))
    system = get_system((ctx.obj or {}).get("system"))

    remote_path = config.resolve_path(path, remote=True)
    
    client = get_async_client()
    
    async def do_list():
        entries = await client.list_files(
            system_name=system,
            path=remote_path,
            recursive=recursive,
            show_hidden=True,
        )
        
        for entry in entries:
            name = entry.get("name") if isinstance(entry, dict) else getattr(entry, "name", None)
            entry_type = entry.get("type") if isinstance(entry, dict) else getattr(entry, "type", None)
            size = entry.get("size") if isinstance(entry, dict) else getattr(entry, "size", 0)
            
            if entry_type == "d":
                _console().print(f"[blue]{name}/[/blue]")
            else:
                _console().print(f"{name}  ({size} bytes)")
    
    asyncio.run(do_list())


@app.command()
def rm(
    ctx: typer.Context,
    paths: List[str] = typer.Argument(..., help="Remote paths to remove"),
    force: bool = typer.Option(False, "--force", "-f", help="Don't prompt for confirmation"),
):
    """Remove remote files or directories."""
    config, system, account = resolve_context(ctx)

    if not force:
        paths_str = ", ".join(paths)
        if not typer.confirm(f"Remove {paths_str}?"):
            raise typer.Abort()
    
    client = get_async_client()
    
    async def do_rm():
        for path in paths:
            remote_path = config.resolve_path(path, remote=True)
            await client.rm(
                system_name=system,
                path=remote_path,
                account=account,
                blocking=True,
            )
            _console().print(f"[green]Removed {remote_path}[/green]")
    
    asyncio.run(do_rm())


@app.command()
def status(
    ctx: typer.Context,
):
    """Show last sync times for configured directories."""
    config = load_config((ctx.obj or {}).get("config_file"))

    from rich.table import Table
    
    table = Table(title="Sync Status")
    table.add_column("Directory")
    table.add_column("Type")
    table.add_column("Last Upload")
    table.add_column("Last Download")
    
    for path, dir_config in config.directories.items():
        local_path = config.resolve_path(path, remote=False)
        
        push_ts = _read_last_sync_timestamp(local_path, "push")
        pull_ts = _read_last_sync_timestamp(local_path, "pull")
        
        push_str = datetime.fromtimestamp(push_ts).strftime("%Y-%m-%d %H:%M:%S") if push_ts > 0 else "-"
        pull_str = datetime.fromtimestamp(pull_ts).strftime("%Y-%m-%d %H:%M:%S") if pull_ts > 0 else "-"
        
        table.add_row(path, dir_config.type.value, push_str, pull_str)
    
    _console().print(table)

# FIXME: ls has already been implemented, others from pyfirerest not, e.g. mkdir, mv, chmod, chown, cp, compress, extract, file, stat, symlink, checksum, head, tail. However, for productivity these commands should probably be implemented relative to the remote workdir, possibly even relative to the current local working directory.
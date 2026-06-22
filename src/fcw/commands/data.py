"""Data transfer command group."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import tarfile
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, List, Optional

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

import firecrest

from fcw.core import load_config, DirectoryType, get_async_client, get_system, get_account, resolve_context, get_error_console, get_output_console

app = typer.Typer(no_args_is_help=True)
_error = get_error_console
_output = get_output_console
logger = logging.getLogger("fcw.data")

# Sync state directory
SYNC_STATE_DIR = ".fcw/sync"


def _spinner(message: str):
    """Create a progress spinner with a message."""
    p = Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=_error())
    p.add_task(message, total=None)
    return p


@contextmanager
def _log_phase(label: str):
    """Log a long-running *local* phase at INFO before and after it runs.

    Use only for fcw-side work pyfirecrest can't report — local tar build and
    local archive extraction. Remote operations (upload/extract/compress/
    download/rm) are logged by pyfirecrest's own `firecrest` logger (per-request
    + wait_for_job heartbeat), surfaced at the same verbosity by
    `configure_logging`; don't shadow them here.
    """
    logger.info("%s ...", label)
    start = time.perf_counter()
    try:
        yield
    finally:
        logger.info("%s done (%.1fs)", label, time.perf_counter() - start)


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

    logger.info("found %d local file(s) modified since %.0f under %s", len(files), since_ts, local_dir)
    return files


async def _list_remote_files(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    remote_dir: str,
) -> list[tuple[str, int, float]]:
    """List remote files under remote_dir recursively as (rel_path, size, mtime).

    Skips directories and fcw/firecrest marker files.
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

        # Skip directories, nameless and marker files
        if entry_type == "d" or not name:
            continue
        if name.startswith(".fcw") or name.startswith(".firecrest"):
            continue

        size = entry.get("size") if isinstance(entry, dict) else getattr(entry, "size", None)
        last_modified = entry.get("lastModified") if isinstance(entry, dict) else getattr(entry, "lastModified", None)
        mtime = datetime.fromisoformat(last_modified).timestamp() if last_modified else 0.0
        files.append((name, int(size) if size is not None else 0, mtime))

    return files


async def _collect_remote_files_since(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    remote_dir: str,
    since_ts: float,
) -> list[tuple[str, int]]:
    """Collect remote files modified since timestamp as (rel_path, size)."""
    entries = await _list_remote_files(client, system, remote_dir)
    files = [(name, size) for name, size, mtime in entries if mtime > since_ts]
    logger.info("found %d remote file(s) modified since %.0f under %s", len(files), since_ts, remote_dir)
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


DEFAULT_CHUNK_SIZE = "2GB"

_SIZE_UNITS = {
    "B": 1,
    "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
    "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
}


def _parse_size(value: str) -> int:
    """Parse a human-readable size like '2GB', '512MiB', or raw bytes into bytes."""
    m = re.fullmatch(r"\s*([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]*)\s*", value)
    if not m:
        raise typer.BadParameter(f"invalid size: {value!r}")
    num, unit = float(m.group(1)), (m.group(2) or "B").upper()
    if unit not in _SIZE_UNITS:
        raise typer.BadParameter(f"unknown size unit {unit!r} in {value!r}")
    return int(num * _SIZE_UNITS[unit])


def _partition_by_size(
    items: list[tuple[Any, int]], chunk_size: int
) -> Iterator[tuple[list[Any], bool]]:
    """Yield (batch, is_oversized) from items = [(payload, size), ...].

    Items with size >= chunk_size are yielded alone with is_oversized=True (the
    caller transfers them directly, without tar); the rest are greedily packed
    into batches whose cumulative size stays below chunk_size. Order preserved.
    """
    batch: list[Any] = []
    batch_bytes = 0
    for payload, size in items:
        if size >= chunk_size:
            if batch:
                yield batch, False
                batch, batch_bytes = [], 0
            yield [payload], True
            continue
        if batch and batch_bytes + size > chunk_size:
            yield batch, False
            batch, batch_bytes = [], 0
        batch.append(payload)
        batch_bytes += size
    if batch:
        yield batch, False


async def _upload_files_chunked(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    files: list[tuple[str, str]],
    remote_dir: str,
    chunk_size: int,
    follow_symlinks: bool = False,
) -> None:
    """Upload (abs_path, rel_path) files into remote_dir in size-bounded chunks.

    Small files are batched into tar archives (cumulative size <= chunk_size); any
    single file >= chunk_size is uploaded directly — pyfirecrest's S3 multipart
    streams it from disk — so no archive ever exceeds the chunk size.
    """
    await client.mkdir(system_name=system, path=remote_dir, create_parents=True)
    sized = [((abs_path, rel_path), os.path.getsize(abs_path)) for abs_path, rel_path in files]

    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "fcw_chunk.tar.gz")
        for batch, oversized in _partition_by_size(sized, chunk_size):
            if oversized:
                abs_path, rel_path = batch[0]
                remote_file = remote_dir.rstrip("/") + "/" + rel_path.replace(os.sep, "/")
                file_remote_dir = os.path.dirname(remote_file)
                await client.mkdir(system_name=system, path=file_remote_dir, create_parents=True)
                logger.info("uploading large file directly: %s (%d bytes)", rel_path, os.path.getsize(abs_path))
                await client.upload(
                    system_name=system,
                    local_file=abs_path,
                    directory=file_remote_dir,
                    filename=os.path.basename(rel_path),
                    account=account,
                    blocking=True,
                    transfer_method="s3",
                )
                continue

            with _log_phase(f"tarring {len(batch)} file(s)"):
                with tarfile.open(archive_path, "w:gz", dereference=follow_symlinks) as tar:
                    for abs_path, rel_path in batch:
                        tar.add(abs_path, arcname=rel_path)
            logger.info("archive built: %d bytes", os.path.getsize(archive_path))

            remote_archive = remote_dir.rstrip("/") + "/.fcw_upload_chunk.tar.gz"
            await client.upload(
                system_name=system,
                local_file=archive_path,
                directory=remote_dir,
                filename=".fcw_upload_chunk.tar.gz",
                account=account,
                blocking=True,
                transfer_method="s3",
            )
            await client.extract(
                system_name=system,
                source_path=remote_archive,
                target_path=remote_dir,
                account=account,
                blocking=True,
            )
            try:
                await client.rm(system_name=system, path=remote_archive, account=account, blocking=True)
            except Exception:
                pass
            os.remove(archive_path)


async def _download_files_chunked(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    files: list[tuple[str, int]],
    remote_dir: str,
    local_dir: str,
    chunk_size: int,
) -> None:
    """Download remote (rel_path, size) files into local_dir in size-bounded chunks.

    Small files are fetched via match-pattern compress into tar archives (cumulative
    size <= chunk_size); any single file >= chunk_size is downloaded directly so no
    archive ever exceeds the chunk size.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        archive_path = os.path.join(tmpdir, "fcw_chunk.tar.gz")
        for batch, oversized in _partition_by_size(files, chunk_size):
            if oversized:
                rel_path = batch[0]
                remote_file = remote_dir.rstrip("/") + "/" + rel_path.replace(os.sep, "/")
                local_dest = os.path.join(local_dir, rel_path)
                os.makedirs(os.path.dirname(local_dest) or ".", exist_ok=True)
                logger.info("downloading large file directly: %s", rel_path)
                await client.download(
                    system_name=system,
                    source_path=remote_file,
                    target_path=local_dest,
                    account=account,
                    blocking=True,
                    transfer_method="s3",
                )
                continue

            match_pattern = _build_emacs_match_pattern(batch, remote_dir)
            remote_archive = f"{remote_dir.rstrip('/')}/../.fcw_download_chunk.tar.gz"

            # fcw retries the whole compress; pyfirecrest logs each attempt's
            # request/transfer-job, this logs the fcw-level retry.
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
                    logger.warning("compress attempt %d/%d failed (%s); retrying in 2s", attempt + 1, num_attempts, e)
                    await asyncio.sleep(2)

            await client.download(
                system_name=system,
                source_path=remote_archive,
                target_path=archive_path,
                account=account,
                blocking=True,
                transfer_method="s3",
            )
            with _log_phase(f"extracting {len(batch)} file(s) into {local_dir}"):
                _extract_dir_archive(archive_path, local_dir)
            try:
                await client.rm(system_name=system, path=remote_archive, account=account, blocking=True)
            except Exception:
                pass
            os.remove(archive_path)


async def _upload_directory(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    local_dir: str,
    remote_dir: str,
    follow_symlinks: bool = False,
    chunk_size: int = _parse_size(DEFAULT_CHUNK_SIZE),
) -> None:
    """Upload a local directory into remote_dir in size-bounded chunks."""
    # Follow a symlinked directory argument to its target so we archive the real
    # contents rather than a lone (dangling-on-remote) symlink member.
    local_dir = os.path.realpath(local_dir)
    logger.info("uploading directory %s -> %s", local_dir, remote_dir)
    files = _collect_local_files_since(local_dir, 0.0, follow_symlinks)
    await _upload_files_chunked(
        client, system, account, files, remote_dir, chunk_size, follow_symlinks
    )


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
    chunk_size: int = _parse_size(DEFAULT_CHUNK_SIZE),
) -> None:
    """Download a remote directory into local_dir in size-bounded chunks."""
    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)
    logger.info("downloading directory %s -> %s", remote_dir, local_dir)
    entries = await _list_remote_files(client, system, remote_dir)
    files = [(name, size) for name, size, _ in entries]
    await _download_files_chunked(
        client, system, account, files, remote_dir, local_dir, chunk_size
    )


async def _upload_incremental(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    local_dir: str,
    remote_dir: str,
    follow_symlinks: bool = False,
    chunk_size: int = _parse_size(DEFAULT_CHUNK_SIZE),
) -> int:
    """Upload only files changed since last sync, in size-bounded chunks.

    Returns number of files uploaded.
    """
    # Dereference a symlinked directory argument to its target (matches
    # _upload_directory) so we sync the real files, not a lone symlink.
    local_dir = os.path.realpath(local_dir)
    last_sync = _read_last_sync_timestamp(local_dir, "push")
    files = _collect_local_files_since(local_dir, last_sync, follow_symlinks)

    if not files:
        return 0

    await _upload_files_chunked(
        client, system, account, files, remote_dir, chunk_size, follow_symlinks
    )
    _write_last_sync_timestamp(local_dir, "push")
    return len(files)


async def _download_incremental(
    client: firecrest.v2.AsyncFirecrest,
    system: str,
    account: str,
    remote_dir: str,
    local_dir: str,
    chunk_size: int = _parse_size(DEFAULT_CHUNK_SIZE),
) -> int:
    """Download only files changed since last sync, in size-bounded chunks.

    Returns number of files downloaded.
    """
    local_dir = os.path.abspath(local_dir)
    os.makedirs(local_dir, exist_ok=True)

    last_sync = _read_last_sync_timestamp(local_dir, "pull")
    files = await _collect_remote_files_since(client, system, remote_dir, last_sync)

    if not files:
        return 0

    await _download_files_chunked(
        client, system, account, files, remote_dir, local_dir, chunk_size
    )
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
    chunk_size: str = typer.Option(
        DEFAULT_CHUNK_SIZE, "--chunk-size",
        help="Max size per upload chunk (e.g. 2GB, 512MB); files larger than this "
             "are uploaded directly. Bounds client memory/temp usage."),
):
    """Upload local files/directories to remote storage."""
    config, system, account = resolve_context(ctx)
    chunk_bytes = _parse_size(chunk_size)

    # Validate directory types
    for path in paths:
        rel_path = os.path.relpath(path, config.workdir.local)
        if not config.can_upload(rel_path) and not force:
            dir_type = config.get_directory_type(rel_path)
            _error().print(
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
                logger.info("uploading %s -> %s", local_path, remote_path)

                if incremental and os.path.isdir(local_path):
                    with _spinner(f"Syncing {local_path}..."):
                        count = await _upload_incremental(
                            client, system, account, local_path, remote_path,
                            follow_symlinks, chunk_bytes,
                        )
                    if count > 0:
                        _error().print(f"[green]Uploaded {count} files to {remote_path}[/green]")
                    else:
                        _error().print(f"[dim]No changes in {local_path}[/dim]")
                elif os.path.isdir(local_path):
                    # Directory upload via tar
                    with _spinner(f"Uploading {local_path}..."):
                        await _upload_directory(
                            client, system, account, local_path, remote_path,
                            follow_symlinks, chunk_bytes,
                        )
                    _error().print(f"[green]Uploaded {local_path} to {remote_path}[/green]")
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
                            transfer_method="s3",
                        )
                    _error().print(f"[green]Uploaded {local_path} to {remote_path}[/green]")
            
            if not watch:
                break
            
            _error().print(f"[dim]Waiting {interval}s...[/dim]")
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
    chunk_size: str = typer.Option(
        DEFAULT_CHUNK_SIZE, "--chunk-size",
        help="Max size per download chunk (e.g. 2GB, 512MB); files larger than this "
             "are downloaded directly. Bounds client memory/temp usage."),
):
    """Download remote files/directories to local storage."""
    config, system, account = resolve_context(ctx)
    chunk_bytes = _parse_size(chunk_size)

    # Validate directory types
    for path in paths:
        if not config.can_download(path) and not force:
            dir_type = config.get_directory_type(path)
            _error().print(
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
                logger.info("downloading %s -> %s", remote_path, local_path)

                if incremental:
                    with _spinner(f"Syncing {rel_path}..."):
                        count = await _download_incremental(
                            client, system, account, remote_path, local_path, chunk_bytes
                        )
                    if count > 0:
                        _error().print(f"[green]Downloaded {count} files from {remote_path}[/green]")
                    else:
                        _error().print(f"[dim]No changes in {rel_path}[/dim]")
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
                                client, system, account, remote_path, local_path, chunk_bytes
                            )
                        _error().print(
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
                                transfer_method="s3",
                            )
                        _error().print(
                            f"[green]Downloaded {remote_path} to {local_path}[/green]"
                        )
            
            if not watch:
                break
            
            _error().print(f"[dim]Waiting {interval}s...[/dim]")
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
                _output().print(f"[blue]{name}/[/blue]")
            else:
                _output().print(f"{name}  ({size} bytes)")
    
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
            _error().print(f"[green]Removed {remote_path}[/green]")
    
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

    _output().print(table)

# FIXME: ls has already been implemented, others from pyfirerest not, e.g. mkdir, mv, chmod, chown, cp, compress, extract, file, stat, symlink, checksum, head, tail. However, for productivity these commands should probably be implemented relative to the remote workdir, possibly even relative to the current local working directory.
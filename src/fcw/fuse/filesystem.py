"""FUSE filesystem implementation using pyfuse3 and FirecREST.

This is a refactored version of the original firecrest-pyfuse3.py with:
- Native async FirecREST client (no thread wrappers)
- Attribute and directory caching with TTL
- Read caching to local temp files
- Configurable cache settings
- Better error mapping to POSIX codes
"""

from __future__ import annotations

import asyncio
import errno
import logging
import os
import re
import shutil
import stat
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import IO, TYPE_CHECKING, Any, Optional

import firecrest
import pyfuse3
import pyfuse3.asyncio
from cachetools import TTLCache
from firecrest import FirecrestException
from firecrest.FirecrestException import HeaderException, UnauthorizedException

from fcw.core.client import _get_auth, _get_firecrest_url, extract_job_id

if TYPE_CHECKING:
    from firecrest.v2 import AsyncFirecrest

logger = logging.getLogger("fcw.fuse")

# Warn (but don't refuse) when a file larger than this is opened with
# non-O_TRUNC write intent: the whole file gets downloaded, edited
# locally, and re-uploaded on flush, which can take minutes over a WAN.
WRITE_WARN_SIZE = 100 * 1024 * 1024  # 100 MB

# Matches cache dirs created by this module: fcw_cache_{pid}_xxxxxx
_CACHE_DIR_RE = re.compile(r"^fcw_cache_(\d+)_")


def _cleanup_stale_cache_dirs() -> None:
    """Remove orphaned fcw_cache_{pid}_* dirs whose owning process is dead.

    Handles the SIGKILL/umount -l case where the runner's finally-block
    cleanup never ran.
    """
    tmp = tempfile.gettempdir()
    try:
        entries = os.listdir(tmp)
    except OSError:
        return
    for name in entries:
        match = _CACHE_DIR_RE.match(name)
        if not match:
            continue
        try:
            pid = int(match.group(1))
        except ValueError:
            continue
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            path = os.path.join(tmp, name)
            logger.info(f"Removing stale cache dir from dead PID {pid}: {path}")
            shutil.rmtree(path, ignore_errors=True)
        except (PermissionError, OSError):
            # PID exists but we can't signal it — leave it alone
            continue


@dataclass
class _InodeOpenState:
    """Shared write buffer and metadata for all file handles on one inode.

    The correctness goal: two concurrent writers on the same inode must not
    end up with independent buffers, or flush() will silently clobber one of
    their writes (last-write-wins). All opens on the same inode share one
    buffer and one refcount.

    ``download_lock`` serializes the lazy first-download path: pyfuse3 can
    dispatch multiple read()/write() calls concurrently on the same inode,
    and without a lock both would see ``cached == False`` and race into
    duplicate ``client.download()`` calls writing the same temp file.
    """
    path: str
    buffer: IO[bytes]
    refcount: int = 1
    dirty: bool = False
    cached: bool = False
    fhs: set[int] = field(default_factory=set)
    download_lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _parse_permissions(perm_str: str) -> int:
    """Convert permission string like 'rwxr-xr-x' to octal mode bits."""
    if len(perm_str) < 9:
        return 0o644
    mode = 0
    for i, char in enumerate(perm_str[:9]):
        if char in "rwxsStT":
            mode |= 1 << (8 - i)
    return mode


def _parse_timestamp(ts_str: str) -> int:
    """Parse a timestamp string to epoch nanoseconds. Tries ISO 8601 then common formats."""
    if not ts_str:
        return 0
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts_str[:19], fmt[:len(fmt)])
            return int(dt.replace(tzinfo=timezone.utc).timestamp()) * 10**9
        except ValueError:
            continue
    try:
        return int(float(ts_str)) * 10**9
    except (ValueError, TypeError):
        return 0


class FirecrestFS(pyfuse3.Operations):
    """FUSE filesystem backed by FirecREST API.

    Features:
    - Async operations using native AsyncFirecrest client
    - TTL-based caching for attributes and directory listings
    - Local file caching for reads
    - Write buffering with upload on flush
    """

    def __init__(
        self,
        client: "AsyncFirecrest",
        system: str,
        remote_root: str,
        account: Optional[str] = None,
        cache_ttl: int = 5,
        read_only: bool = False,
        statfs_partition: Optional[str] = None,
    ):
        super().__init__()
        self.client = client
        self.system = system
        self.remote_root = remote_root.rstrip("/")
        self.account = account
        self.cache_ttl = cache_ttl
        self.read_only = read_only
        self.statfs_partition = statfs_partition

        # Inode management
        self.inode_map: dict[int, str] = {pyfuse3.ROOT_INODE: self.remote_root}
        self.path_to_inode: dict[str, int] = {self.remote_root: pyfuse3.ROOT_INODE}
        self.next_inode = pyfuse3.ROOT_INODE + 1

        # Caching
        self.attr_cache: TTLCache = TTLCache(maxsize=10000, ttl=cache_ttl)
        self.dir_cache: TTLCache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.negative_cache: TTLCache = TTLCache(maxsize=10000, ttl=cache_ttl)
        self.readdir_cache: TTLCache = TTLCache(maxsize=100, ttl=cache_ttl)

        # Clean up stale cache dirs from dead processes, then make ours.
        _cleanup_stale_cache_dirs()
        self.read_cache_dir = tempfile.mkdtemp(prefix=f"fcw_cache_{os.getpid()}_")

        # Per-inode shared open state (refcounted) + fh->inode reverse lookup
        self.inode_state: dict[int, _InodeOpenState] = {}
        self.fh_to_inode: dict[int, int] = {}
        self.next_fh = 1

        # statfs: lazy remote fetch via one-shot job on statfs_partition
        self.statfs_cache: Optional[dict[str, int]] = None
        self.statfs_task: Optional[asyncio.Task] = None

        logger.info(f"Initialized FirecrestFS: {system}:{remote_root}")
        logger.info(f"Cache TTL: {cache_ttl}s, Read-only: {read_only}")
        if statfs_partition:
            logger.info(f"statfs partition: {statfs_partition}")

    def _get_inode(self, path: str) -> int:
        """Get or create inode for path."""
        if path in self.path_to_inode:
            return self.path_to_inode[path]

        inode = self.next_inode
        self.next_inode += 1
        self.inode_map[inode] = path
        self.path_to_inode[path] = inode
        return inode

    def _inode_to_path(self, inode: int) -> str:
        """Convert inode to path."""
        if inode not in self.inode_map:
            raise pyfuse3.FUSEError(errno.ENOENT)
        return self.inode_map[inode]

    def _stat_to_attr(self, fc_stat: dict, inode: int) -> pyfuse3.EntryAttributes:
        """Convert FirecREST stat to pyfuse3 EntryAttributes."""
        entry = pyfuse3.EntryAttributes()

        def get_val(obj, key, default=0):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        entry.st_ino = inode
        entry.generation = 0
        entry.entry_timeout = self.cache_ttl
        entry.attr_timeout = self.cache_ttl

        entry.st_mode = int(get_val(fc_stat, "mode", 0))
        entry.st_nlink = int(get_val(fc_stat, "nlink", 1))
        entry.st_uid = os.getuid()
        entry.st_gid = os.getgid()
        entry.st_rdev = int(get_val(fc_stat, "dev", 0))
        entry.st_size = int(get_val(fc_stat, "size", 0))

        entry.st_atime_ns = int(get_val(fc_stat, "atime", 0)) * 10**9
        entry.st_mtime_ns = int(get_val(fc_stat, "mtime", 0)) * 10**9
        entry.st_ctime_ns = int(get_val(fc_stat, "ctime", 0)) * 10**9

        # Fallback if mode is missing
        if entry.st_mode == 0:
            entry.st_mode = stat.S_IFREG | 0o644

        return entry

    def _map_error(self, e: Exception) -> int:
        """Map FirecREST exception to POSIX errno."""
        if isinstance(e, UnauthorizedException):
            return errno.EACCES

        if isinstance(e, HeaderException):
            headers = e.responses[-1].headers
            if "X-Not-Found" in headers or "X-Invalid-Path" in headers:
                return errno.ENOENT
            if "X-Permission-Denied" in headers:
                return errno.EACCES
            if "X-Not-A-Directory" in headers:
                return errno.ENOTDIR
            if "X-Timeout" in headers:
                return errno.ETIMEDOUT

        if isinstance(e, FirecrestException) and e.responses:
            status = e.responses[-1].status_code
            if status == 404:
                return errno.ENOENT
            if status == 403:
                return errno.EACCES
            if status == 409:
                return errno.EEXIST

        return errno.EIO

    # --- Async wrappers for FirecREST calls ---

    async def _fc_stat(self, path: str) -> dict:
        """Get file stats with caching."""
        if path in self.attr_cache:
            return self.attr_cache[path]
        if path in self.negative_cache:
            raise pyfuse3.FUSEError(errno.ENOENT)

        result = await self.client.stat(system_name=self.system, path=path)

        # Convert to dict if needed
        if not isinstance(result, dict):
            result = {
                "mode": getattr(result, "mode", 0),
                "nlink": getattr(result, "nlink", 1),
                "uid": getattr(result, "uid", os.getuid()),
                "gid": getattr(result, "gid", os.getgid()),
                "size": getattr(result, "size", 0),
                "atime": getattr(result, "atime", 0),
                "mtime": getattr(result, "mtime", 0),
                "ctime": getattr(result, "ctime", 0),
            }

        logger.debug(
            f"stat {os.path.basename(path)}: "
            f"mtime={result.get('mtime')} atime={result.get('atime')} "
            f"ctime={result.get('ctime')} size={result.get('size')}"
        )
        self.attr_cache[path] = result
        return result

    async def _fc_list_files(self, path: str) -> list:
        """List directory with caching."""
        if path in self.dir_cache:
            return self.dir_cache[path]

        result = await self.client.list_files(
            system_name=self.system,
            path=path,
            recursive=False,
            show_hidden=True,
        )

        self.dir_cache[path] = result
        return result

    # --- FUSE Operations ---

    async def statfs(self, ctx):
        """Return filesystem statistics.

        If ``statfs_partition`` is configured, fetches real values from the
        remote via a one-shot ``stat -f`` job and caches them for the mount's
        lifetime. Returns fallback big-fake numbers until the job completes
        so tools that pre-check free space don't spuriously hit ENOSPC.
        """
        # Lazy task kickoff: first statfs() call after mount. We can't start
        # tasks from __init__ since no event loop is running yet.
        if (
            self.statfs_partition
            and self.statfs_cache is None
            and self.statfs_task is None
        ):
            self.statfs_task = asyncio.create_task(self._fetch_statfs())

        stats = pyfuse3.StatvfsData()
        stats.f_namemax = 255

        if self.statfs_cache is not None:
            c = self.statfs_cache
            stats.f_bsize = c["f_bsize"]
            stats.f_frsize = c["f_frsize"]
            stats.f_blocks = c["f_blocks"]
            stats.f_bfree = c["f_bfree"]
            stats.f_bavail = c["f_bavail"]
            stats.f_files = c["f_files"]
            stats.f_ffree = c["f_ffree"]
            return stats

        # Fallback: big numbers so tools don't refuse writes pre-flight.
        stats.f_bsize = 65536
        stats.f_frsize = 65536
        stats.f_blocks = 1 << 40
        stats.f_bfree = 1 << 40
        stats.f_bavail = 1 << 40
        stats.f_files = 1 << 32
        stats.f_ffree = 1 << 32
        return stats

    async def _fetch_statfs(self) -> None:
        """Fetch real statvfs data by running ``stat -f`` on the xfer partition.

        On any failure (submit error, timeout, parse error), logs a warning
        and leaves ``statfs_cache`` unset so ``statfs()`` keeps serving the
        fallback numbers. Does not retry — remount to retry.
        """
        script = (
            "#!/bin/bash\n"
            f"#SBATCH --partition={self.statfs_partition}\n"
            "#SBATCH --time=00:01:00\n"
            "#SBATCH --output=statfs.out\n"
            f"stat -f --format='%s %S %b %f %a %c %d' {self.remote_root!r}\n"
        )
        try:
            submit_kwargs: dict[str, Any] = {
                "system_name": self.system,
                "working_dir": self.remote_root,
                "script_str": script,
            }
            if self.account:
                submit_kwargs["account"] = self.account
            resp = await self.client.submit(**submit_kwargs)
            job_id = extract_job_id(resp)
            if not job_id:
                logger.warning(f"statfs: submit returned no job id: {resp}")
                return

            await self.client.wait_for_job(
                system_name=self.system, job_id=job_id, timeout=60.0
            )

            stdout_path = os.path.join(self.remote_root, "statfs.out")
            output = await self.client.view(
                system_name=self.system, path=stdout_path
            )
            parts = output.strip().split()
            if len(parts) != 7:
                logger.warning(
                    f"statfs: unexpected output from stat -f: {output!r}"
                )
                return
            bsize, frsize, blocks, bfree, bavail, files, ffree = (
                int(p) for p in parts
            )
            self.statfs_cache = {
                "f_bsize": bsize,
                "f_frsize": frsize,
                "f_blocks": blocks,
                "f_bfree": bfree,
                "f_bavail": bavail,
                "f_files": files,
                "f_ffree": ffree,
            }
            logger.info(
                f"statfs: {blocks * frsize / 1e12:.2f} TB total, "
                f"{bavail * frsize / 1e12:.2f} TB available"
            )
        except Exception as e:
            logger.warning(
                f"statfs: failed to fetch real values ({type(e).__name__}: {e}); "
                f"using fallback numbers"
            )

    async def lookup(self, parent_inode: int, name: bytes, ctx=None):
        """Look up a directory entry by name."""
        name_str = name.decode("utf-8")
        parent_path = self._inode_to_path(parent_inode)
        path = os.path.join(parent_path, name_str)

        logger.debug(f"lookup: {path}")

        try:
            fc_stat = await self._fc_stat(path)
            inode = self._get_inode(path)
            return self._stat_to_attr(fc_stat, inode)
        except pyfuse3.FUSEError:
            raise
        except FirecrestException as e:
            err = self._map_error(e)
            if err == errno.ENOENT:
                self.negative_cache[path] = True
            raise pyfuse3.FUSEError(err)
        except Exception as e:
            logger.error(f"lookup error for {path}: {e}")
            raise pyfuse3.FUSEError(errno.EIO)

    async def getattr(self, inode: int, ctx=None):
        """Get file attributes."""
        path = self._inode_to_path(inode)
        logger.debug(f"getattr: {path}")

        # If the file is open with a dirty buffer, report buffer state
        # instead of (stale) remote attrs so editors see the new size.
        state = self.inode_state.get(inode)
        if state is not None and state.dirty:
            entry = pyfuse3.EntryAttributes()
            entry.st_ino = inode
            entry.st_mode = stat.S_IFREG | 0o644
            entry.st_size = os.fstat(state.buffer.fileno()).st_size
            now_ns = int(time.time() * 10**9)
            entry.st_mtime_ns = now_ns
            entry.st_atime_ns = now_ns
            entry.st_ctime_ns = now_ns
            entry.st_uid = os.getuid()
            entry.st_gid = os.getgid()
            return entry

        try:
            fc_stat = await self._fc_stat(path)
            return self._stat_to_attr(fc_stat, inode)
        except FirecrestException as e:
            raise pyfuse3.FUSEError(self._map_error(e))

    async def setattr(self, inode: int, attr, fields, fh, ctx):
        """Set file attributes (truncate, chmod, chown)."""
        path = self._inode_to_path(inode)

        # Check if file only exists locally (created but not yet uploaded)
        state = self.inode_state.get(inode)
        local_only = state is not None and state.dirty and not state.cached

        # Handle truncate
        if fields.update_size:
            if state is not None:
                state.buffer.truncate(attr.st_size)
                state.dirty = True
            else:
                try:
                    with tempfile.NamedTemporaryFile(delete=False,
                                                     dir=self.read_cache_dir) as tmp:
                        tmp_path = tmp.name

                    if attr.st_size > 0:
                        await self.client.download(
                            system_name=self.system,
                            source_path=path,
                            target_path=tmp_path,
                            account=self.account,
                            blocking=True,
                            transfer_method="s3",
                        )

                    os.truncate(tmp_path, attr.st_size)

                    target_dir = os.path.dirname(path)
                    filename = os.path.basename(path)
                    await self.client.upload(
                        system_name=self.system,
                        local_file=tmp_path,
                        directory=target_dir,
                        filename=filename,
                        account=self.account,
                        blocking=True,
                        transfer_method="s3",
                    )
                except Exception as e:
                    logger.error(f"Remote truncate failed: {e}")
                    raise pyfuse3.FUSEError(self._map_error(e))
                finally:
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)

        # Handle chmod — skip for files not yet uploaded
        if fields.update_mode and not local_only:
            mode_octal = f"{stat.S_IMODE(attr.st_mode):03o}"
            try:
                await self.client.chmod(
                    system_name=self.system, path=path, mode=mode_octal
                )
            except Exception as e:
                logger.error(f"chmod failed: {e}")
                raise pyfuse3.FUSEError(self._map_error(e))

        # Handle chown — skip for files not yet uploaded
        if (fields.update_uid or fields.update_gid) and not local_only:
            owner = str(attr.st_uid) if fields.update_uid else ""
            group = str(attr.st_gid) if fields.update_gid else ""
            try:
                await self.client.chown(
                    system_name=self.system, path=path, owner=owner, group=group
                )
            except Exception as e:
                logger.warning(f"chown failed for {path}: {e}")
                raise pyfuse3.FUSEError(errno.EPERM)

        # Invalidate cached attrs
        if path in self.attr_cache:
            del self.attr_cache[path]

        return await self.getattr(inode, ctx)

    async def opendir(self, inode: int, ctx):
        """Open a directory for reading."""
        return inode

    async def readdir(self, inode: int, start_id: int, token):
        """Read directory contents."""
        path = self._inode_to_path(inode)
        logger.debug(f"readdir: {path} start_id={start_id}")

        # Fast path: serve from cached entries list (avoids rebuilding the
        # full EntryAttributes list on every continuation call when
        # pyfuse3's reply buffer fills up on large directories).
        cached_entries = self.readdir_cache.get(inode)
        if cached_entries is not None:
            for i, (name_bytes, attr) in enumerate(cached_entries[start_id:], start_id):
                if attr.st_ino == 0:
                    continue
                if not pyfuse3.readdir_reply(token, name_bytes, attr, i + 1):
                    break
            return

        try:
            files = await self._fc_list_files(path)
        except Exception as e:
            logger.error(f"readdir error for {path}: {e}")
            raise pyfuse3.FUSEError(errno.EIO)

        entries: list[tuple[bytes, pyfuse3.EntryAttributes]] = []

        # Add . and ..
        attr_dot = pyfuse3.EntryAttributes()
        attr_dot.st_ino = inode
        attr_dot.st_mode = stat.S_IFDIR | 0o755
        entries.append((b".", attr_dot))

        attr_dotdot = pyfuse3.EntryAttributes()
        if inode == pyfuse3.ROOT_INODE:
            parent_inode = pyfuse3.ROOT_INODE
        else:
            parent_path = os.path.dirname(path)
            parent_inode = self.path_to_inode.get(parent_path, pyfuse3.ROOT_INODE)
        attr_dotdot.st_ino = parent_inode
        attr_dotdot.st_mode = stat.S_IFDIR | 0o755
        entries.append((b"..", attr_dotdot))

        # Add directory contents
        local_uid = os.getuid()
        local_gid = os.getgid()
        for f in files:
            name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
            ftype = f.get("type") if isinstance(f, dict) else getattr(f, "type", None)

            if not name or name in (".", ".."):
                continue

            # Parse permissions from API response, fall back to defaults
            perm_str = (f.get("permissions") if isinstance(f, dict)
                        else getattr(f, "permissions", None))
            if perm_str:
                perm_bits = _parse_permissions(perm_str)
            else:
                perm_bits = 0o755 if ftype == "d" else 0o644

            if ftype == "d":
                mode = stat.S_IFDIR | perm_bits
            elif ftype == "l":
                mode = stat.S_IFLNK | 0o777
            else:
                mode = stat.S_IFREG | perm_bits

            child_path = os.path.join(path, name)
            child_inode = self._get_inode(child_path)

            # Parse size and timestamp from API response
            size_str = (f.get("size") if isinstance(f, dict)
                        else getattr(f, "size", "0"))
            try:
                file_size = int(size_str)
            except (ValueError, TypeError):
                file_size = 0

            last_mod = (f.get("lastModified") if isinstance(f, dict)
                        else getattr(f, "lastModified", None))
            mtime_ns = _parse_timestamp(last_mod) if last_mod else 0

            attr = pyfuse3.EntryAttributes()
            attr.st_ino = child_inode
            attr.st_mode = mode
            attr.st_nlink = 2 if ftype == "d" else 1
            attr.st_uid = local_uid
            attr.st_gid = local_gid
            attr.st_size = file_size
            attr.st_atime_ns = mtime_ns
            attr.st_mtime_ns = mtime_ns
            attr.st_ctime_ns = mtime_ns
            attr.entry_timeout = self.cache_ttl
            attr.attr_timeout = 0

            entries.append((name.encode("utf-8"), attr))

        # Cache the fully-built entries list so continuation calls and
        # subsequent readdirs within TTL don't rebuild it.
        self.readdir_cache[inode] = entries

        # Send replies
        for i, (name_bytes, attr) in enumerate(entries[start_id:], start_id):
            if attr.st_ino == 0:
                continue
            if not pyfuse3.readdir_reply(token, name_bytes, attr, i + 1):
                break

    async def releasedir(self, fh: int):
        """Release directory handle."""
        pass

    async def open(self, inode: int, flags: int, ctx):
        """Open a file."""
        if self.read_only and (flags & (os.O_WRONLY | os.O_RDWR | os.O_TRUNC)):
            raise pyfuse3.FUSEError(errno.EROFS)

        path = self._inode_to_path(inode)
        is_truncated = bool(flags & os.O_TRUNC)
        write_intent = bool(flags & (os.O_WRONLY | os.O_RDWR))

        # Soft warning for large write opens: the whole file will be
        # downloaded, edited, and re-uploaded on flush. We don't refuse —
        # users can still proceed — but they should know it'll be slow.
        # O_TRUNC opens skip this since we don't download anything.
        if write_intent and not is_truncated:
            try:
                fc_stat = await self._fc_stat(path)
                size = int(
                    fc_stat.get("size", 0)
                    if isinstance(fc_stat, dict)
                    else getattr(fc_stat, "size", 0)
                )
                if size > WRITE_WARN_SIZE:
                    logger.warning(
                        f"Opening {path} ({size / 1e6:.0f} MB) for write: "
                        f"the full file will be downloaded, edited locally, "
                        f"and re-uploaded on flush. This may take a while."
                    )
            except FirecrestException:
                # File may not exist yet — let subsequent ops handle it.
                pass

        fh = self.next_fh
        self.next_fh += 1

        state = self.inode_state.get(inode)
        if state is not None:
            # Second (or later) open of this inode: share the buffer.
            state.refcount += 1
            state.fhs.add(fh)
            self.fh_to_inode[fh] = inode
            if is_truncated:
                state.buffer.seek(0)
                state.buffer.truncate(0)
                state.dirty = True
                state.cached = True
        else:
            buf = tempfile.NamedTemporaryFile(delete=False, dir=self.read_cache_dir)
            state = _InodeOpenState(
                path=path,
                buffer=buf,
                refcount=1,
                dirty=is_truncated,
                cached=is_truncated,  # O_TRUNC short-circuits download
                fhs={fh},
            )
            self.inode_state[inode] = state
            self.fh_to_inode[fh] = inode

        return pyfuse3.FileInfo(fh=fh, direct_io=True)

    async def read(self, fh: int, offset: int, length: int):
        """Read from a file."""
        inode = self.fh_to_inode.get(fh)
        if inode is None:
            raise pyfuse3.FUSEError(errno.EBADF)
        state = self.inode_state[inode]

        # Fast path: buffer already populated.
        if state.dirty or state.cached:
            state.buffer.seek(offset)
            return state.buffer.read(length)

        # Slow path: lazy first download. Serialized so two concurrent
        # FUSE reads on the same inode don't both await client.download()
        # into the same temp file.
        async with state.download_lock:
            # Double-check after acquiring the lock — another task may
            # have completed the download while we were waiting.
            if state.dirty or state.cached:
                state.buffer.seek(offset)
                return state.buffer.read(length)

            logger.debug(f"Downloading {state.path} to cache")
            try:
                buf_name = state.buffer.name
                await self.client.download(
                    system_name=self.system,
                    source_path=state.path,
                    target_path=buf_name,
                    account=self.account,
                    blocking=True,
                    transfer_method="s3",
                )

                # Reopen buffer for read+write (download replaced the file)
                state.buffer.close()
                state.buffer = open(buf_name, "rb+")
                state.cached = True
            except Exception as e:
                logger.error(f"Read error: {e}")
                raise pyfuse3.FUSEError(self._map_error(e))

        state.buffer.seek(offset)
        return state.buffer.read(length)

    async def write(self, fh: int, offset: int, buf: bytes):
        """Write to a file."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        inode = self.fh_to_inode.get(fh)
        if inode is None:
            raise pyfuse3.FUSEError(errno.EBADF)
        state = self.inode_state[inode]

        # Lazy first-download path is serialized via download_lock to
        # prevent two concurrent writes from both racing into
        # client.download() against the same temp file.
        if not state.dirty and not state.cached:
            async with state.download_lock:
                if not state.dirty and not state.cached:
                    logger.debug(f"Downloading {state.path} before write")
                    try:
                        buf_name = state.buffer.name
                        state.buffer.close()

                        await self.client.download(
                            system_name=self.system,
                            source_path=state.path,
                            target_path=buf_name,
                            account=self.account,
                            blocking=True,
                            transfer_method="s3",
                        )

                        state.buffer = open(buf_name, "rb+")
                        state.cached = True
                    except Exception as e:
                        logger.error(f"Download for write failed: {e}")
                        raise pyfuse3.FUSEError(self._map_error(e))

        state.buffer.seek(offset)
        state.buffer.write(buf)
        state.dirty = True

        return len(buf)

    async def flush(self, fh: int):
        """Flush file to storage."""
        inode = self.fh_to_inode.get(fh)
        if inode is None:
            return
        state = self.inode_state[inode]
        if not state.dirty:
            return

        logger.info(f"Uploading {state.path}")

        try:
            state.buffer.flush()
            os.fsync(state.buffer.fileno())

            path = state.path
            target_dir = os.path.dirname(path)
            filename = os.path.basename(path)

            state.dirty = False

            await self.client.upload(
                system_name=self.system,
                local_file=state.buffer.name,
                directory=target_dir,
                filename=filename,
                account=self.account,
                blocking=True,
                transfer_method="s3",
            )

            # Re-stat to populate cache with new mtime so editors
            # don't see a stale mtime on their post-save check
            if path in self.attr_cache:
                del self.attr_cache[path]
            await self._fc_stat(path)

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            state.dirty = True
            raise pyfuse3.FUSEError(self._map_error(e))

    async def release(self, fh: int):
        """Release file handle."""
        inode = self.fh_to_inode.pop(fh, None)
        if inode is None:
            return
        state = self.inode_state.get(inode)
        if state is None:
            return
        state.fhs.discard(fh)
        state.refcount -= 1
        if state.refcount > 0:
            return
        # Last handle on this inode: close and unlink the shared buffer
        try:
            buf_name = state.buffer.name
            state.buffer.close()
            if os.path.exists(buf_name):
                os.unlink(buf_name)
        except Exception as e:
            logger.warning(f"Error cleaning up inode {inode}: {e}")
        self.inode_state.pop(inode, None)

    async def mkdir(self, parent_inode: int, name: bytes, mode: int, ctx):
        """Create a directory."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        path = os.path.join(self._inode_to_path(parent_inode), name.decode("utf-8"))

        try:
            await self.client.mkdir(system_name=self.system, path=path)

            inode = self._get_inode(path)
            entry = pyfuse3.EntryAttributes()
            entry.st_ino = inode
            entry.generation = 0
            entry.entry_timeout = self.cache_ttl
            entry.attr_timeout = self.cache_ttl
            entry.st_mode = stat.S_IFDIR | mode
            entry.st_nlink = 2
            entry.st_uid = ctx.uid
            entry.st_gid = ctx.gid

            now_ns = int(time.time() * 10**9)
            entry.st_atime_ns = now_ns
            entry.st_mtime_ns = now_ns
            entry.st_ctime_ns = now_ns

            # Invalidate parent dir + readdir cache
            parent_path = self._inode_to_path(parent_inode)
            self.dir_cache.pop(parent_path, None)
            self.readdir_cache.pop(parent_inode, None)

            return entry

        except Exception as e:
            logger.error(f"mkdir failed: {e}")
            raise pyfuse3.FUSEError(self._map_error(e))

    async def unlink(self, parent_inode: int, name: bytes, ctx):
        """Remove a file."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        path = os.path.join(self._inode_to_path(parent_inode), name.decode("utf-8"))

        try:
            await self.client.rm(
                system_name=self.system,
                path=path,
                account=self.account,
                blocking=True,
            )

            # Cleanup inode
            if path in self.path_to_inode:
                inode = self.path_to_inode.pop(path)
                self.inode_map.pop(inode, None)

            # Invalidate caches
            self.attr_cache.pop(path, None)
            parent_path = self._inode_to_path(parent_inode)
            self.dir_cache.pop(parent_path, None)
            self.readdir_cache.pop(parent_inode, None)

        except Exception as e:
            logger.error(f"unlink failed: {e}")
            raise pyfuse3.FUSEError(self._map_error(e))

    async def rmdir(self, parent_inode: int, name: bytes, ctx):
        """Remove a directory."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        path = os.path.join(self._inode_to_path(parent_inode), name.decode("utf-8"))

        try:
            # Check if empty — bypass cache to avoid stale ENOTEMPTY
            files = await self.client.list_files(
                system_name=self.system,
                path=path,
                recursive=False,
                show_hidden=True,
            )
            for f in files:
                fname = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
                if fname and fname not in (".", ".."):
                    raise pyfuse3.FUSEError(errno.ENOTEMPTY)

            await self.client.rm(
                system_name=self.system,
                path=path,
                account=self.account,
                blocking=True,
            )

            # Cleanup
            if path in self.path_to_inode:
                inode = self.path_to_inode.pop(path)
                self.inode_map.pop(inode, None)
                self.readdir_cache.pop(inode, None)
            self.attr_cache.pop(path, None)
            self.dir_cache.pop(path, None)
            parent_path = self._inode_to_path(parent_inode)
            self.dir_cache.pop(parent_path, None)
            self.readdir_cache.pop(parent_inode, None)

        except pyfuse3.FUSEError:
            raise
        except Exception as e:
            logger.error(f"rmdir failed: {e}")
            raise pyfuse3.FUSEError(self._map_error(e))

    async def rename(self, parent_inode_old: int, name_old: bytes,
                     parent_inode_new: int, name_new: bytes, flags: int, ctx):
        """Rename/move a file or directory."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        old_path = os.path.join(self._inode_to_path(parent_inode_old), name_old.decode("utf-8"))
        new_path = os.path.join(self._inode_to_path(parent_inode_new), name_new.decode("utf-8"))

        try:
            await self.client.mv(
                system_name=self.system,
                source_path=old_path,
                target_path=new_path,
                account=self.account,
            )

            # Single-pass update of path_to_inode + inode_map + attr/dir/neg
            # caches for the renamed path and all descendants. path_to_inode
            # is the canonical registry — the attr/dir/negative caches are
            # populated via paths that go through _get_inode, so iterating
            # it once and clearing the side caches in the same loop is safe.
            # A prefix-sorted structure (e.g. sortedcontainers.SortedDict)
            # would enable O(log N + k) but isn't worth the dep today.
            old_prefix = old_path + "/"
            affected = [
                p for p in self.path_to_inode
                if p == old_path or p.startswith(old_prefix)
            ]
            for path in affected:
                inode = self.path_to_inode.pop(path)
                updated_path = new_path + path[len(old_path):]
                self.path_to_inode[updated_path] = inode
                self.inode_map[inode] = updated_path
                self.attr_cache.pop(path, None)
                self.dir_cache.pop(path, None)
                self.negative_cache.pop(path, None)
            self.negative_cache.pop(new_path, None)

            # Invalidate readdir cache entries for both parent dirs
            old_parent_inode = self.path_to_inode.get(
                os.path.dirname(old_path)
            )
            new_parent_inode = self.path_to_inode.get(
                os.path.dirname(new_path)
            )
            if old_parent_inode is not None:
                self.readdir_cache.pop(old_parent_inode, None)
            if new_parent_inode is not None:
                self.readdir_cache.pop(new_parent_inode, None)

        except Exception as e:
            logger.error(f"rename failed: {e}")
            raise pyfuse3.FUSEError(self._map_error(e))

    async def create(self, parent_inode: int, name: bytes, mode: int, flags: int, ctx):
        """Create and open a new file."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        path = os.path.join(self._inode_to_path(parent_inode), name.decode("utf-8"))

        # Invalidate negative cache since file is being created
        if path in self.negative_cache:
            del self.negative_cache[path]

        fh = self.next_fh
        self.next_fh += 1
        inode = self._get_inode(path)

        buf = tempfile.NamedTemporaryFile(delete=False, dir=self.read_cache_dir)
        self.inode_state[inode] = _InodeOpenState(
            path=path,
            buffer=buf,
            refcount=1,
            dirty=True,
            cached=True,  # new empty file — nothing to download
            fhs={fh},
        )
        self.fh_to_inode[fh] = inode

        entry = pyfuse3.EntryAttributes()
        entry.st_ino = inode
        entry.generation = 0
        entry.entry_timeout = self.cache_ttl
        entry.attr_timeout = self.cache_ttl
        entry.st_mode = stat.S_IFREG | mode
        entry.st_nlink = 1
        entry.st_uid = ctx.uid
        entry.st_gid = ctx.gid
        entry.st_size = 0

        now_ns = int(time.time() * 10**9)
        entry.st_atime_ns = now_ns
        entry.st_mtime_ns = now_ns
        entry.st_ctime_ns = now_ns

        # Invalidate parent dir + readdir cache
        parent_path = self._inode_to_path(parent_inode)
        self.dir_cache.pop(parent_path, None)
        self.readdir_cache.pop(parent_inode, None)

        return (pyfuse3.FileInfo(fh=fh, direct_io=True), entry)


def run_filesystem(
    mountpoint: str,
    remote_root: str,
    system: str,
    account: str | None = None,
    cache_ttl: int = 5,
    read_only: bool = False,
    allow_other: bool = False,
    debug: bool = False,
    statfs_partition: str | None = None,
):
    """Run the FUSE filesystem.

    Args:
        mountpoint: Local directory to mount at
        remote_root: Remote directory root
        system: FirecREST system name
        account: SLURM account (optional)
        cache_ttl: Cache TTL in seconds
        read_only: Mount read-only
        allow_other: Allow other users to access mount
        debug: Enable debug logging
        statfs_partition: SLURM partition to run ``stat -f`` on for real df
    """
    if debug:
        logging.basicConfig(level=logging.DEBUG)
        logging.getLogger("fcw.fuse").setLevel(logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Create async client
    client = firecrest.v2.AsyncFirecrest(
        firecrest_url=_get_firecrest_url(),
        authorization=_get_auth(),
    )

    fs = FirecrestFS(
        client=client,
        system=system,
        remote_root=remote_root,
        account=account,
        cache_ttl=cache_ttl,
        read_only=read_only,
        statfs_partition=statfs_partition,
    )

    fuse_options = set(pyfuse3.default_options)
    fuse_options.add("fsname=fcw")
    if allow_other:
        fuse_options.add("allow_other")
    if debug:
        fuse_options.add("debug")

    pyfuse3.asyncio.enable()
    pyfuse3.init(fs, mountpoint, fuse_options)

    async def _main() -> None:
        try:
            await pyfuse3.main()
        finally:
            # Cancel the background statfs task on the same loop it
            # was created on, to avoid "Task was destroyed but it is
            # pending" warnings on shutdown.
            if fs.statfs_task is not None and not fs.statfs_task.done():
                fs.statfs_task.cancel()
                try:
                    await fs.statfs_task
                except (asyncio.CancelledError, Exception):
                    pass

    try:
        asyncio.run(_main())
    except KeyboardInterrupt:
        pass
    finally:
        pyfuse3.close()
        asyncio.run(client.close_session())
        shutil.rmtree(fs.read_cache_dir, ignore_errors=True)

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
import shutil
import stat
import tempfile
import time
from typing import TYPE_CHECKING, Optional

import firecrest
import pyfuse3
import pyfuse3.asyncio
from cachetools import TTLCache
from firecrest import FirecrestException
from firecrest.FirecrestException import HeaderException, UnauthorizedException

from fcw.core.client import _get_auth, _get_firecrest_url

if TYPE_CHECKING:
    from firecrest.v2 import AsyncFirecrest

logger = logging.getLogger("fcw.fuse")


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
    ):
        super().__init__()
        self.client = client
        self.system = system
        self.remote_root = remote_root.rstrip("/")
        self.account = account
        self.cache_ttl = cache_ttl
        self.read_only = read_only

        # Inode management
        self.inode_map: dict[int, str] = {pyfuse3.ROOT_INODE: self.remote_root}
        self.path_to_inode: dict[str, int] = {self.remote_root: pyfuse3.ROOT_INODE}
        self.next_inode = pyfuse3.ROOT_INODE + 1

        # Caching
        self.attr_cache: TTLCache = TTLCache(maxsize=10000, ttl=cache_ttl)
        self.dir_cache: TTLCache = TTLCache(maxsize=1000, ttl=cache_ttl)
        self.read_cache_dir = tempfile.mkdtemp(prefix="fcw_cache_")

        # File handle management (for writes)
        self.open_files: dict[int, dict] = {}
        self.next_fh = 1

        logger.info(f"Initialized FirecrestFS: {system}:{remote_root}")
        logger.info(f"Cache TTL: {cache_ttl}s, Read-only: {read_only}")

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
        entry.st_uid = int(get_val(fc_stat, "uid", os.getuid()))
        entry.st_gid = int(get_val(fc_stat, "gid", os.getgid()))
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
        """Return filesystem statistics."""
        # TODO: Query remote via job and cache result
        # For now, return reasonable defaults
        stats = pyfuse3.StatvfsData()
        stats.f_bsize = 65536
        stats.f_frsize = 65536
        stats.f_blocks = 10000000
        stats.f_bfree = 5000000
        stats.f_bavail = 5000000
        stats.f_files = 1000000
        stats.f_ffree = 500000
        stats.f_namemax = 255
        return stats

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
        except FirecrestException as e:
            raise pyfuse3.FUSEError(self._map_error(e))
        except Exception as e:
            logger.error(f"lookup error for {path}: {e}")
            raise pyfuse3.FUSEError(errno.EIO)

    async def getattr(self, inode: int, ctx=None):
        """Get file attributes."""
        path = self._inode_to_path(inode)
        logger.debug(f"getattr: {path}")

        # Check if file is open with dirty buffer
        for fh, data in self.open_files.items():
            if data["path"] == path and data["dirty"]:
                entry = pyfuse3.EntryAttributes()
                entry.st_ino = inode
                entry.st_mode = stat.S_IFREG | 0o644
                entry.st_size = os.fstat(data["buffer"].fileno()).st_size
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

    async def opendir(self, inode: int, ctx):
        """Open a directory for reading."""
        return inode

    async def readdir(self, inode: int, start_id: int, token):
        """Read directory contents."""
        path = self._inode_to_path(inode)
        logger.debug(f"readdir: {path}")

        try:
            files = await self._fc_list_files(path)
        except Exception as e:
            logger.error(f"readdir error for {path}: {e}")
            raise pyfuse3.FUSEError(errno.EIO)

        entries = []

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
        for f in files:
            name = f.get("name") if isinstance(f, dict) else getattr(f, "name", None)
            ftype = f.get("type") if isinstance(f, dict) else getattr(f, "type", None)

            if not name or name in (".", ".."):
                continue

            mode = stat.S_IFREG | 0o644
            if ftype == "d":
                mode = stat.S_IFDIR | 0o755
            elif ftype == "l":
                mode = stat.S_IFLNK | 0o777

            child_path = os.path.join(path, name)
            child_inode = self._get_inode(child_path)

            attr = pyfuse3.EntryAttributes()
            attr.st_ino = child_inode
            attr.st_mode = mode

            entries.append((name.encode("utf-8"), attr))

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
        fh = self.next_fh
        self.next_fh += 1

        is_truncated = bool(flags & os.O_TRUNC)

        self.open_files[fh] = {
            "path": path,
            "dirty": is_truncated,
            "buffer": tempfile.NamedTemporaryFile(delete=False, dir=self.read_cache_dir),
            "cached": False,
        }

        return pyfuse3.FileInfo(fh=fh, direct_io=True)

    async def read(self, fh: int, offset: int, length: int):
        """Read from a file."""
        if fh not in self.open_files:
            raise pyfuse3.FUSEError(errno.EBADF)

        data = self.open_files[fh]

        # If we have a dirty buffer, read from it
        if data["dirty"] or data["cached"]:
            f = data["buffer"]
            f.seek(offset)
            return f.read(length)

        # Download entire file to cache on first read
        logger.debug(f"Downloading {data['path']} to cache")

        try:
            await self.client.download(
                system_name=self.system,
                source_path=data["path"],
                target_path=data["buffer"].name,
                account=self.account,
                blocking=True,
            )
            data["cached"] = True

            # Reopen buffer for reading
            data["buffer"].close()
            data["buffer"] = open(data["buffer"].name, "rb+")

            f = data["buffer"]
            f.seek(offset)
            return f.read(length)

        except Exception as e:
            logger.error(f"Read error: {e}")
            raise pyfuse3.FUSEError(self._map_error(e))

    async def write(self, fh: int, offset: int, buf: bytes):
        """Write to a file."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        if fh not in self.open_files:
            raise pyfuse3.FUSEError(errno.EBADF)

        data = self.open_files[fh]

        # Download file first if not already dirty/cached
        if not data["dirty"] and not data["cached"]:
            logger.debug(f"Downloading {data['path']} before write")
            try:
                data["buffer"].close()

                await self.client.download(
                    system_name=self.system,
                    source_path=data["path"],
                    target_path=data["buffer"].name,
                    account=self.account,
                    blocking=True,
                )

                data["buffer"] = open(data["buffer"].name, "rb+")
            except Exception as e:
                logger.error(f"Download for write failed: {e}")
                raise pyfuse3.FUSEError(self._map_error(e))

        f = data["buffer"]
        f.seek(offset)
        f.write(buf)
        data["dirty"] = True

        return len(buf)

    async def flush(self, fh: int):
        """Flush file to storage."""
        if fh not in self.open_files:
            return

        data = self.open_files[fh]
        if not data["dirty"]:
            return

        logger.info(f"Uploading {data['path']}")

        try:
            f = data["buffer"]
            f.flush()
            os.fsync(f.fileno())

            path = data["path"]
            target_dir = os.path.dirname(path)
            filename = os.path.basename(path)

            data["dirty"] = False

            await self.client.upload(
                system_name=self.system,
                local_file=data["buffer"].name,
                directory=target_dir,
                filename=filename,
                account=self.account,
                blocking=True,
            )

            # Invalidate cache
            if path in self.attr_cache:
                del self.attr_cache[path]

        except Exception as e:
            logger.error(f"Upload failed: {e}")
            data["dirty"] = True
            raise pyfuse3.FUSEError(self._map_error(e))

    async def release(self, fh: int):
        """Release file handle."""
        if fh in self.open_files:
            data = self.open_files[fh]
            try:
                data["buffer"].close()
                if os.path.exists(data["buffer"].name):
                    os.unlink(data["buffer"].name)
            except Exception as e:
                logger.warning(f"Error cleaning up fh {fh}: {e}")
            del self.open_files[fh]

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

            # Invalidate parent dir cache
            parent_path = self._inode_to_path(parent_inode)
            if parent_path in self.dir_cache:
                del self.dir_cache[parent_path]

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
            if path in self.attr_cache:
                del self.attr_cache[path]
            parent_path = self._inode_to_path(parent_inode)
            if parent_path in self.dir_cache:
                del self.dir_cache[parent_path]

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
            if path in self.attr_cache:
                del self.attr_cache[path]
            if path in self.dir_cache:
                del self.dir_cache[path]
            parent_path = self._inode_to_path(parent_inode)
            if parent_path in self.dir_cache:
                del self.dir_cache[parent_path]

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

            # Update inode mappings for renamed path and all children
            old_prefix = old_path + "/"
            for path in list(self.path_to_inode.keys()):
                if path == old_path or path.startswith(old_prefix):
                    inode = self.path_to_inode.pop(path)
                    updated_path = new_path + path[len(old_path):]
                    self.path_to_inode[updated_path] = inode
                    self.inode_map[inode] = updated_path

            # Invalidate caches
            for cache in [self.attr_cache, self.dir_cache]:
                for key in list(cache.keys()):
                    if key == old_path or key.startswith(old_prefix):
                        del cache[key]

        except Exception as e:
            logger.error(f"rename failed: {e}")
            raise pyfuse3.FUSEError(self._map_error(e))

    async def create(self, parent_inode: int, name: bytes, mode: int, flags: int, ctx):
        """Create and open a new file."""
        if self.read_only:
            raise pyfuse3.FUSEError(errno.EROFS)

        path = os.path.join(self._inode_to_path(parent_inode), name.decode("utf-8"))

        fh = self.next_fh
        self.next_fh += 1
        inode = self._get_inode(path)

        self.open_files[fh] = {
            "path": path,
            "dirty": True,
            "buffer": tempfile.NamedTemporaryFile(delete=False, dir=self.read_cache_dir),
            "cached": False,
        }

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

        # Invalidate parent dir cache
        parent_path = self._inode_to_path(parent_inode)
        if parent_path in self.dir_cache:
            del self.dir_cache[parent_path]

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
    )

    fuse_options = set(pyfuse3.default_options)
    fuse_options.add("fsname=fcw")
    if allow_other:
        fuse_options.add("allow_other")
    if debug:
        fuse_options.add("debug")

    pyfuse3.asyncio.enable()
    pyfuse3.init(fs, mountpoint, fuse_options)

    try:
        asyncio.run(pyfuse3.main())
    except KeyboardInterrupt:
        pass
    finally:
        pyfuse3.close()
        asyncio.run(client.close_session())
        shutil.rmtree(fs.read_cache_dir, ignore_errors=True)

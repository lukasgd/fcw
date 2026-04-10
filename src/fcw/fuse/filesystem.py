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
from datetime import datetime, timezone
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
        self.negative_cache: TTLCache = TTLCache(maxsize=10000, ttl=cache_ttl)
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

    async def setattr(self, inode: int, attr, fields, fh, ctx):
        """Set file attributes (truncate, chmod, chown)."""
        path = self._inode_to_path(inode)

        # Check if file only exists locally (created but not yet uploaded)
        local_only = any(
            d["path"] == path and d["dirty"] and not d["cached"]
            for d in self.open_files.values()
        )

        # Handle truncate
        if fields.update_size:
            target_fh = fh
            if target_fh is None:
                for open_fh, data in self.open_files.items():
                    if data["path"] == path:
                        target_fh = open_fh
                        break

            if target_fh is not None and target_fh in self.open_files:
                data = self.open_files[target_fh]
                data["buffer"].truncate(attr.st_size)
                data["dirty"] = True
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

            # Re-stat to populate cache with new mtime so editors
            # don't see a stale mtime on their post-save check
            if path in self.attr_cache:
                del self.attr_cache[path]
            await self._fc_stat(path)

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

        # Invalidate negative cache since file is being created
        if path in self.negative_cache:
            del self.negative_cache[path]

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

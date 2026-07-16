"""Tests for sync markers, file collection, and match patterns."""

import logging
import os
import tarfile
import time
from datetime import datetime

import pytest

import typer

import fcw.commands.data as data
from fcw.commands.data import (
    _build_emacs_match_pattern,
    _collect_local_files_since,
    _extract_dir_archive,
    _get_sync_marker_path,
    _parse_size,
    _partition_by_size,
    _read_last_sync_timestamp,
    _remote_is_dir,
    _upload_files_chunked,
    _upload_incremental,
    _write_last_sync_timestamp,
)


class TestSyncMarkers:
    def test_write_and_read(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _write_last_sync_timestamp("data/raw", "push", ts=1234567890.0)
        result = _read_last_sync_timestamp("data/raw", "push")
        assert result == 1234567890.0

    def test_missing_marker_returns_zero(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        result = _read_last_sync_timestamp("nonexistent", "push")
        assert result == 0.0

    def test_marker_path_safe_name(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        path = _get_sync_marker_path("data/raw", "push")
        assert "data_raw.push.marker" in str(path)

    def test_auto_timestamp(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        before = time.time()
        _write_last_sync_timestamp("test", "pull")
        after = time.time()
        ts = _read_last_sync_timestamp("test", "pull")
        assert before <= ts <= after


class TestCollectLocalFiles:
    def test_collects_modified_files(self, tmp_path):
        cutoff = time.time() - 1
        (tmp_path / "new.txt").write_text("new")
        files = _collect_local_files_since(str(tmp_path), cutoff)
        rel_paths = [rel for _, rel in files]
        assert "new.txt" in rel_paths

    def test_skips_old_files(self, tmp_path):
        (tmp_path / "old.txt").write_text("old")
        cutoff = time.time() + 1
        files = _collect_local_files_since(str(tmp_path), cutoff)
        assert len(files) == 0

    def test_skips_fcw_hidden_files(self, tmp_path):
        cutoff = time.time() - 1
        (tmp_path / ".fcw_marker").write_text("marker")
        (tmp_path / "real.txt").write_text("real")
        files = _collect_local_files_since(str(tmp_path), cutoff)
        rel_paths = [rel for _, rel in files]
        assert "real.txt" in rel_paths
        assert ".fcw_marker" not in rel_paths

    def test_recursive(self, tmp_path):
        cutoff = time.time() - 1
        subdir = tmp_path / "sub"
        subdir.mkdir()
        (subdir / "deep.txt").write_text("deep")
        files = _collect_local_files_since(str(tmp_path), cutoff)
        rel_paths = [rel for _, rel in files]
        assert os.path.join("sub", "deep.txt") in rel_paths

    def _setup_symlinked_subdir(self, tmp_path):
        """Create local_dir/ with a real file and a symlinked subdir → external dir."""
        external = tmp_path / "external"
        external.mkdir()
        (external / "linked.txt").write_text("payload")
        local_dir = tmp_path / "local"
        local_dir.mkdir()
        (local_dir / "real.txt").write_text("real")
        os.symlink(external, local_dir / "link")
        return str(local_dir)

    def test_skips_symlinked_subdir_by_default(self, tmp_path):
        local_dir = self._setup_symlinked_subdir(tmp_path)
        files = _collect_local_files_since(local_dir, 0)
        rel_paths = [rel for _, rel in files]
        assert "real.txt" in rel_paths
        assert os.path.join("link", "linked.txt") not in rel_paths

    def test_follows_symlinked_subdir_when_enabled(self, tmp_path):
        local_dir = self._setup_symlinked_subdir(tmp_path)
        files = _collect_local_files_since(local_dir, 0, follow_symlinks=True)
        rel_paths = [rel for _, rel in files]
        assert os.path.join("link", "linked.txt") in rel_paths


class TestBuildEmacMatchPattern:
    def test_single_file(self):
        pattern = _build_emacs_match_pattern(["file.txt"], "/remote/data")
        assert r"file\.txt" in pattern
        assert pattern.startswith("^")
        assert pattern.endswith("$")

    def test_multiple_files(self):
        pattern = _build_emacs_match_pattern(["a.txt", "b.txt"], "/remote/data")
        assert r"a\.txt" in pattern
        assert r"b\.txt" in pattern
        assert r"\|" in pattern

    def test_empty_list(self):
        pattern = _build_emacs_match_pattern([], "/remote/data")
        assert pattern == "^$"

    def test_uses_source_basename(self):
        pattern = _build_emacs_match_pattern(["file.txt"], "/remote/mydir")
        assert "mydir" in pattern


class TestExtractDirArchive:
    def _make_archive(self, tmp_path, members):
        """Build a gzip tar rooted at the dir basename (e.g. outputs/...)."""
        src = tmp_path / "src"
        archive = tmp_path / "archive.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            for name, content in members.items():
                f = src / name
                f.parent.mkdir(parents=True, exist_ok=True)
                f.write_text(content)
                tar.add(f, arcname=name)
        return str(archive)

    def test_extracts_contents_without_double_nesting(self, tmp_path):
        archive = self._make_archive(tmp_path, {
            "outputs/results.txt": "data",
            "outputs/sub/x.txt": "nested",
        })
        local_dir = tmp_path / "outputs"
        _extract_dir_archive(archive, str(local_dir))

        assert (local_dir / "results.txt").read_text() == "data"
        assert (local_dir / "sub" / "x.txt").read_text() == "nested"
        # The bug: archive rooted at "outputs/" must not nest under local_dir again.
        assert not (local_dir / "outputs").exists()


class _StubAsyncClient:
    """Records the remote operations an upload performs, in order."""

    def __init__(self):
        self.calls = []

    async def mkdir(self, **kwargs):
        self.calls.append("mkdir")

    async def upload(self, **kwargs):
        self.calls.append("upload")

    async def extract(self, **kwargs):
        self.calls.append("extract")

    async def rm(self, **kwargs):
        self.calls.append("rm")


class TestDataLogging:
    """fcw logs its orchestration/local context (counts, local tar/extract); the
    remote ops themselves are left to pyfirecrest's own `firecrest` logger. These
    lock the fcw-side contract without shadowing pyfirecrest."""

    def test_collect_local_files_logs_count(self, tmp_path, caplog):
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        with caplog.at_level(logging.INFO, logger="fcw.data"):
            _collect_local_files_since(str(tmp_path), 0)
        assert any("found 2 local file(s)" in r.message for r in caplog.records)

    async def test_upload_incremental_logs_local_context(self, tmp_path, monkeypatch, caplog):
        monkeypatch.chdir(tmp_path)
        local_dir = tmp_path / "data"
        local_dir.mkdir()
        (local_dir / "f.txt").write_text("payload")

        client = _StubAsyncClient()
        with caplog.at_level(logging.INFO, logger="fcw.data"):
            count = await _upload_incremental(
                client, "sys", "acct", str(local_dir), "/remote/data"
            )

        assert count == 1
        assert client.calls == ["mkdir", "upload", "extract", "rm"]

        msgs = [r.message for r in caplog.records if r.levelno == logging.INFO]
        joined = "\n".join(msgs)
        # The helper logs mechanics only: the decision (count) and local-only steps.
        # It does NOT shadow the pyfirecrest upload/extract/rm calls (firecrest's to
        # log), nor the "uploading X -> Y" intent line (a command-level concern).
        assert "found 1 local file(s)" in joined
        assert "tarring 1 file(s) ..." in joined
        assert any("archive built:" in m for m in msgs)
        assert "uploading archive" not in joined
        assert "extracting archive on remote" not in joined
        assert not any(m.startswith("uploading ") and "->" in m for m in msgs)

        def idx(needle):
            return next(i for i, m in enumerate(msgs) if needle in m)

        assert idx("found 1 local file(s)") < idx("tarring")


class TestParseSize:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("2GB", 2_000_000_000),
            ("512MB", 512_000_000),
            ("1KiB", 1024),
            ("1024", 1024),
            ("1.5GB", 1_500_000_000),
            ("2 gb", 2_000_000_000),
        ],
    )
    def test_valid(self, text, expected):
        assert _parse_size(text) == expected

    def test_invalid_unit(self):
        with pytest.raises(typer.BadParameter):
            _parse_size("5XB")

    def test_invalid_format(self):
        with pytest.raises(typer.BadParameter):
            _parse_size("abc")


class TestPartitionBySize:
    def test_packs_small_items(self):
        items = [("a", 30), ("b", 30), ("c", 30)]
        assert list(_partition_by_size(items, 100)) == [(["a", "b", "c"], False)]

    def test_flushes_when_exceeding(self):
        items = [("a", 60), ("b", 60)]
        assert list(_partition_by_size(items, 100)) == [(["a"], False), (["b"], False)]

    def test_cumulative_equal_stays_together(self):
        items = [("a", 50), ("b", 50)]
        assert list(_partition_by_size(items, 100)) == [(["a", "b"], False)]

    def test_oversized_isolated_and_order_preserved(self):
        items = [("a", 30), ("big", 200), ("b", 30)]
        assert list(_partition_by_size(items, 100)) == [
            (["a"], False),
            (["big"], True),
            (["b"], False),
        ]

    def test_boundary_equal_is_oversized(self):
        assert list(_partition_by_size([("a", 100)], 100)) == [(["a"], True)]


class _RecordingClient:
    """Records each remote op with its kwargs, in order."""

    def __init__(self):
        self.calls = []

    async def mkdir(self, **kw):
        self.calls.append(("mkdir", kw))

    async def upload(self, **kw):
        self.calls.append(("upload", kw))

    async def extract(self, **kw):
        self.calls.append(("extract", kw))

    async def rm(self, **kw):
        self.calls.append(("rm", kw))


class TestChunkedUpload:
    async def test_large_file_uploaded_directly(self, tmp_path):
        d = tmp_path / "data"
        d.mkdir()
        (d / "a.txt").write_text("x" * 10)
        (d / "b.txt").write_text("y" * 10)
        (d / "big.bin").write_bytes(b"z" * 200)
        files = [
            (str(d / "a.txt"), "a.txt"),
            (str(d / "b.txt"), "b.txt"),
            (str(d / "big.bin"), "big.bin"),
        ]

        client = _RecordingClient()
        await _upload_files_chunked(client, "sys", "acct", files, "/remote/data", 100, False)

        ops = [op for op, _ in client.calls]
        uploaded = [kw["filename"] for op, kw in client.calls if op == "upload"]
        # The two small files go in one tar batch (one extract); the big file is
        # streamed directly with no tar wrapper (no extra extract).
        assert ops.count("extract") == 1
        assert sorted(uploaded) == sorted(["big.bin", ".fcw_upload_chunk.tar.gz"])


class _LsClient:
    """Minimal async client exposing list_files() returning fixed parent entries."""

    def __init__(self, entries):
        self._entries = entries
        self.listed = []

    async def list_files(self, *, system_name, path, **kw):
        self.listed.append(path)
        return self._entries


class TestRemoteIsDir:
    """`data download` file-vs-directory detection, from the `ls` type field.

    Regression for the bug where a plain file was misclassified as a directory and
    downloaded as a local dir. Detection uses the `ls` ``type`` field (not stat
    mode, which omits type bits — eth-cscs/firecrest#171).
    """

    async def test_regular_file_is_not_dir(self):
        client = _LsClient([
            {"name": "f.out", "type": "-"},
            {"name": "sub", "type": "d"},
        ])
        assert await _remote_is_dir(client, "sys", "/remote/f.out") is False
        assert client.listed == ["/remote"]  # classified via the parent listing

    async def test_directory_is_dir(self):
        client = _LsClient([
            {"name": "d", "type": "d"},
            {"name": "f.out", "type": "-"},
        ])
        assert await _remote_is_dir(client, "sys", "/remote/d") is True

    async def test_missing_path_raises(self):
        client = _LsClient([{"name": "other", "type": "-"}])
        with pytest.raises(FileNotFoundError):
            await _remote_is_dir(client, "sys", "/remote/nope")


class TestDownloadIncrementalMarker:
    """Incremental download must skip unchanged files (regression: the pull marker
    used local `time.time()` vs remote mtimes — two clocks — so every unchanged file
    re-downloaded). The marker is now the newest REMOTE mtime, so a second run with
    unchanged entries transfers nothing, regardless of the API's timezone semantics.
    """

    async def test_marker_uses_max_remote_mtime_and_skips_second_run(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        entries = [
            {"name": "a.txt", "type": "-", "size": 10, "lastModified": "2024-01-01T00:00:00"},
            {"name": "b.txt", "type": "-", "size": 20, "lastModified": "2024-01-02T00:00:00"},
        ]
        client = _LsClient(entries)

        transfers = []

        async def _fake_chunked(cl, system, account, files, remote_dir, local_dir, chunk_size):
            transfers.append(list(files))

        monkeypatch.setattr(data, "_download_files_chunked", _fake_chunked)

        # First run: no marker yet -> both files pulled, marker = newest remote mtime.
        n1 = await data._download_incremental(client, "sys", "acct", "/remote/outputs", "outputs")
        assert n1 == 2
        assert len(transfers) == 1 and len(transfers[0]) == 2

        expected = datetime.fromisoformat("2024-01-02T00:00:00").timestamp()
        marker = _read_last_sync_timestamp(os.path.abspath("outputs"), "pull")
        assert marker == pytest.approx(expected)

        # Second run, same entries: nothing changed -> no files pulled, no transfer.
        n2 = await data._download_incremental(client, "sys", "acct", "/remote/outputs", "outputs")
        assert n2 == 0
        assert len(transfers) == 1

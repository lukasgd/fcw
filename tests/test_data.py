"""Tests for sync markers, file collection, and match patterns."""

import os
import tarfile
import time

import pytest

from fcw.commands.data import (
    _build_emacs_match_pattern,
    _collect_local_files_since,
    _extract_dir_archive,
    _get_sync_marker_path,
    _read_last_sync_timestamp,
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

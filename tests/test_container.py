"""Tests for container helper functions."""

import subprocess
from unittest.mock import patch

import pytest

from pathlib import Path

from fcw.commands.container import (
    _create_rebuilt_toml,
    _derive_container_name,
    _derive_rebuilt_toml_path,
    _detect_container_runtime,
    _find_container_config,
    _generate_load_and_resolve_block,
    _isolated_staging_dir,
    _merge_build_args,
    _parse_patch_arg,
    _parse_patch_mounts,
    _patches_index_path,
    _podman_setup_block,
    _read_sidecar,
    _record_patch_in_index,
    _resolve_remote_tar,
    _resync_container_patches,
    _scan_staging_dirs,
    _sidecar_path,
    _staging_cleanup_block,
    _stage_to_build_arg_name,
    _update_toml_bind_mount,
    _write_sidecar,
)


class TestDetectContainerRuntime:
    def test_podman_found(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess([], 0)
            assert _detect_container_runtime() == "podman"

    def test_docker_fallback(self):
        def side_effect(cmd, **kwargs):
            if cmd[0] == "podman":
                raise FileNotFoundError
            return subprocess.CompletedProcess([], 0)

        with patch("subprocess.run", side_effect=side_effect):
            assert _detect_container_runtime() == "docker"

    def test_none_found(self):
        with patch("subprocess.run", side_effect=FileNotFoundError):
            with pytest.raises(RuntimeError, match="No container runtime found"):
                _detect_container_runtime()


class TestFindContainerConfig:
    def test_exact_match(self, sample_config):
        result = _find_container_config(sample_config, "my-fcw-app:latest")
        assert result is not None
        assert result.tag == "my-fcw-app:latest"

    def test_prefix_match(self, sample_config):
        """An intermediate stage tag should match by image name prefix."""
        result = _find_container_config(sample_config, "my-fcw-app:v2-download")
        assert result is not None
        assert result.tag == "my-fcw-app:latest"  # matched the 'app' container

    def test_prefix_match_no_tag(self, sample_config):
        """A bare image name (no colon) should match."""
        result = _find_container_config(sample_config, "my-fcw-app")
        assert result is not None
        assert result.tag == "my-fcw-app:latest"

    def test_no_match(self, sample_config):
        result = _find_container_config(sample_config, "unknown:v1")
        assert result is None

    def test_exact_takes_precedence(self, sample_config):
        """Exact tag match should win over prefix match."""
        result = _find_container_config(sample_config, "fcw-aux:latest")
        assert result is not None
        assert result.tag == "fcw-aux:latest"

    def test_different_name_no_match(self, sample_config):
        """A completely different image name should not match."""
        result = _find_container_config(sample_config, "otherapp:latest")
        assert result is None


class TestResolveRemoteTar:
    def test_tag_to_tar_with_config(self, sample_config):
        """Tag matching a config entry should use its remote_path."""
        result = _resolve_remote_tar("my-fcw-app:latest", sample_config)
        assert result == "/scratch/user/test-project/ce-images/my-fcw-app+latest.tar"

    def test_prefix_match_uses_config_path(self, sample_config):
        """Intermediate stage tag should resolve via prefix-matched config."""
        result = _resolve_remote_tar("my-fcw-app:v2-download", sample_config)
        assert result == "/scratch/user/test-project/ce-images/my-fcw-app+v2-download.tar"

    def test_unknown_tag_falls_back(self, sample_config):
        """Unknown tag falls back to ce-images/ directory."""
        result = _resolve_remote_tar("unknown:v1", sample_config)
        assert result == "/scratch/user/test-project/ce-images/unknown+v1.tar"

    def test_tag_with_registry(self, sample_config):
        result = _resolve_remote_tar("registry.io/my-fcw-app:v1", sample_config)
        # Contains "/" so treated as path, not tag
        assert result == "/scratch/user/test-project/registry.io/my-fcw-app:v1"

    def test_tar_path(self, sample_config):
        result = _resolve_remote_tar("ce-images/custom.tar", sample_config)
        assert result == "/scratch/user/test-project/ce-images/custom.tar"

    def test_absolute_path(self, sample_config):
        result = _resolve_remote_tar("/abs/path.tar", sample_config)
        assert result == "/abs/path.tar"

    def test_simple_name(self, sample_config):
        """Bare name without colon should still look up config."""
        result = _resolve_remote_tar("my-fcw-app", sample_config)
        assert result == "/scratch/user/test-project/ce-images/my-fcw-app.tar"


class TestPodmanSetupBlock:
    def test_returns_string(self):
        block = _podman_setup_block()
        assert isinstance(block, str)

    def test_contains_systemd_wait(self):
        block = _podman_setup_block()
        assert "pgrep" in block
        assert "systemd" in block

    def test_contains_podman_reset(self):
        block = _podman_setup_block()
        assert "podman system reset -f || true" in block

    def test_contains_storage_conf(self):
        block = _podman_setup_block()
        assert "storage.conf" in block
        assert 'driver = "overlay"' in block

    def test_contains_xdg_setup(self):
        block = _podman_setup_block()
        assert "XDG_RUNTIME_DIR" in block
        assert "mktemp" in block
        assert "chmod 700" in block

    def test_contains_home_export(self):
        block = _podman_setup_block()
        assert 'export HOME=' in block

    def test_contains_shm_cleanup(self):
        block = _podman_setup_block()
        assert "/dev/shm/$USER" in block


class TestParsePatchMounts:
    def test_single_patch_mount(self):
        toml = '''
image = "/scratch/app.sqsh"
mounts = [
    "/scratch/project/.patches/code:/workspace/BrainBERT",
    "${SCRATCH}",
]
'''
        result = _parse_patch_mounts(toml)
        assert result == [("/scratch/project/.patches/code", "/workspace/BrainBERT")]

    def test_multiple_patch_mounts(self):
        toml = '''
mounts = [
    "/scratch/.patches/code:/workspace/code",
    "/scratch/.patches/data:/workspace/data",
]
'''
        result = _parse_patch_mounts(toml)
        assert len(result) == 2
        assert result[0] == ("/scratch/.patches/code", "/workspace/code")
        assert result[1] == ("/scratch/.patches/data", "/workspace/data")

    def test_no_patch_mounts(self):
        toml = '''
mounts = [
    "${SCRATCH}/data:/mnt/data",
]
'''
        result = _parse_patch_mounts(toml)
        assert result == []

    def test_no_mounts_at_all(self):
        toml = 'image = "foo.sqsh"\n'
        result = _parse_patch_mounts(toml)
        assert result == []

    def test_mixed_mounts(self):
        toml = '''
mounts = [
    "/scratch/.patches/code:/workspace/code",
    "/data/shared:/mnt/shared",
]
'''
        result = _parse_patch_mounts(toml)
        assert len(result) == 1
        assert result[0] == ("/scratch/.patches/code", "/workspace/code")


class TestCreateRebuiltToml:
    def test_removes_patch_mounts(self, tmp_path):
        original = tmp_path / "container.toml"
        original.write_text('''\
image = "/scratch/app.sqsh"
mounts = [
    "/scratch/.patches/code:/workspace/code",
    "${SCRATCH}/data:/mnt/data",
]
writable = true
''')
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new))
        content = new.read_text()
        assert ".patches" not in content
        assert "${SCRATCH}/data:/mnt/data" in content
        assert "writable = true" in content

    def test_preserves_non_patch_content(self, tmp_path):
        original = tmp_path / "container.toml"
        original.write_text('''\
image = "foo.sqsh"
mounts = [
    "${SCRATCH}/data:/data",
    "/scratch/.patches/code:/workspace",
]
workdir = "/workspace"
writable = true
''')
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new))
        content = new.read_text()
        assert 'workdir = "/workspace"' in content
        assert "${SCRATCH}/data:/data" in content
        assert ".patches" not in content

    def test_all_mounts_are_patches(self, tmp_path):
        original = tmp_path / "container.toml"
        original.write_text('''\
image = "foo.sqsh"
mounts = [
    "/scratch/.patches/code:/workspace/code",
    "/scratch/.patches/data:/workspace/data",
]
''')
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new))
        content = new.read_text()
        assert ".patches" not in content
        assert "mounts = []" in content

    def test_original_unchanged(self, tmp_path):
        original_content = '''\
image = "foo.sqsh"
mounts = [
    "/scratch/.patches/code:/workspace",
]
'''
        original = tmp_path / "container.toml"
        original.write_text(original_content)
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new))
        assert original.read_text() == original_content

    def test_rewrites_image_path(self, tmp_path):
        original = tmp_path / "container.toml"
        original.write_text('''\
image = "/scratch/old.sqsh"
mounts = [
    "/scratch/.patches/code:/workspace",
]
''')
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new), "/scratch/new.sqsh")
        content = new.read_text()
        assert 'image = "/scratch/new.sqsh"' in content
        assert "old.sqsh" not in content
        assert ".patches" not in content

    def test_rewrites_empty_image(self, tmp_path):
        original = tmp_path / "container.toml"
        original.write_text('image = ""\n')
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new), "/scratch/new.sqsh")
        assert new.read_text() == 'image = "/scratch/new.sqsh"\n'

    def test_no_image_line_no_insert(self, tmp_path):
        original = tmp_path / "container.toml"
        original.write_text('writable = true\n')
        new = tmp_path / "container-v2.toml"
        _create_rebuilt_toml(str(original), str(new), "/scratch/new.sqsh")
        assert 'image' not in new.read_text()


class TestPatchesIndex:
    def test_record_and_read_roundtrip(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        _record_patch_in_index("app", "code", str(tmp_path / "code"))
        _record_patch_in_index("app", "configs", str(tmp_path / "configs"))
        idx_path = _patches_index_path("app")
        assert idx_path.exists()
        assert idx_path.parent == Path(".fcw/patches")
        import json
        idx = json.loads(idx_path.read_text())
        assert idx["code"] == str(tmp_path / "code")
        assert idx["configs"] == str(tmp_path / "configs")

    def test_missing_index_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        from fcw.commands.container import _read_patches_index
        assert _read_patches_index("nope") == {}


class TestResyncContainerPatches:
    def _make_config(self, tmp_path, monkeypatch, *, toml_body: str):
        """Write a minimal fcw.yaml + TOML, load and cd into tmp_path."""
        import textwrap
        from fcw.core.config import load_config
        (tmp_path / "env").mkdir()
        (tmp_path / "env" / "container.toml").write_text(toml_body)
        (tmp_path / "fcw.yaml").write_text(textwrap.dedent("""\
            project: p
            workdir:
              remote: /scratch/p
              local: .
            containers:
              app:
                file: ./Dockerfile
                tag: app:latest
                remote_path: ce-images/
                toml: ./env/container.toml
            """))
        monkeypatch.chdir(tmp_path)
        return load_config(str(tmp_path / "fcw.yaml"))

    def test_noop_without_toml(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "fcw.commands.data._upload_incremental",
            lambda *a, **kw: calls.append(a) or 0,
        )
        cfg = self._make_config(tmp_path, monkeypatch, toml_body='image = ""\n')
        # Drop the toml reference on the container
        cfg.containers["app"].toml = None
        _resync_container_patches(cfg, "app", "sys", "acct")
        assert calls == []

    def test_noop_without_patch_mounts(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "fcw.commands.data._upload_incremental",
            lambda *a, **kw: calls.append(a) or 0,
        )
        cfg = self._make_config(
            tmp_path, monkeypatch, toml_body='image = ""\nmounts = []\n',
        )
        _resync_container_patches(cfg, "app", "sys", "acct")
        assert calls == []

    def test_noop_when_index_missing(self, tmp_path, monkeypatch):
        calls = []
        monkeypatch.setattr(
            "fcw.commands.data._upload_incremental",
            lambda *a, **kw: calls.append(a) or 0,
        )
        toml_body = (
            'image = ""\n'
            'mounts = [\n'
            '    "/scratch/p/.patches/code:/workspace/code",\n'
            ']\n'
        )
        cfg = self._make_config(tmp_path, monkeypatch, toml_body=toml_body)
        _resync_container_patches(cfg, "app", "sys", "acct")
        assert calls == []

    def test_happy_path_calls_upload_incremental(self, tmp_path, monkeypatch):
        dump = tmp_path / "code"
        dump.mkdir()
        (dump / "a.py").write_text("print('a')\n")
        toml_body = (
            'image = ""\n'
            'mounts = [\n'
            '    "/scratch/p/.patches/code:/workspace/code",\n'
            ']\n'
        )
        cfg = self._make_config(tmp_path, monkeypatch, toml_body=toml_body)
        _record_patch_in_index("app", "code", str(dump))

        calls = []

        async def fake_upload_incremental(client, system, account, local, remote):
            calls.append(("up_inc", local, remote))
            return 1

        monkeypatch.setattr(
            "fcw.commands.data._upload_incremental", fake_upload_incremental
        )

        uploads = []

        class FakeClient:
            async def upload(self, **kw):
                uploads.append(kw)

        monkeypatch.setattr(
            "fcw.commands.container.get_async_client", lambda: FakeClient()
        )

        _resync_container_patches(cfg, "app", "sys", "acct")

        assert len(calls) == 1
        assert calls[0][1] == str(dump)
        assert calls[0][2] == "/scratch/p/.patches/code"
        # No sidecar on disk, so no upload call expected
        assert uploads == []

    def test_uploads_sidecar_when_present(self, tmp_path, monkeypatch):
        dump = tmp_path / "code"
        dump.mkdir()
        (dump / "a.py").write_text("x\n")
        sidecar = tmp_path / "code.meta.json"
        sidecar.write_text('{"stage": "download", "container_path": "/opt"}')
        toml_body = (
            'image = ""\n'
            'mounts = [\n'
            '    "/scratch/p/.patches/code:/opt",\n'
            ']\n'
        )
        cfg = self._make_config(tmp_path, monkeypatch, toml_body=toml_body)
        _record_patch_in_index("app", "code", str(dump))

        async def fake_upload_incremental(client, system, account, local, remote):
            return 0

        monkeypatch.setattr(
            "fcw.commands.data._upload_incremental", fake_upload_incremental
        )

        uploads = []

        class FakeClient:
            async def upload(self, **kw):
                uploads.append(kw)

        monkeypatch.setattr(
            "fcw.commands.container.get_async_client", lambda: FakeClient()
        )

        _resync_container_patches(cfg, "app", "sys", "acct")

        assert len(uploads) == 1
        assert uploads[0]["filename"] == "code.meta.json"
        assert uploads[0]["directory"] == "/scratch/p/.patches"


class TestScanStagingDirs:
    async def test_parses_timestamps_and_skips_unsuffixed(self):
        class FakeClient:
            async def list_files(self, **kw):
                return [
                    {"name": "app-20260101T120000"},
                    {"name": "app-20260414T000000"},
                    {"name": "no-timestamp-here"},
                ]

        out = await _scan_staging_dirs(FakeClient(), "sys", "/anywhere")
        names = [n for n, _ in out]
        assert "app-20260101T120000" in names
        assert "app-20260414T000000" in names
        assert "no-timestamp-here" in names
        by_name = dict(out)
        assert by_name["app-20260101T120000"] is not None
        assert by_name["no-timestamp-here"] is None

    async def test_listing_failure_returns_empty(self):
        class FakeClient:
            async def list_files(self, **kw):
                raise RuntimeError("boom")

        assert await _scan_staging_dirs(FakeClient(), "sys", "/x") == []


class TestDeriveContainerName:
    def test_simple_version(self):
        assert _derive_container_name("app", "my-app:v1", "my-app:v2") == "app-v2"

    def test_dotted_version(self):
        assert _derive_container_name("app", "my-app:v1", "my-app:24.04") == "app-24-04"

    def test_no_colon(self):
        assert _derive_container_name("app", "latest", "rebuilt") == "app-rebuilt"

    def test_complex_suffix(self):
        assert _derive_container_name("web", "img:v0", "img:v1.2-beta") == "web-v1-2-beta"

    def test_idempotent_chain_v1_to_v2_to_v3(self):
        """Repeated rebuilds should not stack suffixes."""
        assert _derive_container_name("app-v2", "my-app:v2", "my-app:v3") == "app-v3"

    def test_stem_not_stripped_when_no_match(self):
        """If original_name does not end with the parent tag suffix, keep it whole."""
        assert _derive_container_name("node-burn", "burn:v1", "burn:v2") == "node-burn-v2"


class TestIsolatedStagingDir:
    def test_shape(self, sample_config):
        result = _isolated_staging_dir(sample_config, "rebuild", "app-v2")
        # Under workdir.remote / .fcw/<base>/<key>-<ts>
        assert ".fcw/rebuild/app-v2-" in result
        # Timestamp suffix: YYYYMMDDTHHMMSS (15 chars)
        tail = result.rsplit("-", 1)[-1]
        assert len(tail) == 15 and "T" in tail

    def test_sanitizes_special_chars(self, sample_config):
        result = _isolated_staging_dir(sample_config, "deploy", "my/app:v2")
        assert "my-app-v2" in result
        assert ":" not in result.split("/")[-1]
        assert "/" not in result.rsplit("/.fcw/", 1)[-1].split("/")[-1].split("-20")[0]

    def test_empty_key_falls_back(self, sample_config):
        result = _isolated_staging_dir(sample_config, "deploy", "")
        assert "/.fcw/deploy/build-" in result


class TestStagingCleanupBlock:
    def test_contains_rm_and_trap(self):
        block = _staging_cleanup_block("/scratch/.fcw/rebuild/app-v2-20260414T103000")
        assert "rm -rf /scratch/.fcw/rebuild/app-v2-20260414T103000" in block
        assert "trap" in block
        assert "EXIT" in block

    def test_quotes_path_with_special_chars(self):
        """Paths with shell metacharacters must be shell-quoted."""
        block = _staging_cleanup_block("/tmp/a b/x")
        assert "'/tmp/a b/x'" in block

    def test_preserves_on_failure(self):
        """Cleanup must only run on successful exit, else debugging is impossible."""
        block = _staging_cleanup_block("/tmp/x")
        assert "$? -eq 0" in block


class TestDeriveRebuiltTomlPath:
    def test_simple_new_version(self):
        result = _derive_rebuilt_toml_path(
            Path("env/container.toml"), "app:v1", "app:v2"
        )
        assert result == Path("env/container-v2.toml")

    def test_chain_replaces_parent_suffix(self):
        result = _derive_rebuilt_toml_path(
            Path("env/container-v2.toml"), "app:v2", "app:v3"
        )
        assert result == Path("env/container-v3.toml")

    def test_preserves_non_standard_stem(self):
        result = _derive_rebuilt_toml_path(
            Path("env/node-burn.toml"), "app:v1", "app:v2"
        )
        assert result == Path("env/node-burn-v2.toml")

    def test_sanitizes_dotted_tag(self):
        result = _derive_rebuilt_toml_path(
            Path("env/container.toml"), "app:latest", "app:24.04"
        )
        assert result == Path("env/container-24-04.toml")


class TestMergeBuildArgs:
    def test_config_only(self):
        result = _merge_build_args({"BASE": "ubuntu:24.04"}, None)
        assert result == ["BASE=ubuntu:24.04"]

    def test_cli_only(self):
        result = _merge_build_args(None, ["BASE=ubuntu:24.04"])
        assert result == ["BASE=ubuntu:24.04"]

    def test_cli_overrides_config(self):
        result = _merge_build_args(
            {"BASE": "ubuntu:22.04", "EXTRA": "yes"},
            ["BASE=ubuntu:24.04"],
        )
        assert "BASE=ubuntu:24.04" in result
        assert "EXTRA=yes" in result
        assert len(result) == 2

    def test_both_none(self):
        result = _merge_build_args(None, None)
        assert result == []

    def test_empty_dicts(self):
        result = _merge_build_args({}, [])
        assert result == []


class TestStageToBuildArgName:
    def test_download(self):
        assert _stage_to_build_arg_name("download") == "DOWNLOAD_IMAGE"

    def test_runtime_download(self):
        assert _stage_to_build_arg_name("runtime-download") == "RUNTIME_DOWNLOAD_IMAGE"

    def test_simple_name(self):
        assert _stage_to_build_arg_name("base") == "BASE_IMAGE"


class TestSidecar:
    def test_path_is_sibling_with_suffix(self, tmp_path):
        dump = tmp_path / "code"
        dump.mkdir()
        assert _sidecar_path(str(dump)) == dump.with_name("code.meta.json")

    def test_write_then_read_roundtrip(self, tmp_path):
        dump = tmp_path / "code"
        dump.mkdir()
        _write_sidecar(
            str(dump),
            stage="runtime-download",
            container_path="/opt/BrainBERT",
            source_image="app:v1-runtime-download",
        )
        meta = _read_sidecar(str(dump))
        assert meta is not None
        assert meta["stage"] == "runtime-download"
        assert meta["container_path"] == "/opt/BrainBERT"
        assert meta["source_image"] == "app:v1-runtime-download"
        assert "extracted_at" in meta

    def test_read_missing_returns_none(self, tmp_path):
        assert _read_sidecar(str(tmp_path / "nothing")) is None

    def test_read_corrupt_returns_none(self, tmp_path):
        dump = tmp_path / "code"
        dump.mkdir()
        _sidecar_path(str(dump)).write_text("{not json")
        assert _read_sidecar(str(dump)) is None


class TestParsePatchArg:
    def test_no_colon_returns_none_target(self):
        assert _parse_patch_arg("./code") == ("./code", None)

    def test_with_target(self):
        assert _parse_patch_arg("./code:/opt/app") == ("./code", "/opt/app")

    def test_splits_on_first_colon(self):
        assert _parse_patch_arg("./code:/opt:sub") == ("./code", "/opt:sub")


class TestUpdateTomlBindMount:
    def test_appends_to_existing_mounts(self, tmp_path):
        toml = tmp_path / "c.toml"
        toml.write_text('image = "foo.sqsh"\nmounts = [\n    "${SCRATCH}:/data",\n]\n')
        _update_toml_bind_mount(toml, "/scratch/.patches/code:/workspace", "/workspace")
        content = toml.read_text()
        assert '"/scratch/.patches/code:/workspace"' in content
        assert '"${SCRATCH}:/data"' in content

    def test_replaces_existing_mount_for_same_container_path(self, tmp_path):
        toml = tmp_path / "c.toml"
        toml.write_text('mounts = [\n    "/old:/workspace",\n]\n')
        _update_toml_bind_mount(toml, "/new:/workspace", "/workspace")
        content = toml.read_text()
        assert '"/new:/workspace"' in content
        assert "/old" not in content

    def test_adds_mounts_array_when_absent(self, tmp_path):
        toml = tmp_path / "c.toml"
        toml.write_text('image = "foo.sqsh"\n')
        _update_toml_bind_mount(toml, "/p:/w", "/w")
        assert '"/p:/w"' in toml.read_text()


class TestGenerateLoadAndResolveBlock:
    def test_single_stage(self):
        load, args = _generate_load_and_resolve_block(
            [("download", "app:v1-download")],
            "/scratch/ce-images",
        )
        assert "app+v1-download.tar" in load
        assert "podman load" in load
        assert "DOWNLOAD_IMAGE_ID" in load
        assert "--build-arg DOWNLOAD_IMAGE=$DOWNLOAD_IMAGE_ID" in args
        # Architecture check present
        assert "IMAGE_ARCH" in load
        assert "NODE_ARCH" in load

    def test_multiple_stages(self):
        load, args = _generate_load_and_resolve_block(
            [("download", "app:v1-download"), ("runtime-download", "app:v1-runtime-download")],
            "/scratch/ce-images",
        )
        assert "app+v1-download.tar" in load
        assert "app+v1-runtime-download.tar" in load
        assert "DOWNLOAD_IMAGE_ID" in load
        assert "RUNTIME_DOWNLOAD_IMAGE_ID" in load
        assert "--build-arg DOWNLOAD_IMAGE=$DOWNLOAD_IMAGE_ID" in args
        assert "--build-arg RUNTIME_DOWNLOAD_IMAGE=$RUNTIME_DOWNLOAD_IMAGE_ID" in args

    def test_empty_stages(self):
        load, args = _generate_load_and_resolve_block([], "/scratch")
        assert load == ""
        assert args == ""

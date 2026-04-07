"""Tests for container helper functions."""

import subprocess
from unittest.mock import patch

import pytest

from fcw.commands.container import (
    _create_rebuilt_toml,
    _derive_container_name,
    _detect_container_runtime,
    _find_container_config,
    _generate_load_and_resolve_block,
    _merge_build_args,
    _parse_patch_mounts,
    _podman_setup_block,
    _resolve_remote_tar,
    _stage_to_build_arg_name,
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


class TestDeriveContainerName:
    def test_simple_version(self):
        assert _derive_container_name("app", "my-app:v2") == "app-v2"

    def test_dotted_version(self):
        assert _derive_container_name("app", "my-app:24.04") == "app-24-04"

    def test_no_colon(self):
        assert _derive_container_name("app", "latest") == "app-latest"

    def test_complex_suffix(self):
        assert _derive_container_name("web", "img:v1.2-beta") == "web-v1-2-beta"


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

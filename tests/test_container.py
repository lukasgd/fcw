"""Tests for container helper functions."""

import subprocess
from unittest.mock import patch

import pytest

from fcw.commands.container import (
    _detect_container_runtime,
    _find_container_config,
    _podman_setup_block,
    _resolve_remote_tar,
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

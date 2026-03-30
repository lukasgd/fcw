"""Tests for config validate helpers (local and remote phases)."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

from fcw.commands.config import (
    _human_time_ago,
    _validate_containers_remote,
    _validate_directories_remote,
    _validate_env_vars_remote,
    _validate_local,
)

# ---------------------------------------------------------------------------
# _human_time_ago
# ---------------------------------------------------------------------------

class TestHumanTimeAgo:
    def test_just_now(self):
        assert _human_time_ago(time.time()) == "just now"

    def test_minutes(self):
        assert _human_time_ago(time.time() - 300) == "5m ago"

    def test_hours(self):
        assert _human_time_ago(time.time() - 7200) == "2h ago"

    def test_days(self):
        assert _human_time_ago(time.time() - 172800) == "2d ago"


# ---------------------------------------------------------------------------
# _validate_local
# ---------------------------------------------------------------------------

class TestValidateLocal:
    def _make_ctx(self, config_file=None):
        ctx = MagicMock()
        ctx.obj = {"config_file": config_file}
        return ctx

    def test_config_loads(self, tmp_path, sample_config_yaml, monkeypatch):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        # Clear required env vars so we can test the error path
        for var in ["FIRECREST_URL", "FIRECREST_CLIENT_ID",
                     "FIRECREST_CLIENT_SECRET", "AUTH_TOKEN_URL"]:
            monkeypatch.delenv(var, raising=False)
        ctx = self._make_ctx(str(config_path))
        errors, warnings = [], []
        config = _validate_local(ctx, errors, warnings)
        assert config is not None
        assert config.project == "test-project"
        # Required env vars missing → errors
        assert any("Missing required" in e for e in errors)

    def test_missing_config_returns_none(self, tmp_path):
        ctx = self._make_ctx(str(tmp_path / "nonexistent.yaml"))
        errors, warnings = [], []
        config = _validate_local(ctx, errors, warnings)
        assert config is None
        assert any("Config file error" in e for e in errors)

    def test_job_script_warning(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        ctx = self._make_ctx(str(config_path))
        errors, warnings = [], []
        _validate_local(ctx, errors, warnings)
        # Scripts don't exist in tmp_path
        assert any("script not found" in w for w in warnings)

    def test_job_script_found(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        (tmp_path / "slurm").mkdir()
        (tmp_path / "slurm" / "preprocess.sh").write_text("#!/bin/bash")
        (tmp_path / "slurm" / "train.sh").write_text("#!/bin/bash")
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            ctx = self._make_ctx(str(config_path))
            errors, warnings = [], []
            _validate_local(ctx, errors, warnings)
            script_warnings = [w for w in warnings if "script not found" in w]
            assert len(script_warnings) == 0
        finally:
            os.chdir(old_cwd)

    def test_container_dockerfile_warning(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        ctx = self._make_ctx(str(config_path))
        errors, warnings = [], []
        _validate_local(ctx, errors, warnings)
        assert any("Dockerfile not found" in w for w in warnings)

    def test_directory_exists_locally(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        # Create some local directories
        (tmp_path / "data" / "raw").mkdir(parents=True)
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            ctx = self._make_ctx(str(config_path))
            errors, warnings = [], []
            _validate_local(ctx, errors, warnings)
            # data/raw exists, others don't
            assert any("'data/raw' exists locally" not in w for w in warnings)
            assert any("not found locally" in w for w in warnings)
        finally:
            os.chdir(old_cwd)

    def test_required_env_vars(self, tmp_path, sample_config_yaml, monkeypatch):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        monkeypatch.setenv("FIRECREST_URL", "https://example.com")
        monkeypatch.setenv("FIRECREST_CLIENT_ID", "id")
        monkeypatch.setenv("FIRECREST_CLIENT_SECRET", "secret")
        monkeypatch.setenv("AUTH_TOKEN_URL", "https://auth.example.com")
        ctx = self._make_ctx(str(config_path))
        errors, warnings = [], []
        _validate_local(ctx, errors, warnings)
        required_errors = [e for e in errors if "Missing required" in e]
        assert len(required_errors) == 0

    def test_optional_env_vars_warning(self, tmp_path, sample_config_yaml, monkeypatch):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        monkeypatch.delenv("FIRECREST_SYSTEM", raising=False)
        ctx = self._make_ctx(str(config_path))
        errors, warnings = [], []
        _validate_local(ctx, errors, warnings)
        assert any("FIRECREST_SYSTEM" in w for w in warnings)


# ---------------------------------------------------------------------------
# _validate_env_vars_remote
# ---------------------------------------------------------------------------

class TestValidateEnvVarsRemote:
    def test_user_matches(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_USER", "testuser")
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        # Nested user dict format (real API response)
        client.userinfo.return_value = {
            "user": {"id": "25680", "name": "testuser"},
            "groups": [],
        }
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert len(warnings) == 0

    def test_user_matches_flat(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_USER", "testuser")
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        # Flat format fallback
        client.userinfo.return_value = {"name": "testuser", "groups": []}
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert len(warnings) == 0

    def test_user_mismatch(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_USER", "wronguser")
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {
            "user": {"id": "25680", "name": "testuser"},
            "groups": [],
        }
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert any("does not match remote user" in w for w in warnings)

    def test_account_in_groups(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_ACCOUNT", "mygroup")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {
            "groups": [{"name": "mygroup"}, {"name": "other"}],
        }
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert len(warnings) == 0

    def test_account_not_in_groups(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_ACCOUNT", "missing")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {
            "groups": [{"name": "group1"}, {"name": "group2"}],
        }
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert any("not found in user groups" in w for w in warnings)

    def test_account_groups_as_strings(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_ACCOUNT", "mygroup")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {"groups": ["mygroup", "other"]}
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert len(warnings) == 0

    def test_home_mismatch(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_HOME", "/wrong/home")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {"home": "/real/home", "groups": []}
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.side_effect = Exception("not found")
            mock_ac.return_value = mock_async
            _validate_env_vars_remote(client, "sys", warnings)
        assert any("does not match remote home" in w for w in warnings)
        assert any("not accessible" in w for w in warnings)

    def test_home_matches(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_HOME", "/users/testuser")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {"home": "/users/testuser", "groups": []}
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.return_value = []
            mock_ac.return_value = mock_async
            _validate_env_vars_remote(client, "sys", warnings)
        assert len(warnings) == 0

    def test_scratch_mismatch(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_SCRATCH", "/wrong/scratch")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {"scratch": "/real/scratch", "groups": []}
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.side_effect = Exception("not found")
            mock_ac.return_value = mock_async
            _validate_env_vars_remote(client, "sys", warnings)
        assert any("does not match remote scratch" in w for w in warnings)

    def test_reservation_found(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_RESERVATION", "my-res")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {"groups": []}
        client.reservations.return_value = [
            {"ReservationName": "my-res", "State": "ACTIVE"},
            {"ReservationName": "other-res", "State": "ACTIVE"},
        ]
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert len(warnings) == 0

    def test_reservation_not_found(self, monkeypatch):
        monkeypatch.setenv("FIRECREST_RESERVATION", "missing-res")
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        client = MagicMock()
        client.userinfo.return_value = {"groups": []}
        client.reservations.return_value = [
            {"ReservationName": "other-res", "State": "ACTIVE"},
        ]
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert any("not found" in w for w in warnings)

    def test_userinfo_failure(self, monkeypatch):
        monkeypatch.delenv("FIRECREST_USER", raising=False)
        monkeypatch.delenv("FIRECREST_ACCOUNT", raising=False)
        monkeypatch.delenv("FIRECREST_HOME", raising=False)
        monkeypatch.delenv("FIRECREST_SCRATCH", raising=False)
        monkeypatch.delenv("FIRECREST_RESERVATION", raising=False)
        client = MagicMock()
        client.userinfo.side_effect = Exception("API error")
        warnings = []
        _validate_env_vars_remote(client, "sys", warnings)
        assert any("Could not fetch userinfo" in w for w in warnings)


# ---------------------------------------------------------------------------
# _validate_containers_remote
# ---------------------------------------------------------------------------

class TestValidateContainersRemote:
    def test_image_found(self, sample_config):
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.return_value = [
                {"name": "my-fcw-app+latest.sqsh", "type": "f"},
                {"name": "fcw-aux+latest.sqsh", "type": "f"},
            ]
            mock_ac.return_value = mock_async
            _validate_containers_remote(MagicMock(), "sys", sample_config, warnings)
        assert len(warnings) == 0

    def test_image_not_found(self, sample_config):
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.return_value = [
                {"name": "other-image.sqsh", "type": "f"},
            ]
            mock_ac.return_value = mock_async
            _validate_containers_remote(MagicMock(), "sys", sample_config, warnings)
        assert any("image not found" in w for w in warnings)

    def test_images_dir_not_accessible(self, sample_config):
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.side_effect = Exception("not found")
            mock_ac.return_value = mock_async
            _validate_containers_remote(MagicMock(), "sys", sample_config, warnings)
        assert any("not accessible" in w for w in warnings)


# ---------------------------------------------------------------------------
# _validate_directories_remote
# ---------------------------------------------------------------------------

class TestValidateDirectoriesRemote:
    def test_directory_exists_never_synced(self, sample_config, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.return_value = []
            mock_ac.return_value = mock_async
            _validate_directories_remote(MagicMock(), "sys", sample_config, warnings)
        # All directories found, none synced
        assert not any("not found on remote" in w for w in warnings)

    def test_directory_not_found(self, sample_config, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.side_effect = Exception("not found")
            mock_ac.return_value = mock_async
            _validate_directories_remote(MagicMock(), "sys", sample_config, warnings)
        assert any("not found on remote" in w for w in warnings)

    def test_with_sync_marker(self, sample_config, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # Write a sync marker for data/raw
        from fcw.commands.data import _write_last_sync_timestamp
        _write_last_sync_timestamp("data/raw", "push", ts=time.time() - 3600)

        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.return_value = []
            mock_ac.return_value = mock_async
            _validate_directories_remote(MagicMock(), "sys", sample_config, warnings)
        # No warnings — directories exist on remote
        assert not any("not found on remote" in w for w in warnings)

    def test_diff_mode(self, sample_config, monkeypatch, tmp_path):
        monkeypatch.chdir(tmp_path)
        # Create a local data directory with files
        (tmp_path / "data" / "raw").mkdir(parents=True)
        (tmp_path / "data" / "raw" / "file1.txt").write_text("data")
        (tmp_path / "data" / "raw" / "file2.txt").write_text("data")

        warnings = []
        with patch("fcw.commands.config.get_async_client") as mock_ac:
            mock_async = AsyncMock()
            mock_async.list_files.return_value = [
                {"name": "file1.txt", "type": "f"},
                {"name": "file2.txt", "type": "f"},
                {"name": "file3.txt", "type": "f"},
            ]
            mock_ac.return_value = mock_async
            _validate_directories_remote(MagicMock(), "sys", sample_config, warnings, diff=True)
        # No errors expected — just informational output
        assert not any("not found on remote" in w for w in warnings)

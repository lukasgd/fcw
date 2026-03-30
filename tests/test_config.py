"""Tests for config loading, env expansion, and path resolution."""

import os

import pytest
import yaml

from fcw.core.config import (
    DirectoryType,
    FcwConfig,
    expand_config_refs,
    expand_env_vars,
    generate_default_config,
    load_config,
)


class TestLoadConfig:
    def test_basic(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        config = load_config(str(config_path))

        assert config.project == "test-project"
        assert config.workdir.remote == "/scratch/user/test-project"
        assert config.workdir.local == "."
        assert "data/raw" in config.directories
        assert "preprocess" in config.jobs
        assert "app" in config.containers

    def test_missing_file_returns_default(self, tmp_path):
        os.chdir(tmp_path)
        config = load_config()
        assert config.project == "default"
        assert config.workdir.remote == ""

    def test_explicit_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(str(tmp_path / "nonexistent.yaml"))

    def test_directory_types_parsed(self, sample_config):
        assert sample_config.directories["data/raw"].type == DirectoryType.IN
        assert sample_config.directories["data/processed"].type == DirectoryType.OUT
        assert sample_config.directories["code"].type == DirectoryType.BOTH

    def test_job_env_parsed(self, sample_config):
        assert sample_config.jobs["preprocess"].env["DATA_IN"] == "data/raw"
        assert sample_config.jobs["train"].time == "12:00:00"
        assert sample_config.jobs["train"].nodes == 1

    def test_container_parsed(self, sample_config):
        assert sample_config.containers["app"].tag == "my-fcw-app:latest"
        assert sample_config.containers["app"].remote_path == "./ce-images/"


class TestEnvVarExpansion:
    def test_simple_var(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        assert expand_env_vars("${TEST_VAR}") == "hello"

    def test_var_with_default_present(self, monkeypatch):
        monkeypatch.setenv("TEST_VAR", "hello")
        assert expand_env_vars("${TEST_VAR:-world}") == "hello"

    def test_var_with_default_missing(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert expand_env_vars("${MISSING_VAR:-fallback}") == "fallback"

    def test_var_missing_no_default(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        assert expand_env_vars("${MISSING_VAR}") == "${MISSING_VAR}"

    def test_multiple_vars(self, monkeypatch):
        monkeypatch.setenv("A", "1")
        monkeypatch.setenv("B", "2")
        assert expand_env_vars("${A}/${B}") == "1/2"

    def test_no_vars(self):
        assert expand_env_vars("plain text") == "plain text"


class TestConfigRefExpansion:
    def test_simple_ref(self):
        data = {"workdir": {"remote": "/scratch/user"}}
        assert expand_config_refs("${workdir.remote}/data", data) == "/scratch/user/data"

    def test_missing_ref_unchanged(self):
        data = {"workdir": {"remote": "/scratch"}}
        assert expand_config_refs("${missing.key}", data) == "${missing.key}"

    def test_non_string_ref_unchanged(self):
        data = {"workdir": {"nodes": 4}}
        assert expand_config_refs("${workdir.nodes}", data) == "${workdir.nodes}"


class TestResolvePath:
    def test_remote(self, sample_config):
        assert sample_config.resolve_path("data/raw", remote=True) == "/scratch/user/test-project/data/raw"

    def test_local(self, sample_config):
        result = sample_config.resolve_path("data/raw", remote=False)
        assert result.endswith("data/raw")

    def test_absolute_unchanged(self, sample_config):
        assert sample_config.resolve_path("/absolute/path", remote=True) == "/absolute/path"


class TestDirectoryTypeEnforcement:
    def test_can_upload_in(self, sample_config):
        assert sample_config.can_upload("data/raw") is True

    def test_cannot_upload_out(self, sample_config):
        assert sample_config.can_upload("data/processed") is False

    def test_can_upload_both(self, sample_config):
        assert sample_config.can_upload("code") is True

    def test_can_download_out(self, sample_config):
        assert sample_config.can_download("data/processed") is True

    def test_cannot_download_in(self, sample_config):
        assert sample_config.can_download("data/raw") is False

    def test_can_download_both(self, sample_config):
        assert sample_config.can_download("code") is True

    def test_prefix_match(self, sample_config):
        assert sample_config.can_upload("data/raw/subdir") is True
        assert sample_config.can_upload("data/processed/subdir") is False

    def test_unknown_dir_defaults_both(self, sample_config):
        assert sample_config.can_upload("unknown") is True
        assert sample_config.can_download("unknown") is True


class TestResolveContainerImage:
    def test_with_remote_path(self, sample_config):
        cont = sample_config.containers["app"]
        result = sample_config.resolve_container_image(cont)
        assert result == "/scratch/user/test-project/ce-images/my-fcw-app+latest.sqsh"

    def test_without_remote_path(self, sample_config):
        cont = sample_config.containers["aux"]
        result = sample_config.resolve_container_image(cont)
        # No remote_path configured, falls back to ce-images/
        assert result == "/scratch/user/test-project/ce-images/fcw-aux+latest.sqsh"

    def test_tag_with_colon_converted(self, sample_config):
        cont = sample_config.containers["app"]
        result = sample_config.resolve_container_image(cont)
        assert "+" in result
        assert ":" not in os.path.basename(result)


class TestResolveContainerImagesDir:
    def test_with_remote_path(self, sample_config):
        cont = sample_config.containers["app"]
        result = sample_config.resolve_container_images_dir(cont)
        assert result == "/scratch/user/test-project/ce-images"

    def test_without_remote_path(self, sample_config):
        cont = sample_config.containers["aux"]
        result = sample_config.resolve_container_images_dir(cont)
        assert result == "/scratch/user/test-project/ce-images"

    def test_normalized(self, sample_config):
        """Paths should be normalized (no trailing slashes or ./)."""
        cont = sample_config.containers["app"]
        result = sample_config.resolve_container_images_dir(cont)
        assert not result.endswith("/")
        assert "/." not in result


class TestDirectoryTypeMissing:
    def test_missing_value_alias(self):
        assert DirectoryType("input") == DirectoryType.IN
        assert DirectoryType("output") == DirectoryType.OUT
        assert DirectoryType("bidirectional") == DirectoryType.BOTH


class TestUnknownConfigKeys:
    def test_warns_on_unknown_key(self, tmp_path):
        config_yaml = "project: test\nunknown_key: value\n"
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(config_yaml)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(str(config_path))
            assert len(w) == 1
            assert "unknown_key" in str(w[0].message)

    def test_no_warning_on_valid_keys(self, tmp_path):
        config_yaml = "project: test\nworkdir:\n  remote: /scratch\n"
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(config_yaml)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            load_config(str(config_path))
            assert len(w) == 0


class TestGenerateDefaultConfig:
    def test_valid_yaml(self):
        config_str = generate_default_config()
        data = yaml.safe_load(config_str)
        assert data["project"] == "my-fcw-app"
        assert "workdir" in data
        assert "directories" in data
        assert "jobs" in data

    def test_roundtrip(self, tmp_path):
        config_str = generate_default_config()
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(config_str)
        config = load_config(str(config_path))
        assert config.project == "my-fcw-app"

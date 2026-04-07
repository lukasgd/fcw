"""Tests for config loading, env expansion, and path resolution."""

import os
import textwrap

import pytest
import yaml

from fcw.core.config import (
    ContainerConfig,
    DirectoryType,
    FcwConfig,
    add_container_to_config,
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

    def test_job_env_parsed(self, sample_config):
        assert sample_config.jobs["preprocess"].env["DATA_IN"] == "data/raw"
        assert sample_config.jobs["train"].time == "12:00:00"
        assert sample_config.jobs["train"].nodes == 1

    def test_container_parsed(self, sample_config):
        assert sample_config.containers["app"].tag == "my-fcw-app:latest"
        assert sample_config.containers["app"].remote_path == "./ce-images/"
        assert sample_config.containers["app"].toml == "./env/container.toml"

    def test_job_container_parsed(self, sample_config):
        assert sample_config.jobs["preprocess"].container == "app"
        assert sample_config.jobs["train"].container == "app"

    def test_build_args_parsed(self, tmp_path):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(textwrap.dedent("""\
            project: test
            workdir:
              remote: /tmp/test
              local: .
            containers:
              app:
                file: ./Dockerfile
                tag: my-app:latest
                build_args:
                  BASE_IMAGE: ubuntu:24.04
                  EXTRA_FLAG: value
        """))
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = load_config(str(config_path))
        finally:
            os.chdir(old_cwd)
        assert cfg.containers["app"].build_args == {
            "BASE_IMAGE": "ubuntu:24.04",
            "EXTRA_FLAG": "value",
        }

    def test_build_args_none_when_absent(self, sample_config):
        assert sample_config.containers["app"].build_args is None

    def test_platform_parsed(self, tmp_path):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(textwrap.dedent("""\
            project: test
            workdir:
              remote: /tmp/test
              local: .
            containers:
              app:
                file: ./Dockerfile
                tag: my-app:latest
                platform: linux/arm64
        """))
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            cfg = load_config(str(config_path))
        finally:
            os.chdir(old_cwd)
        assert cfg.containers["app"].platform == "linux/arm64"


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

    def test_can_upload_unknown_defaults_both(self, sample_config):
        assert sample_config.can_upload("something") is True

    def test_can_download_out(self, sample_config):
        assert sample_config.can_download("data/processed") is True

    def test_cannot_download_in(self, sample_config):
        assert sample_config.can_download("data/raw") is False

    def test_can_download_unknown_defaults_both(self, sample_config):
        assert sample_config.can_download("something") is True

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
        # Container TOML inlining fields
        assert data["containers"]["app"]["toml"] == "./env/container.toml"
        assert data["jobs"]["train"]["container"] == "app"

    def test_roundtrip(self, tmp_path):
        config_str = generate_default_config()
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(config_str)
        config = load_config(str(config_path))
        assert config.project == "my-fcw-app"


class TestAddContainerToConfig:
    def test_appends_entry(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        add_container_to_config(
            config_path,
            "app-v2",
            ContainerConfig(
                file="./Dockerfile",
                tag="my-app:v2",
                remote_path="./ce-images/",
                toml="./env/container-v2.toml",
            ),
        )
        config = load_config(str(config_path))
        assert "app-v2" in config.containers
        assert config.containers["app-v2"].tag == "my-app:v2"
        assert config.containers["app-v2"].toml == "./env/container-v2.toml"
        # Original entries still intact
        assert "app" in config.containers
        assert config.containers["app"].tag == "my-fcw-app:latest"
        assert "aux" in config.containers

    def test_duplicate_name_raises(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        with pytest.raises(ValueError, match="already exists"):
            add_container_to_config(
                config_path,
                "app",
                ContainerConfig(file="./Dockerfile", tag="dup:v1"),
            )

    def test_preserves_comments(self, tmp_path):
        yaml_with_comments = """\
# My project
project: test

# Container defs
containers:
  app:
    file: ./Dockerfile
    tag: my-app:latest

jobs:
  train:
    script: train.sh
"""
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(yaml_with_comments)
        add_container_to_config(
            config_path,
            "app-v2",
            ContainerConfig(file="./Dockerfile", tag="my-app:v2"),
        )
        content = config_path.read_text()
        assert "# My project" in content
        assert "# Container defs" in content
        # New entry is present
        assert "app-v2:" in content
        assert "my-app:v2" in content
        # Jobs section still intact
        assert "jobs:" in content

    def test_no_containers_block_raises(self, tmp_path):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text("project: test\n")
        with pytest.raises(ValueError, match="containers"):
            add_container_to_config(
                config_path,
                "app",
                ContainerConfig(file="./Dockerfile", tag="app:v1"),
            )

    def test_optional_fields_omitted(self, tmp_path, sample_config_yaml):
        config_path = tmp_path / "fcw.yaml"
        config_path.write_text(sample_config_yaml)
        add_container_to_config(
            config_path,
            "minimal",
            ContainerConfig(file="./Dockerfile", tag="minimal:v1"),
        )
        # Verify it loads correctly and has expected fields
        config = load_config(str(config_path))
        assert "minimal" in config.containers
        assert config.containers["minimal"].file == "./Dockerfile"
        assert config.containers["minimal"].tag == "minimal:v1"
        assert config.containers["minimal"].remote_path is None
        assert config.containers["minimal"].stage is None
        assert config.containers["minimal"].toml is None
        # Verify the raw YAML doesn't contain optional fields for this entry
        content = config_path.read_text()
        # Find the minimal block in raw text
        idx = content.index("  minimal:\n")
        block = content[idx:]
        # Grab lines until next entry or top-level key
        block_lines = block.split("\n")[1:]  # skip "  minimal:" itself
        entry_lines = []
        for line in block_lines:
            if line.startswith("    "):
                entry_lines.append(line.strip())
            elif line.strip() == "":
                continue
            else:
                break
        assert "file: ./Dockerfile" in entry_lines
        assert "tag: minimal:v1" in entry_lines
        assert not any("remote_path" in l for l in entry_lines)
        assert not any("stage" in l for l in entry_lines)
        assert not any("toml" in l for l in entry_lines)

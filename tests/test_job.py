"""Tests for SBATCH parsing, env injection, and script manipulation."""

import pytest

from fcw.commands.job import (
    _apply_sbatch_overrides,
    _inject_env_vars,
    _parse_sbatch_args,
    _resolve_job_env,
)
from fcw.core.config import JobConfig


SAMPLE_SCRIPT = """\
#!/bin/bash -l

#SBATCH --job-name preprocess
#SBATCH --time 12:00:00
#SBATCH --nodes 1

set -euxo pipefail

echo "hello"
"""


class TestApplySbatchOverrides:
    def test_replace_existing(self):
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {"time": "24:00:00"})
        assert "#SBATCH --time=24:00:00" in result
        assert "12:00:00" not in result

    def test_insert_new(self):
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {"gpus-per-node": "4"})
        assert "#SBATCH --gpus-per-node=4" in result
        # Original directives preserved
        assert "#SBATCH --time 12:00:00" in result

    def test_insert_after_last_sbatch(self):
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {"gpus-per-node": "4"})
        lines = result.split("\n")
        gpus_idx = next(i for i, l in enumerate(lines) if "gpus-per-node" in l)
        nodes_idx = next(i for i, l in enumerate(lines) if "--nodes" in l)
        assert gpus_idx > nodes_idx

    def test_no_sbatch_in_script(self):
        script = "#!/bin/bash\necho hello\n"
        result = _apply_sbatch_overrides(script, {"time": "1:00:00"})
        assert "#SBATCH --time=1:00:00" in result

    def test_empty_overrides(self):
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {})
        assert result == SAMPLE_SCRIPT

    def test_multiple_overrides(self):
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {"time": "24:00:00", "nodes": "4"})
        assert "#SBATCH --time=24:00:00" in result
        assert "#SBATCH --nodes=4" in result


class TestInjectEnvVars:
    def test_basic_injection(self):
        result = _inject_env_vars(SAMPLE_SCRIPT, {"DATA_DIR": "/data"})
        assert 'export DATA_DIR="${DATA_DIR:-/data}"' in result

    def test_inserted_after_sbatch_block(self):
        result = _inject_env_vars(SAMPLE_SCRIPT, {"DATA_DIR": "/data"})
        lines = result.split("\n")
        export_idx = next(i for i, l in enumerate(lines) if "export DATA_DIR" in l)
        last_sbatch_idx = max(i for i, l in enumerate(lines) if l.strip().startswith("#SBATCH"))
        assert export_idx > last_sbatch_idx

    def test_shell_default_syntax(self):
        result = _inject_env_vars(SAMPLE_SCRIPT, {"VAR": "default_val"})
        assert "${VAR:-default_val}" in result

    def test_empty_env(self):
        result = _inject_env_vars(SAMPLE_SCRIPT, {})
        assert result == SAMPLE_SCRIPT

    def test_multiple_vars(self):
        result = _inject_env_vars(SAMPLE_SCRIPT, {"A": "1", "B": "2"})
        assert 'export A="${A:-1}"' in result
        assert 'export B="${B:-2}"' in result


class TestParseSbatchArgs:
    def test_with_separator(self):
        overrides, remaining = _parse_sbatch_args(["--time", "1:00", "--", "train.sh"])
        assert overrides == {"time": "1:00"}
        assert remaining == ["train.sh"]

    def test_key_equals_value(self):
        overrides, remaining = _parse_sbatch_args(["--time=1:00", "--", "train.sh"])
        assert overrides == {"time": "1:00"}
        assert remaining == ["train.sh"]

    def test_no_separator(self):
        overrides, remaining = _parse_sbatch_args(["train.sh"])
        assert overrides == {}
        assert remaining == ["train.sh"]

    def test_multiple_overrides(self):
        overrides, remaining = _parse_sbatch_args(
            ["--time", "1:00", "--nodes", "4", "--", "train.sh"]
        )
        assert overrides == {"time": "1:00", "nodes": "4"}
        assert remaining == ["train.sh"]

    def test_empty_args(self):
        overrides, remaining = _parse_sbatch_args([])
        assert overrides == {}
        assert remaining == []


class TestResolveJobEnv:
    def test_expands_relative_paths(self, sample_config):
        job_config = sample_config.jobs["preprocess"]
        env = _resolve_job_env(sample_config, job_config, {})
        assert env["DATA_IN"] == "/scratch/user/test-project/data/raw"
        assert env["DATA_OUT"] == "/scratch/user/test-project/data/processed"

    def test_overrides_applied(self, sample_config):
        job_config = sample_config.jobs["train"]
        # Relative override values are resolved against remote workdir
        env = _resolve_job_env(sample_config, job_config, {"EXTRA": "mydir"})
        assert env["EXTRA"] == "/scratch/user/test-project/mydir"
        # Absolute override values are preserved as-is
        env = _resolve_job_env(sample_config, job_config, {"EXTRA": "/abs/path"})
        assert env["EXTRA"] == "/abs/path"

    def test_override_replaces_config(self, sample_config):
        job_config = sample_config.jobs["preprocess"]
        env = _resolve_job_env(sample_config, job_config, {"DATA_IN": "/custom/path"})
        assert env["DATA_IN"] == "/custom/path"

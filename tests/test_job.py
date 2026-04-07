"""Tests for SBATCH parsing, env injection, script manipulation, and container TOML inlining."""

import pytest

from fcw.commands.job import (
    _apply_sbatch_overrides,
    _build_container_toml,
    _inject_container_toml,
    _inject_env_vars,
    _parse_sbatch_args,
    _resolve_job_env,
)
from fcw.core.config import ContainerConfig, FcwConfig, JobConfig, WorkdirConfig


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
        assert "#SBATCH --time 12:00:00" not in result

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
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {"time": "12:00:00", "nodes": "4"})
        assert "#SBATCH --time=12:00:00" in result
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


class TestSbatchOverridesCLI:
    """Integration tests: verify SBATCH overrides pass through Typer/Click."""

    def test_submit_with_overrides_dry_run(self, tmp_path):
        """SBATCH options before -- must not be rejected by Typer."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        # Minimal config and script
        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
        )
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\n#SBATCH --time 01:00:00\necho hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "submit", "--dry-run",
                "--time", "12:00:00", "--nodes", "4",
                "--", str(script),
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert "--time=12:00:00" in result.output or "--time 12:00:00" in result.output
        assert "--nodes=4" in result.output or "--nodes 4" in result.output

    def test_submit_raw_script_infers_container(self, tmp_path):
        """Submitting a raw script that uses FCW_CONTAINER_TOML infers the container."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
            "containers:\n  app:\n    file: Dockerfile\n    tag: my-app:latest\n"
            "    remote_path: ./ce-images/\n    toml: ./env/container.toml\n"
        )
        toml_dir = tmp_path / "env"
        toml_dir.mkdir()
        (toml_dir / "container.toml").write_text('[container]\nimage = "placeholder"\n')
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\n"
            "srun --environment ${FCW_CONTAINER_TOML} echo hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, ["job", "submit", "--dry-run", str(script)])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert "FCW_CONTAINER_TOML" in result.output
        assert "FCWEOF" in result.output  # heredoc was injected

    def test_submit_without_separator(self, tmp_path):
        """Without --, args are treated as script name (no overrides)."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
        )
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\n#SBATCH --time 01:00:00\necho hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "submit", "--dry-run", str(script),
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        # Original time should be preserved (no override)
        assert "#SBATCH --time 01:00:00" in result.output


    def test_submit_sbatch_after_script_is_error(self, tmp_path):
        """SBATCH options after the script name should be a hard error."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
        )
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\necho hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "submit", "--dry-run", str(script), "--nodes", "4",
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 1
        assert "--nodes" in result.output
        assert "SBATCH" in result.output

    def test_submit_flag_as_job_name_is_error(self, tmp_path):
        """SBATCH option in place of job name should give a clear error."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
        )
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\necho hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "submit", "--dry-run", "--nodes", "4", str(script),
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 1
        output = " ".join(result.output.split())  # normalize Rich line wrapping
        assert "looks like an SBATCH option" in output
        assert "-- separator" in output

    def test_submit_sbatch_after_script_with_separator_is_error(self, tmp_path):
        """SBATCH options after script even with -- separator should error."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
        )
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\necho hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "submit", "--dry-run",
                "--time", "1:00:00", "--", str(script), "--nodes", "4",
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 1
        assert "--nodes" in result.output

    def test_submit_fcw_flags_after_script_still_work(self, tmp_path):
        """fcw-specific flags like --set after the script should not be rejected."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
        )
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\necho hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "submit", "--dry-run", str(script), "--set", "X=1",
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert "export X=" in result.output


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


class TestInjectContainerToml:
    def test_basic_injection(self):
        toml = 'image = "/scratch/app.sqsh"\nwritable = true\n'
        result = _inject_container_toml(SAMPLE_SCRIPT, toml)
        assert 'cat > /dev/shm/fcw-container-${SLURM_JOB_ID}.toml' in result
        assert 'image = "/scratch/app.sqsh"' in result
        assert "export FCW_CONTAINER_TOML=" in result

    def test_quoted_heredoc(self):
        toml = 'mounts = ["${SCRATCH}"]\n'
        result = _inject_container_toml(SAMPLE_SCRIPT, toml)
        assert "<< 'FCWEOF'" in result
        assert "${SCRATCH}" in result

    def test_inserted_after_sbatch(self):
        toml = 'image = "/scratch/app.sqsh"\n'
        result = _inject_container_toml(SAMPLE_SCRIPT, toml)
        lines = result.split("\n")
        toml_idx = next(i for i, l in enumerate(lines) if "FCW_CONTAINER_TOML" in l)
        last_sbatch_idx = max(i for i, l in enumerate(lines) if l.strip().startswith("#SBATCH"))
        assert toml_idx > last_sbatch_idx

    def test_before_env_vars(self):
        toml = 'image = "/scratch/app.sqsh"\n'
        script = _inject_container_toml(SAMPLE_SCRIPT, toml)
        result = _inject_env_vars(script, {"DATA_DIR": "/data"})
        lines = result.split("\n")
        toml_idx = next(i for i, l in enumerate(lines) if "FCW_CONTAINER_TOML" in l)
        export_idx = next(i for i, l in enumerate(lines) if "export DATA_DIR" in l)
        assert toml_idx < export_idx

    def test_no_sbatch_in_script(self):
        script = "#!/bin/bash\necho hello\n"
        toml = 'image = "/scratch/app.sqsh"\n'
        result = _inject_container_toml(script, toml)
        assert "FCW_CONTAINER_TOML" in result
        # Should be after shebang
        lines = result.split("\n")
        assert lines[0] == "#!/bin/bash"


class TestBuildContainerToml:
    @pytest.fixture
    def config_with_toml(self, tmp_path):
        toml_path = tmp_path / "env" / "container.toml"
        toml_path.parent.mkdir(parents=True)
        toml_path.write_text(
            'image = "placeholder"\n\nmounts = [\n    "${SCRATCH}",\n]\n\n'
            'workdir = "/workspace"\n\nwritable = true\n'
        )
        return FcwConfig(
            project="test",
            workdir=WorkdirConfig(remote="/scratch/user/test"),
            containers={
                "app": ContainerConfig(
                    file="./Dockerfile",
                    tag="my-app:latest",
                    remote_path="./ce-images/",
                    toml=str(toml_path),
                ),
            },
        )

    @pytest.fixture
    def config_without_toml(self):
        return FcwConfig(
            project="test",
            workdir=WorkdirConfig(remote="/scratch/user/test"),
            containers={
                "app": ContainerConfig(
                    file="./Dockerfile",
                    tag="my-app:latest",
                    remote_path="./ce-images/",
                ),
            },
        )

    def test_reads_toml_and_overrides_image(self, config_with_toml):
        result = _build_container_toml(config_with_toml, "app")
        assert 'image = "/scratch/user/test/ce-images/my-app+latest.sqsh"' in result
        assert "${SCRATCH}" in result
        assert "workdir" in result

    def test_generates_minimal_without_toml(self, config_without_toml):
        result = _build_container_toml(config_without_toml, "app")
        assert 'image = "/scratch/user/test/ce-images/my-app+latest.sqsh"' in result
        assert "writable = true" in result

    def test_unknown_container_exits(self, config_without_toml):
        from click.exceptions import Exit
        with pytest.raises(Exit):
            _build_container_toml(config_without_toml, "nonexistent")

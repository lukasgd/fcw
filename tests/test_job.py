"""Tests for SBATCH parsing, env injection, script manipulation, and container TOML inlining."""

import pytest

from collections import Counter

from fcw.commands.job import (
    _apply_sbatch_overrides,
    _build_container_toml,
    _build_jobs_table,
    _fmt_duration,
    _fmt_epoch,
    _follow_stream,
    _follow_streams,
    _inject_container_toml,
    _inject_env_vars,
    _job_stream_paths,
    _parse_sbatch_args,
    _report_final_state,
    _resolve_job_env,
    _rewrite_environment_path,
    _warn_env_bindings,
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

    def test_flag_renders_bare(self):
        """An empty-string value is a valueless flag: `#SBATCH --exclusive` (no '=')."""
        result = _apply_sbatch_overrides(SAMPLE_SCRIPT, {"exclusive": ""})
        assert "#SBATCH --exclusive" in result
        assert "#SBATCH --exclusive=" not in result

    def test_flag_not_duplicated_when_present(self):
        script = "#!/bin/bash\n#SBATCH --exclusive\necho hi\n"
        result = _apply_sbatch_overrides(script, {"exclusive": ""})
        assert result.count("#SBATCH --exclusive") == 1

    def test_same_value_no_warning(self, capsys):
        """No warning when override matches existing script value."""
        _apply_sbatch_overrides(SAMPLE_SCRIPT, {"nodes": "1"})
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "Overriding" not in output

    def test_different_value_warns(self, capsys):
        """Warning when override replaces a different script value."""
        _apply_sbatch_overrides(SAMPLE_SCRIPT, {"nodes": "4"})
        captured = capsys.readouterr()
        output = captured.out + captured.err
        assert "Overriding" in output
        assert "--nodes" in output


class TestJobConfigSbatchOptions:
    def test_returns_set_fields(self):
        job = JobConfig(script="train.sh", time="12:00:00", nodes=4)
        assert job.sbatch_options() == {"time": "12:00:00", "nodes": "4"}

    def test_skips_none_fields(self):
        job = JobConfig(script="train.sh", time="1:00:00")
        opts = job.sbatch_options()
        assert "time" in opts
        assert "nodes" not in opts

    def test_converts_underscores_to_hyphens(self):
        job = JobConfig(script="train.sh", gpus_per_node=4, cpus_per_task=8)
        opts = job.sbatch_options()
        assert opts == {"gpus-per-node": "4", "cpus-per-task": "8"}

    def test_excludes_non_sbatch_fields(self):
        job = JobConfig(script="train.sh", container="app", env={"K": "V"}, time="1:00:00")
        opts = job.sbatch_options()
        assert "script" not in opts
        assert "container" not in opts
        assert "env" not in opts

    def test_sbatch_catchall_passthrough(self):
        job = JobConfig(script="train.sh", sbatch={"mem": "32G", "qos": "normal"})
        opts = job.sbatch_options()
        assert opts == {"mem": "32G", "qos": "normal"}

    def test_sbatch_catchall_kebab_and_stringify(self):
        job = JobConfig(script="train.sh", sbatch={"mem_per_gpu": "8G", "threads_per_core": 2})
        opts = job.sbatch_options()
        assert opts == {"mem-per-gpu": "8G", "threads-per-core": "2"}

    def test_sbatch_boolean_flags(self):
        job = JobConfig(script="train.sh", sbatch={"exclusive": True, "requeue": False})
        opts = job.sbatch_options()
        assert opts["exclusive"] == ""   # valueless flag
        assert "requeue" not in opts     # False -> omitted

    def test_typed_field_wins_over_catchall(self):
        job = JobConfig(script="train.sh", nodes=4, sbatch={"nodes": 8})
        assert job.sbatch_options()["nodes"] == "4"

    def test_empty_sbatch_is_noop(self):
        job = JobConfig(script="train.sh", time="1:00:00")
        assert job.sbatch_options() == {"time": "1:00:00"}

    def test_empty_when_no_sbatch_fields(self):
        job = JobConfig(script="train.sh")
        assert job.sbatch_options() == {}


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

    def test_bare_flag(self):
        """A value-less flag parses to an empty-string value (CLI flag support)."""
        overrides, remaining = _parse_sbatch_args(["--exclusive", "--", "train.sh"])
        assert overrides == {"exclusive": ""}
        assert remaining == ["train.sh"]

    def test_flag_interleaved_with_valued(self):
        overrides, remaining = _parse_sbatch_args(
            ["--exclusive", "--nodes", "4", "--", "train.sh"]
        )
        assert overrides == {"exclusive": "", "nodes": "4"}
        assert remaining == ["train.sh"]


class TestRewriteEnvironmentPath:
    def test_rewrites_matching_path(self):
        script = "srun --environment ./env/foo.toml bash -c x\n"
        out = _rewrite_environment_path(script, "./env/foo.toml")
        assert "--environment ${FCW_CONTAINER_TOML}" in out
        assert "foo.toml" not in out

    def test_matches_modulo_leading_dotslash(self):
        script = "srun --environment=env/foo.toml app\n"
        out = _rewrite_environment_path(script, "./env/foo.toml")
        assert "--environment=${FCW_CONTAINER_TOML}" in out

    def test_leaves_other_paths_alone(self):
        script = "srun --environment ./env/other.toml app\n"
        out = _rewrite_environment_path(script, "./env/foo.toml")
        assert out == script

    def test_leaves_fcw_var_untouched(self):
        script = "srun --environment ${FCW_CONTAINER_TOML} app\n"
        out = _rewrite_environment_path(script, "./env/foo.toml")
        assert out == script


class TestWarnEnvBindings:
    # Warnings are diagnostics -> emitted on stderr.
    def test_managed_var_no_warning(self, capsys):
        _warn_env_bindings("srun --environment ${FCW_CONTAINER_TOML} x\n",
                           "./env/foo.toml", bound=True)
        assert capsys.readouterr().err == ""

    def test_managed_path_no_warning(self, capsys):
        _warn_env_bindings("srun --environment ./env/foo.toml x\n",
                           "./env/foo.toml", bound=True)
        assert capsys.readouterr().err == ""

    def test_foreign_environment_warns(self, capsys):
        _warn_env_bindings("srun --environment ./env/other.toml x\n",
                           "./env/foo.toml", bound=True)
        assert "outside the" in capsys.readouterr().err

    def test_bound_but_no_reference_warns(self, capsys):
        _warn_env_bindings("srun echo hi\n", "./env/foo.toml", bound=True)
        assert "no effect" in capsys.readouterr().err

    def test_reference_without_binding_warns(self, capsys):
        _warn_env_bindings("srun --environment ${FCW_CONTAINER_TOML} x\n",
                           None, bound=False)
        assert "none is bound" in capsys.readouterr().err

    def test_no_binding_no_reference_silent(self, capsys):
        _warn_env_bindings("srun echo hi\n", None, bound=False)
        assert capsys.readouterr().err == ""


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

    def _write_container_config(self, tmp_path):
        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
            "containers:\n  app:\n    file: Dockerfile\n    tag: my-app:latest\n"
            "    remote_path: ./ce-images/\n    toml: ./env/container.toml\n"
        )
        toml_dir = tmp_path / "env"
        toml_dir.mkdir()
        (toml_dir / "container.toml").write_text('image = "placeholder"\n')

    def test_submit_raw_script_no_inference(self, tmp_path):
        """A raw script is NOT auto-bound: it warns and injects nothing."""
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        self._write_container_config(tmp_path)
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
        assert "FCWEOF" not in result.output       # no heredoc injected
        assert "none is bound" in result.output    # warned instead

    def test_submit_raw_script_explicit_container(self, tmp_path):
        """--container binds a raw script's env explicitly (heredoc injected)."""
        from typer.testing import CliRunner
        from fcw.cli import app
        from fcw.commands import container as container_mod
        import os

        self._write_container_config(tmp_path)
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\n"
            "srun --environment ${FCW_CONTAINER_TOML} echo hi\n"
        )

        runner = CliRunner()
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            import unittest.mock as mock
            with mock.patch.object(container_mod, "_resync_container_patches", lambda *a, **k: None):
                result = runner.invoke(
                    app, ["job", "submit", "--container", "app", "--dry-run", str(script)]
                )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert "FCWEOF" in result.output  # heredoc injected via explicit binding

    def test_submit_rewrites_hardcoded_environment_path(self, tmp_path):
        """A script's hardcoded --environment <configured toml> is redirected."""
        from typer.testing import CliRunner
        from fcw.cli import app
        from fcw.commands import container as container_mod
        import os
        import unittest.mock as mock

        self._write_container_config(tmp_path)
        script = tmp_path / "train.sh"
        script.write_text(
            "#!/bin/bash\n#SBATCH --job-name test\n"
            "srun --environment ./env/container.toml echo hi\n"
        )

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            with mock.patch.object(container_mod, "_resync_container_patches", lambda *a, **k: None):
                result = CliRunner().invoke(
                    app, ["job", "submit", "--container", "app", "--dry-run", str(script)]
                )
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 0, result.output
        assert "--environment ${FCW_CONTAINER_TOML}" in result.output
        assert "--environment ./env/container.toml" not in result.output

    def test_submit_container_environment_mutually_exclusive(self, tmp_path):
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        self._write_container_config(tmp_path)
        script = tmp_path / "train.sh"
        script.write_text("#!/bin/bash\n#SBATCH --job-name test\necho hi\n")
        (tmp_path / "env" / "alt.toml").write_text('image = "x"\n')

        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = CliRunner().invoke(app, [
                "job", "submit", "--container", "app",
                "--environment", "./env/alt.toml", "--dry-run", str(script),
            ])
        finally:
            os.chdir(old_cwd)

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

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


class TestRunContainerIntegration:
    """`fcw job run` container support + patch-awareness."""

    def _mk_config(self, tmp_path):
        config = tmp_path / "fcw.yaml"
        config.write_text(
            "project: test\nworkdir:\n  remote: /tmp/test\n  local: .\n"
            "containers:\n  app:\n    file: Dockerfile\n    tag: my-app:latest\n"
            "    remote_path: ./ce-images/\n    toml: ./env/container.toml\n"
        )
        toml_dir = tmp_path / "env"
        toml_dir.mkdir()
        (toml_dir / "container.toml").write_text('image = "placeholder"\n')

    def test_rejects_container_and_environment_together(self, tmp_path):
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        self._mk_config(tmp_path)
        env_toml = tmp_path / "other.toml"
        env_toml.write_text('image = "other"\n')

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "run", "--container", "app", "--environment", str(env_toml),
                "--dry-run", "--", "echo hi",
            ])
        finally:
            os.chdir(old)

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output

    def test_dry_run_injects_container_toml_and_csrun(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from fcw.cli import app
        from fcw.commands import container as container_mod
        import os

        self._mk_config(tmp_path)
        monkeypatch.setattr(container_mod, "_resync_container_patches",
                            lambda *a, **kw: None)

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "run", "--container", "app", "--dry-run", "--", "csrun hostname",
            ])
        finally:
            os.chdir(old)

        assert result.exit_code == 0, result.output
        assert "FCW_CONTAINER_TOML" in result.output
        assert "FCWEOF" in result.output
        assert "csrun()" in result.output
        assert "csrun hostname" in result.output

    def test_calls_resync_for_container(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from fcw.cli import app
        from fcw.commands import container as container_mod
        import os

        self._mk_config(tmp_path)
        calls = []
        monkeypatch.setattr(
            container_mod, "_resync_container_patches",
            lambda cfg, name, system, account: calls.append((name, system, account)),
        )

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "run", "--container", "app", "--dry-run", "--", "echo hi",
            ])
        finally:
            os.chdir(old)

        assert result.exit_code == 0, result.output
        assert len(calls) == 1
        assert calls[0][0] == "app"

    def test_environment_path_injects_verbatim(self, tmp_path):
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        self._mk_config(tmp_path)
        env_toml = tmp_path / "custom.toml"
        sentinel = 'image = "/custom/path.sqsh"\n# SENTINEL-xyzzy\n'
        env_toml.write_text(sentinel)

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "run", "--environment", str(env_toml), "--dry-run", "--", "csrun hi",
            ])
        finally:
            os.chdir(old)

        assert result.exit_code == 0, result.output
        assert "SENTINEL-xyzzy" in result.output
        assert "/custom/path.sqsh" in result.output

    def test_missing_environment_path_errors(self, tmp_path):
        from typer.testing import CliRunner
        from fcw.cli import app
        import os

        self._mk_config(tmp_path)

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "run", "--environment", str(tmp_path / "nope.toml"),
                "--dry-run", "--", "echo hi",
            ])
        finally:
            os.chdir(old)

        assert result.exit_code == 1
        assert "not found" in result.output

    def test_dry_run_does_not_submit(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner
        from fcw.cli import app
        from fcw.commands import job as job_mod
        from fcw.commands import container as container_mod
        import os

        self._mk_config(tmp_path)
        monkeypatch.setattr(container_mod, "_resync_container_patches",
                            lambda *a, **kw: None)

        def _boom(*a, **kw):
            raise AssertionError("dry-run must not touch the client")
        monkeypatch.setattr(job_mod, "get_client", _boom)

        runner = CliRunner()
        old = os.getcwd()
        os.chdir(tmp_path)
        try:
            result = runner.invoke(app, [
                "job", "run", "--container", "app", "--dry-run", "--", "echo hi",
            ])
        finally:
            os.chdir(old)

        assert result.exit_code == 0, result.output


class _ScriptedClient:
    """Async client stub for _follow_stream tests.

    Each stat() call advances to the next (state, text) snapshot, simulating the
    remote file and job state evolving over poll cycles. tail() serves byte ranges
    of the current snapshot, mirroring the FirecREST ``tail -c +N`` semantics.
    """

    def __init__(self, snapshots):
        self._snaps = list(snapshots)
        self._i = -1
        self._state = snapshots[0][0]
        self._data = b""

    async def job_info(self, system_name, jobid):
        return [{"status": {"state": self._state}}]

    async def stat(self, system_name, path):
        if self._i + 1 < len(self._snaps):
            self._i += 1
        self._state, text = self._snaps[self._i]
        self._data = text.encode("utf-8")
        return {"size": len(self._data)}

    async def tail(self, system_name, path, num_bytes=None, num_lines=None,
                   exclude_beginning=False):
        if exclude_beginning and num_bytes is not None:
            text = self._data[num_bytes - 1:].decode("utf-8")  # tail -c +N
            start = num_bytes
        elif num_lines is not None:
            text = "\n".join(self._data.decode("utf-8").split("\n")[-num_lines:])
            start = 1
        else:
            text = self._data.decode("utf-8")
            start = 1
        # Mirror the real FirecREST tail payload shape (a dict, not a bare string).
        return {"content": text, "contentType": "bytes",
                "startPosition": start, "endPosition": -1}


class TestFollowStream:
    """Regression coverage for the offset-tracking tail -f follow loop."""

    async def test_appends_each_chunk_exactly_once(self, capsys):
        """An append-only growing file is streamed in full, with no drops or dups."""
        snaps = [
            ("RUNNING", "line1\nline2\n"),
            ("RUNNING", "line1\nline2\nline3\n"),
            ("RUNNING", "line1\nline2\nline3\nline4\nline5\nline6\n"),
            ("COMPLETED", "line1\nline2\nline3\nline4\nline5\nline6\n"),
        ]
        client = _ScriptedClient(snaps)
        await _follow_stream(client, "sys", "42", "/log.out",
                             lines=10, tail_only=False, interval=0)
        out = capsys.readouterr().out
        assert out == "line1\nline2\nline3\nline4\nline5\nline6\n"

    async def test_truncation_resets_offset(self, capsys):
        """A shrinking file (rotation/truncation) resets the offset and re-reads."""
        snaps = [
            ("RUNNING", "aaaa"),
            ("RUNNING", "bb"),      # truncated below current offset
            ("RUNNING", "bbcc"),
            ("COMPLETED", "bbcc"),
        ]
        client = _ScriptedClient(snaps)
        await _follow_stream(client, "sys", "42", "/log.out",
                             lines=10, tail_only=False, interval=0)
        out = capsys.readouterr().out
        assert out == "aaaabbcc"

    async def test_prefix_applied_per_line(self, capsys):
        """In `both` mode each emitted line carries the stream prefix."""
        snaps = [
            ("RUNNING", "a\nb\n"),
            ("COMPLETED", "a\nb\n"),
        ]
        client = _ScriptedClient(snaps)
        await _follow_stream(client, "sys", "42", "/log.err",
                             lines=10, tail_only=False, interval=0,
                             prefix="[stderr] ")
        out = capsys.readouterr().out
        assert out == "[stderr] a\n[stderr] b\n"

    async def test_stops_after_terminal_state(self, capsys):
        """The loop returns once the job is terminal (does not hang)."""
        snaps = [("COMPLETED", "done\n")]
        client = _ScriptedClient(snaps)
        await _follow_stream(client, "sys", "42", "/log.out",
                             lines=10, tail_only=False, interval=0)
        assert "done" in capsys.readouterr().out

    async def test_drains_output_flushed_after_terminal(self, capsys):
        """Bytes flushed after the job is already terminal are still emitted.

        Guards the post-terminal drain: SLURM marks a job COMPLETED before the
        output file finishes flushing, so breaking on the first terminal poll
        would drop the tail of the log. Here the file grows across snapshots
        that are already COMPLETED.
        """
        snaps = [
            ("RUNNING", "x\n"),
            ("COMPLETED", "x\n"),
            ("COMPLETED", "x\ny\n"),
            ("COMPLETED", "x\ny\nz\n"),
            ("COMPLETED", "x\ny\nz\n"),
        ]
        client = _ScriptedClient(snaps)
        await _follow_stream(client, "sys", "42", "/log.out",
                             lines=10, tail_only=False, interval=0)
        assert capsys.readouterr().out == "x\ny\nz\n"


class _WaitClient:
    """Sync client stub exposing only wait_for_job, for _report_final_state."""

    def __init__(self, state):
        self._state = state

    def wait_for_job(self, system_name, job_id):
        return [{"status": {"state": self._state}}]


class TestReportFinalState:
    """The shared wait/report path used by `job submit --wait` and `job run`."""

    def test_completed_does_not_raise(self):
        _report_final_state(_WaitClient("COMPLETED"), "sys", "42")

    def test_state_list_is_normalized(self):
        _report_final_state(_WaitClient(["COMPLETED"]), "sys", "42")

    @pytest.mark.parametrize("state", ["FAILED", "TIMEOUT", "OUT_OF_MEMORY"])
    def test_failure_exits_nonzero(self, state):
        import typer

        with pytest.raises(typer.Exit) as exc:
            _report_final_state(_WaitClient(state), "sys", "42")
        assert exc.value.exit_code == 1


class _MetadataClient:
    """Sync client stub exposing only job_metadata, for _job_stream_paths."""

    def __init__(self, metadata):
        self._metadata = metadata

    def job_metadata(self, system_name, jobid):
        return self._metadata


class TestJobStreamPaths:
    """Metadata-based stdout/stderr resolution shared by `job logs` and
    `job submit --follow`."""

    def test_resolves_and_expands_both_streams(self):
        client = _MetadataClient({
            "standardOutput": "/scratch/out-%j.log",
            "standardError": "/scratch/err-%j.log",
        })
        assert _job_stream_paths(client, "sys", "42") == (
            "/scratch/out-42.log", "/scratch/err-42.log",
        )

    def test_unwraps_list_metadata(self):
        client = _MetadataClient([{"standardOutput": "/o-%j.out"}])
        assert _job_stream_paths(client, "sys", "7") == ("/o-7.out", None)

    def test_empty_metadata_returns_none_pair(self):
        assert _job_stream_paths(_MetadataClient(None), "sys", "1") == (None, None)
        assert _job_stream_paths(_MetadataClient([]), "sys", "1") == (None, None)


class TestFollowStreams:
    """Thin orchestration over _follow_stream (which is covered separately)."""

    def test_single_stream_passes_through(self, capsys, monkeypatch):
        from fcw.commands import job as job_mod

        client = _ScriptedClient([("RUNNING", "a\n"), ("COMPLETED", "a\nb\n")])
        monkeypatch.setattr(job_mod, "get_async_client", lambda: client)
        _follow_streams("sys", "42", [("stdout", "/log.out", "out")],
                        tail=False, lines=10, interval=0)
        assert capsys.readouterr().out == "a\nb\n"

    def test_multi_stream_prefixes_lines(self, capsys, monkeypatch):
        from fcw.commands import job as job_mod

        # Stateless stub (the two concurrent followers share one client, so it
        # must not carry per-path state): a tiny already-complete 2-byte file.
        class _OneShot:
            async def job_info(self, system_name, jobid):
                return [{"status": {"state": "COMPLETED"}}]

            async def stat(self, system_name, path):
                return {"size": 2}

            async def tail(self, system_name, path, num_bytes=None,
                           num_lines=None, exclude_beginning=False):
                return {"content": "x\n", "startPosition": 1, "endPosition": -1}

        monkeypatch.setattr(job_mod, "get_async_client", _OneShot)
        _follow_streams("sys", "42",
                        [("stdout", "/o", "out"), ("stderr", "/e", "err")],
                        tail=False, lines=10, interval=0)
        out = capsys.readouterr().out
        assert "[stdout] x" in out
        assert "[stderr] x" in out


def _completed_job(job_id, elapsed):
    """A real-shaped (v2) completed job, modeled on an actual job_info dump."""
    return {
        "jobId": job_id,
        "name": "fcw-run",
        "status": {"state": "COMPLETED", "stateReason": "None", "exitCode": 0},
        "time": {"elapsed": elapsed, "start": 1780704385, "end": 1780704394,
                 "suspended": 0, "limit": 1800},
        "account": "csstaff",
        "allocationNodes": 1,
        "nodes": "nid007658",
        "partition": "debug",
        "user": "lukasd",
        "priority": 781149,
    }


def _table_cols(table):
    """Map a Rich table to {header: [cell, ...]}."""
    return {col.header: list(col.cells) for col in table.columns}


class TestFmtHelpers:
    def test_fmt_duration(self):
        assert _fmt_duration(9) == "0:00:09"
        assert _fmt_duration(1800) == "0:30:00"
        assert _fmt_duration(90061) == "1-1:01:01"
        assert _fmt_duration(None) == ""
        assert _fmt_duration("") == ""
        assert _fmt_duration(-5) == ""

    def test_fmt_epoch(self):
        assert _fmt_epoch(0) == ""
        assert _fmt_epoch(None) == ""
        out = _fmt_epoch(1780704385)
        assert out and "-" in out and ":" in out


class TestBuildJobsTable:
    def _jobs(self):
        return [_completed_job("2478513", 9), _completed_job("2478514", 12)]

    def test_default_columns_and_values(self):
        table, counts = _build_jobs_table(self._jobs())
        cols = _table_cols(table)
        assert cols["Job ID"] == ["2478513", "2478514"]
        assert cols["User"] == ["lukasd", "lukasd"]
        assert cols["Partition"] == ["debug", "debug"]
        assert cols["State"] == ["COMPLETED", "COMPLETED"]
        assert cols["Nodes"] == ["1", "1"]
        assert cols["Elapsed"] == ["0:00:09", "0:00:12"]
        # Reason (both "None") and Time Left (terminal) are empty -> hidden.
        assert "Reason" not in cols
        assert "Time Left" not in cols
        assert counts == Counter({"COMPLETED": 2})

    def test_long_adds_columns(self):
        table, _ = _build_jobs_table(self._jobs(), long=True)
        cols = _table_cols(table)
        assert cols["Account"] == ["csstaff", "csstaff"]
        assert cols["Nodelist"] == ["nid007658", "nid007658"]
        assert cols["Time Limit"] == ["0:30:00", "0:30:00"]
        assert all(cols["Start"]) and all(cols["End"])
        assert cols["Priority"] == ["781149", "781149"]

    def test_state_filter(self):
        _, counts = _build_jobs_table(self._jobs(), state="completed")
        assert counts == Counter({"COMPLETED": 2})
        table, counts = _build_jobs_table(self._jobs(), state="running")
        assert _table_cols(table)["Job ID"] == []
        assert counts == Counter()

    def test_partition_and_user_filters(self):
        assert _build_jobs_table(self._jobs(), partition="debug")[1].total() == 2
        assert _build_jobs_table(self._jobs(), partition="gpu")[1].total() == 0
        assert _build_jobs_table(self._jobs(), user="lukasd")[1].total() == 2
        assert _build_jobs_table(self._jobs(), user="other")[1].total() == 0

    def test_pending_shows_reason_blank_start(self):
        pending = _completed_job("99", 0)
        pending["status"] = {"state": "PENDING", "stateReason": "Resources"}
        pending["time"] = {"elapsed": 0, "start": 0, "end": 0, "limit": 1800}
        table, _ = _build_jobs_table([pending], long=True)
        cols = _table_cols(table)
        assert cols["Reason"] == ["Resources"]   # shown now (non-empty)
        assert "Start" not in cols               # epoch 0 -> blank -> hidden

    def test_running_shows_time_left(self):
        running = _completed_job("100", 600)
        running["status"] = {"state": "RUNNING", "stateReason": None}
        running["time"] = {"elapsed": 600, "start": 1780704385, "end": 0, "limit": 1800}
        cols = _table_cols(_build_jobs_table([running])[0])
        assert cols["Time Left"] == ["0:20:00"]   # 1800 - 600


class TestJobListAllUsersError:
    """The all-users path (allusers=True → cluster-wide sacct) fails gracefully."""

    def _runner(self, monkeypatch, boom_on_allusers):
        from fcw.commands import job as job_mod

        class _Stub:
            def job_info(self, system_name, **kw):
                if boom_on_allusers and kw.get("allusers"):
                    raise RuntimeError("last request: 500 Error executing Slurm command")
                return []

        monkeypatch.setattr(job_mod, "get_system", lambda *a, **k: "sys")
        monkeypatch.setattr(job_mod, "get_client", lambda: _Stub())
        from typer.testing import CliRunner
        return CliRunner()

    def test_all_users_failure_is_friendly(self, monkeypatch):
        from fcw.cli import app

        runner = self._runner(monkeypatch, boom_on_allusers=True)
        result = runner.invoke(app, ["job", "list", "--all-users"])
        assert result.exit_code != 0
        assert "all users" in result.output.lower()
        assert "500" in result.output  # underlying error surfaced, not swallowed

    def test_plain_list_unaffected(self, monkeypatch):
        from fcw.cli import app

        runner = self._runner(monkeypatch, boom_on_allusers=True)
        result = runner.invoke(app, ["job", "list"])
        assert result.exit_code == 0

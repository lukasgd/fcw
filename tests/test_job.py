"""Tests for SBATCH parsing, env injection, script manipulation, and container TOML inlining."""

import pytest

from fcw.commands.job import (
    _apply_sbatch_overrides,
    _build_container_toml,
    _follow_stream,
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
        assert opts == {"time": "1:00:00"}

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
                "job", "run", "-c", "app", "-e", str(env_toml),
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
                "job", "run", "-c", "app", "--dry-run", "--", "csrun hostname",
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
                "job", "run", "-c", "app", "--dry-run", "--", "echo hi",
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
                "job", "run", "-e", str(env_toml), "--dry-run", "--", "csrun hi",
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
                "job", "run", "-e", str(tmp_path / "nope.toml"),
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
                "job", "run", "-c", "app", "--dry-run", "--", "echo hi",
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

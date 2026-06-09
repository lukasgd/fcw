"""End-to-end workflow tests for the basic example.

Tests are ordered to form a complete workflow: config validation,
container build/deploy, data transfer, job submission, and cleanup.
Tests within each class run in declaration order and may depend on
prior state (e.g., container must be built before jobs can run).

Requires: --run-e2e flag or FCW_E2E=1 environment variable.
Runs with: --example basic (default).
"""

import os
import re

import pytest
from helpers import (
    assert_container_deploy,
    assert_data_download,
    assert_data_upload,
    assert_ok,
    assert_remote_ls_contains,
    assert_remote_ls_not_contains,
    assert_sqsh_exists,
    extract_job_id,
    invoke,
)

from fcw.cli import app

pytestmark = [pytest.mark.e2e, pytest.mark.example("basic")]


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_config_validate(self, runner):
        """fcw config validate succeeds with valid credentials."""
        result = runner.invoke(app, ["config", "validate"])
        assert_ok(result)

    def test_config_show(self, runner):
        """fcw config show displays resolved config."""
        result = runner.invoke(app, ["config", "show"])
        assert_ok(result)
        assert "remote:" in result.output or "test-fcw" in result.output


# ---------------------------------------------------------------------------
# 2. Remote Setup
# ---------------------------------------------------------------------------

class TestRemoteSetup:
    def test_setup_remote_dirs(self, client, system, account, remote_workdir):
        """Create remote directory structure."""
        for subdir in ["data/raw", "data/processed", "outputs", "logs", "env", "ce-images"]:  # FIXME: none of these should be created explicitly, but as part of e2e workflow
            try:
                client.mkdir(
                    system_name=system,
                    path=f"{remote_workdir}/{subdir}",
                    create_parents=True,
                )
            except Exception:
                pass  # May already exist

    def test_upload_env(self, runner, timed_step):  # FIXME: env is no longer a data dir (Dockerfiles should be uploaded as part of container build, container TOML as part of deploy/patch/rebuild and up-to-date checked in job submit/run). Data transfer, which is tested here, is already covered by seprarate tests.
        """Upload env/ directory."""
        invoke(runner, ["data", "upload", "env"], "upload-env", timed_step)

    def test_upload_slurm(self, runner, timed_step):
        """Upload slurm/ directory."""
        invoke(runner, ["data", "upload", "slurm"], "upload-slurm", timed_step)


# ---------------------------------------------------------------------------
# 3. Container Build (multi-stage) + Deploy (single-stage)
# ---------------------------------------------------------------------------

class TestContainerBuildDeploy:
    """Multi-stage container workflow: build locally, push, build-remote."""

    def test_container_build_local(self, runner, timed_step, remote_platform):  # TODO: uses low-level CLI - should probably have a test that uses container config as well and make sure it builds local_stage
        """Build download stage locally."""
        cmd = [
            "container", "build", "--stage", "download",
            "-t", "ubuntu-fcw-basic:24.04-download", "-f", "env/Dockerfile.app",
        ]
        if remote_platform:
            cmd.extend(["--platform", remote_platform])
        cmd.extend(["--context", "."])
        invoke(runner, cmd, "container-build-local", timed_step)

    def test_container_build_save(self, runner, remote_platform):
        """Build and save image to tar file (--save)."""
        cmd = [
            "container", "build", "--stage", "download",
            "-t", "ubuntu-fcw-basic:24.04-download",
            "-f", "env/Dockerfile.app",
            "--save", "test-image.tar",
        ]
        if remote_platform:
            cmd.extend(["--platform", remote_platform])
        cmd.extend(["--context", "."])
        result = runner.invoke(app, cmd)
        assert_ok(result)
        assert os.path.exists("test-image.tar")
        os.unlink("test-image.tar")

    def test_container_push(self, runner, timed_step, remote_platform):  # TODO: uses low-level CLI - should probably have a test that uses container config as well
        """Push image to remote."""
        cmd = ["container", "push"]
        if remote_platform:
            cmd.extend(["--platform", remote_platform])
        cmd.append("ubuntu-fcw-basic:24.04-download")
        invoke(runner, cmd, "container-push", timed_step)

    def test_container_build_remote(self, runner, timed_step):  # TODO: uses low-level CLI - should probably have a test that uses container config as well
        """Build offline stage on cluster + enroot import."""
        invoke(runner, [
            "container", "build-remote", "ubuntu-fcw-basic:24.04-download",
            "-f", "env/Dockerfile.app", "-t", "ubuntu-fcw-basic:24.04",
            "--stage", "build-offline", "--enroot", "--wait",
        ], "container-build-remote", timed_step)

    def test_verify_sqsh(self, runner):
        """Verify squashfs image exists on remote."""
        assert_sqsh_exists(runner, "ce-images", "ubuntu-fcw-basic+24.04.sqsh")

    def test_container_list_local(self, runner):
        """List local container images."""
        result = runner.invoke(app, ["container", "list"])
        assert_ok(result)


class TestContainerDeploy:
    """Single-command deploy workflow: build + push + import in one step."""

    def test_container_deploy(self, runner, timed_step, remote_platform):
        """Deploy aux container (build+push+import)."""
        assert_container_deploy(runner, timed_step, "aux", platform=remote_platform)

    def test_verify_deploy(self, runner):
        """Verify deployed sqsh exists on remote."""
        assert_sqsh_exists(runner, "ce-images", "fcw-aux+latest.sqsh")

    def test_container_list_remote(self, runner):
        """List remote container images (should show both sqsh files)."""
        result = runner.invoke(app, ["container", "list", "--remote"])
        assert_ok(result)


# ---------------------------------------------------------------------------
# 4. Container Iteration (extract + patch)
# ---------------------------------------------------------------------------

class TestContainerIterate:
    """Code iteration workflow: extract from container, patch with bind-mount."""

    def test_job_run_container(self, runner, timed_step):
        """`fcw job run --container app -- 'csrun ...'` should inject the TOML + csrun
        shorthand and execute the command inside the container."""
        import uuid
        sentinel = uuid.uuid4().hex
        with timed_step("job-run-container"):
            result = runner.invoke(app, [
                "job", "run", "--remote-script",
                "--container", "app",
                "--", f"csrun echo RUN-SENTINEL-{sentinel}",
            ])
        assert_ok(result)
        job_id = result.stdout.strip().split("\n")[-1].split(" ")[-1]

        wait = runner.invoke(app, ["job", "wait", job_id])
        assert_ok(wait)
        logs = runner.invoke(app, ["job", "logs", job_id])
        assert_ok(logs)
        assert f"RUN-SENTINEL-{sentinel}" in logs.output

    def test_container_extract(self, runner, timed_step):
        """Extract files from aux container stage to local directory.

        Also verifies the sidecar `.meta.json` is written next to the dump.
        """
        invoke(runner, [
            "container", "extract",
            "aux", "/workspace/aux",
            "extracted-code",
            "--wait",
        ], "container-extract", timed_step)
        assert os.path.isdir("extracted-code")
        # Contents land de-nested: /workspace/aux/test.txt -> extracted-code/test.txt,
        # not extracted-code/aux/test.txt (the basename wrapper must be gone, so the
        # dump root maps directly onto the container path for patch/rebuild).
        assert os.path.isfile("extracted-code/test.txt")
        assert not os.path.exists("extracted-code/aux")
        sidecar = "extracted-code.meta.json"
        assert os.path.exists(sidecar)
        import json
        meta = json.loads(open(sidecar).read())
        assert meta["container_path"] == "/workspace/aux"
        assert "stage" in meta

    def test_container_patch(self, runner, timed_step):
        """Upload patched code and add bind-mount entry to container TOML."""
        if os.path.isdir("extracted-code"):
            path_arg = "extracted-code"
        else:
            path_arg = "data/raw:/workspace"

        invoke(runner, [
            "container", "patch",
            "--container", "app",
            path_arg,
        ], "container-patch", timed_step)
        content = open("env/container.toml").read()
        assert ".patches/" in content

    def test_auto_resync_on_run(self, runner, timed_step):  # TODO: could we have both a test with job run and one with job submit + wait, to verify auto-resync works in both cases and since they overwrite the same file a re-resync is covered as well?
        """Edit the local dump *without* re-running `patch`, then run a job.

        The auto-resync hook in `fcw job run` should incrementally upload
        the new file to the remote .patches/ dir so the next job sees it.
        """
        if not os.path.isdir("extracted-code"):
            pytest.skip("requires test_container_extract to have produced extracted-code")

        for round in range(2):
            import uuid
            sentinel_value = uuid.uuid4().hex
            sentinel = os.path.join("extracted-code", "fcw-resync-sentinel.txt")
            with open(sentinel, "w") as f:
                f.write(sentinel_value)

            with timed_step("job-run-auto-resync"):
                result = runner.invoke(app, [
                    "job", "run", "--remote-script",
                    "--container", "app",
                    "--", "csrun cat /workspace/aux/fcw-resync-sentinel.txt",
                ])
            assert_ok(result)

            ls = runner.invoke(app, ["data", "ls", ".patches/extracted-code", "-R"])
            assert_ok(ls)
            assert "fcw-resync-sentinel.txt" in ls.output

            job_id = result.stdout.strip().split("\n")[-1].split(" ")[-1]

            wait = runner.invoke(app, ["job", "wait", job_id])
            assert_ok(wait)
            logs = runner.invoke(app, ["job", "logs", job_id])
            assert_ok(logs)

            assert sentinel_value in logs.output, f"Sentinel value not found in logs: {logs.output}"


# ---------------------------------------------------------------------------
# 4b. Container Rebuild (bake patches into new image)
# ---------------------------------------------------------------------------

class TestContainerRebuild:
    """Rebuild workflow: patch the app container's TOML, then rebuild."""

    def test_patch_app_container(self, runner):
        """Patch the app container so its TOML has .patches/ mounts for rebuild."""
        patch_source = "extracted-code" if os.path.isdir("extracted-code") else "data/raw"
        result = runner.invoke(app, [
            "container", "patch",
            "--container", "app",
            f"{patch_source}:/workspace/patched",
        ])
        assert_ok(result)
        content = open("env/container.toml").read()
        assert ".patches/" in content

    def test_container_rebuild_dry_run(self, runner):
        """Dry run should print the SLURM script without submitting."""
        result = runner.invoke(app, [
            "container", "rebuild", "app",
            "--tag", "ubuntu-fcw-basic:v2",
            "--dry-run",
        ])
        assert_ok(result)
        assert "fcw-container-rebuild" in result.output
        assert "podman" in result.output

    def test_container_rebuild(self, runner, timed_step):
        """Full rebuild: bake patches, create new TOML and config entry."""
        invoke(runner, [
            "container", "rebuild", "app",
            "--tag", "ubuntu-fcw-basic:v2",
            "--enroot", "--wait",
        ], "container-rebuild", timed_step)
        from fcw.core.config import load_config
        config = load_config("fcw.yaml")
        assert "app-v2" in config.containers
        assert config.containers["app-v2"].tag == "ubuntu-fcw-basic:v2"

    def test_restore_app_toml(self, runner):
        """Remove .patches/ mounts from the original app TOML.

        The rebuild creates a new container entry with a clean TOML, but the
        original env/container.toml still has the .patches/ bind-mounts.
        Strip them so subsequent job tests (which use container: app) don't
        fail trying to mount non-existent patch directories.
        """
        from fcw.commands.container import _create_rebuilt_toml
        toml_path = "env/container.toml"
        _create_rebuilt_toml(toml_path, toml_path)
        content = open(toml_path).read()
        assert ".patches/" not in content


# ---------------------------------------------------------------------------
# 5. Data Transfer
# ---------------------------------------------------------------------------

class TestDataUpload:
    def test_upload_data(self, runner, timed_step):
        """Upload raw data."""
        assert_data_upload(runner, timed_step, "data/raw")

    def test_verify_data(self, runner):
        """Verify data on remote."""
        assert_remote_ls_contains(runner, "data/raw", "test.txt")

    def test_upload_incremental(self, runner, timed_step):
        """Re-upload with --incremental (should skip unchanged files)."""
        assert_data_upload(runner, timed_step, "data/raw", incremental=True)

    def test_data_status(self, runner):
        """Show sync status for configured directories."""
        result = runner.invoke(app, ["data", "status"])
        assert_ok(result)


# ---------------------------------------------------------------------------
# 6. Job Submission
# ---------------------------------------------------------------------------

class TestJobSubmission:
    def test_submit_preprocess(self, submit_job):
        """Submit and wait for preprocess job."""
        submit_job("preprocess")

    def test_verify_preprocess(self, runner):
        """Verify preprocess output exists."""
        assert_remote_ls_contains(runner, "data/processed", "preprocessed_files.txt")

    def test_submit_train(self, submit_job):
        """Submit and wait for train job."""
        submit_job("train")

    def test_verify_train(self, runner):
        """Verify train output files exist."""
        assert_remote_ls_contains(runner, "outputs", "train_output_")

    def test_submit_evaluate(self, submit_job):
        """Submit and wait for evaluate job."""
        submit_job("evaluate")

    def test_verify_evaluate(self, runner):
        """Verify evaluate output exists."""
        assert_remote_ls_contains(runner, "outputs", "eval_summary_")

    def test_submit_with_env_override(self, runner):
        """Submit preprocess with --set to redirect output directory."""
        result = runner.invoke(app, [
            "job", "submit", "--remote-script", "--wait",
            "--set", "DATA_OUT=outputs",
            "--", "preprocess",
        ])
        assert_ok(result)
        assert_remote_ls_contains(runner, "outputs", "preprocessed_files.txt")


# ---------------------------------------------------------------------------
# 7. Job Management
# ---------------------------------------------------------------------------

class TestJobManagement:
    """Job lifecycle: list, run, status, logs, cancel."""

    def test_job_list(self, runner):
        """List recent jobs on the cluster (default + --long), with summary."""
        result = runner.invoke(app, ["job", "list"])
        assert_ok(result)
        # Assert on the plainly-printed summary line, not column headers: Rich
        # truncates wide-table headers to ~80 cols under CliRunner (e.g.
        # "Partition" -> "Partiti…"), so header text is not a reliable signal.
        assert "Total:" in result.output

        result = runner.invoke(app, ["job", "list", "--long"])
        assert_ok(result)
        # NB: --all-users is intentionally not exercised here — it runs a
        # cluster-wide sacct that is slow / 500s on busy systems (handled
        # gracefully by the command, covered hermetically instead).

    def test_job_run_and_wait(self, runner, shared_state, timed_step):
        """Run ad-hoc command, then wait for it with 'job wait'."""
        with timed_step("job-run-and-wait"):
            result = runner.invoke(app, [
                "job", "run", "--remote-script",
                "--", "echo hello-from-fcw-run",
            ])
            assert_ok(result)
            job_id = extract_job_id(result.output)
            assert job_id, f"Could not extract job ID from: {result.output}"
            shared_state["run_job_id"] = job_id

            result = runner.invoke(app, ["job", "wait", job_id])
        assert_ok(result)

    def test_job_status(self, runner, shared_state):
        """Check status of a completed job."""
        job_id = shared_state.get("run_job_id")
        if not job_id:
            pytest.skip("No job ID from previous test")
        result = runner.invoke(app, ["job", "status", job_id])
        assert_ok(result)

    def test_job_logs(self, runner, shared_state):
        """View stdout logs of a completed job."""
        job_id = shared_state.get("run_job_id")
        if not job_id:
            pytest.skip("No job ID from previous test")
        result = runner.invoke(app, ["job", "logs", job_id])
        if result.exit_code != 0 and "metadata" not in result.output.lower():
            assert_ok(result)

    def test_job_cancel(self, runner):
        """Submit a long-running job and cancel it."""
        result = runner.invoke(app, [
            "job", "run", "--remote-script",
            "--", "sleep 600",
        ])
        assert_ok(result)
        job_id = extract_job_id(result.output)
        assert job_id, f"Could not extract job ID from: {result.output}"

        result = runner.invoke(app, ["job", "cancel", job_id])
        assert_ok(result)


# ---------------------------------------------------------------------------
# 7.4 Job run: --wait and --follow
# ---------------------------------------------------------------------------

class TestJobRunWaitFollow:
    """`fcw job run` blocking modes: --wait (report on completion) and --follow
    (stream output live until the job finishes)."""

    def test_run_wait(self, runner, timed_step):
        """--wait blocks until completion and reports the final state."""
        with timed_step("job-run-wait"):
            result = runner.invoke(app, [
                "job", "run", "--remote-script", "--wait", "--", "echo run-wait-ok",
            ])
        assert_ok(result)
        assert "completed" in result.output.lower()

    def test_run_follow(self, runner, timed_step):
        """--follow streams the job's output live, then exits on completion."""
        with timed_step("job-run-follow"):
            result = runner.invoke(app, [
                "job", "run", "--remote-script", "--follow",
                "--", "echo run-A; sleep 1; echo run-B",
            ])
        assert_ok(result)
        assert "run-A" in result.output
        assert "run-B" in result.output
        assert "completed" in result.output.lower()

    def test_submit_follow(self, runner, timed_step, tmp_path):
        """`job submit --follow` resolves the output path from job metadata,
        streams the script's output live, then exits on completion."""
        script = tmp_path / "submit_follow.sh"
        script.write_text(
            "#!/bin/bash -l\n#SBATCH --time 00:05:00\n"
            "echo submit-A; sleep 1; echo submit-B\n"
        )
        with timed_step("job-submit-follow"):
            result = runner.invoke(app, [
                "job", "submit", "--remote-script", "--follow",
                "--", str(script),
            ])
        assert_ok(result)
        assert "submit-A" in result.output
        assert "submit-B" in result.output
        assert "completed" in result.output.lower()


# ---------------------------------------------------------------------------
# 7.5 Job logs: stream selection + live follow
# ---------------------------------------------------------------------------

class TestJobLogsStreaming:
    """Real-engine coverage for `job logs` stream selection and tail -f follow.

    The unit tests (tests/test_job.py::TestFollowStream) exercise the offset
    arithmetic against a stub; these confirm the assumptions that stub can't:
    that FirecREST `tail(..., exclude_beginning=True)` is `tail -c +N`, that the
    tail payload is a dict, and that job metadata exposes a distinct standardError
    path. Both tests share one job (one scheduling wait); the stream-selection
    test waits for completion itself, so neither depends on the other's order.
    """

    @pytest.fixture(scope="class")
    def streaming_job(self):
        """Submit one job that writes distinct stdout/stderr markers, then emits
        a few lines over ~4s. Explicit --error keeps the two streams in separate
        files (plain `fcw job run` only sets --output, combining them)."""
        from typer.testing import CliRunner

        r = CliRunner()
        result = r.invoke(app, [
            "job", "run", "--remote-script", "--error", "fcw-run-%j.err", "--",
            "echo OUT_MARKER; echo ERR_MARKER >&2; "
            "for i in 1 2 3 4; do echo follow-line-$i; sleep 1; done",
        ])
        assert_ok(result)
        job_id = extract_job_id(result.output)
        assert job_id, f"Could not extract job ID from: {result.output}"
        return job_id

    def test_logs_follow_incremental(self, runner, timed_step, streaming_job):
        """--follow streams a growing file in full and in order, then exits.

        Following must reproduce every line exactly once and in order — the
        regression guard for the offset/tail-payload handling, validated against
        the real tail/stat endpoints. Drains the job to completion.
        """
        with timed_step("job-logs-follow"):
            result = runner.invoke(app, ["job", "logs", streaming_job, "--follow"])
        assert_ok(result)

        seen = re.findall(r"follow-line-\d+", result.output)
        assert seen == [f"follow-line-{i}" for i in range(1, 5)], result.output

    def test_logs_stream_selection(self, runner, streaming_job):
        """stdout/stderr/both select the right stream.

        Self-sufficient: waits for the job to finish first (a no-op when the
        follow test already drained it), so it also passes run in isolation.
        """
        result = runner.invoke(app, ["job", "wait", streaming_job])
        assert_ok(result)

        # stdout only
        result = runner.invoke(app, ["job", "logs", streaming_job])
        assert_ok(result)
        assert "OUT_MARKER" in result.output
        assert "ERR_MARKER" not in result.output

        # stderr only
        result = runner.invoke(app, ["job", "logs", streaming_job, "--stream", "stderr"])
        assert_ok(result)
        assert "ERR_MARKER" in result.output

        # both
        result = runner.invoke(app, ["job", "logs", streaming_job, "--stream", "both"])
        assert_ok(result)
        assert "OUT_MARKER" in result.output
        assert "ERR_MARKER" in result.output


# ---------------------------------------------------------------------------
# 8. Data Download
# ---------------------------------------------------------------------------

class TestDataDownload:
    def test_download_outputs(self, runner, timed_step):
        """Download outputs directory."""
        assert_data_download(runner, timed_step, "outputs")
        # Contents land directly in outputs/, not a nested outputs/outputs/.
        assert os.path.isdir("outputs")
        assert not os.path.isdir(os.path.join("outputs", "outputs"))

    def test_download_incremental(self, runner, timed_step):
        """Re-download with --incremental (should skip unchanged files)."""
        assert_data_download(runner, timed_step, "outputs", incremental=True)
        # The incremental path must de-nest too (regression guard for bug #4).
        assert not os.path.isdir(os.path.join("outputs", "outputs"))


# ---------------------------------------------------------------------------
# 8.5 Symlink + nested-file upload (bug #3)
# ---------------------------------------------------------------------------

class TestSymlinkAndNestedUpload:
    """Symlinks are dereferenced only with -L; nested single-file uploads
    create their remote parents. Fixtures are built at runtime (a committed
    symlink wouldn't survive the example copytree)."""

    def test_upload_symlinked_dir(self, runner, timed_step, tmp_path):
        target = tmp_path / "real_target"
        os.makedirs(target, exist_ok=True)
        with open(os.path.join(target, "inner.txt"), "w") as f:
            f.write("payload")
        link = os.path.join("data", "raw", "linked")
        if os.path.islink(link) or os.path.exists(link):
            os.remove(link)
        os.symlink(str(target), link)
        try:
            invoke(runner, ["data", "upload", "--follow-symlinks", "data/raw"],
                   "upload-symlinked", timed_step)
            assert_remote_ls_contains(runner, "data/raw/linked", "inner.txt")
        finally:
            os.remove(link)

    def test_upload_nested_file_creates_parents(self, runner, timed_step):
        nested = os.path.join("data", "raw", "newsub", "deep")
        os.makedirs(nested, exist_ok=True)
        with open(os.path.join(nested, "file.txt"), "w") as f:
            f.write("x")
        invoke(runner, ["data", "upload", os.path.join(nested, "file.txt")],
               "upload-nested-file", timed_step)
        assert_remote_ls_contains(runner, "data/raw/newsub/deep", "file.txt")


# ---------------------------------------------------------------------------
# 9. Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_data_rm(self, runner):
        """Remove a remote directory."""
        result = runner.invoke(app, ["data", "rm", "--force", "data/processed"])
        assert_ok(result)

    def test_verify_rm(self, runner):
        """Verify the directory was removed."""
        assert_remote_ls_not_contains(runner, "data/processed", "preprocessed_files.txt")

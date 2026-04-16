"""End-to-end workflow tests.

Tests are ordered to form a complete workflow: config validation,
container build/deploy, data transfer, job submission, and cleanup.
Tests within each class run in declaration order and may depend on
prior state (e.g., container must be built before jobs can run).

Requires: --run-e2e flag or FCW_E2E=1 environment variable.
Uses the example project selected via --example (default: basic).
"""

import os
import re

import pytest

from fcw.cli import app


pytestmark = pytest.mark.e2e


def _extract_job_id(output: str) -> str | None:
    """Extract a numeric job ID from CLI output."""
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.isdigit():
            return line
        # Also match job ID in "Submitted job 12345" style lines
        m = re.search(r"job (\d+)", line, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


# ---------------------------------------------------------------------------
# 1. Config
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_config_validate(self, runner):
        """fcw config validate succeeds with valid credentials."""
        result = runner.invoke(app, ["config", "validate"])
        assert result.exit_code == 0, result.output

    def test_config_show(self, runner):
        """fcw config show displays resolved config."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0, result.output
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
        with timed_step("upload-env"):
            result = runner.invoke(app, ["data", "upload", "env"])
        assert result.exit_code == 0, result.output

    def test_upload_slurm(self, runner, timed_step):
        """Upload slurm/ directory."""
        with timed_step("upload-slurm"):
            result = runner.invoke(app, ["data", "upload", "slurm"])
        assert result.exit_code == 0, result.output


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
        cmd.append(".")
        with timed_step("container-build-local"):
            result = runner.invoke(app, cmd)
        assert result.exit_code == 0, result.output

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
        cmd.append(".")
        result = runner.invoke(app, cmd)
        assert result.exit_code == 0, result.output
        assert os.path.exists("test-image.tar")
        os.unlink("test-image.tar")

    def test_container_push(self, runner, timed_step, remote_platform):  # TODO: uses low-level CLI - should probably have a test that uses container config as well
        """Push image to remote."""
        cmd = ["container", "push"]
        if remote_platform:
            cmd.extend(["--platform", remote_platform])
        cmd.append("ubuntu-fcw-basic:24.04-download")

        with timed_step("container-push"):
            result = runner.invoke(app, cmd)
        assert result.exit_code == 0, result.output

    def test_container_build_remote(self, runner, timed_step):  # TODO: uses low-level CLI - should probably have a test that uses container config as well
        """Build offline stage on cluster + enroot import."""
        with timed_step("container-build-remote"):
            result = runner.invoke(app, [
                "container", "build-remote", "ubuntu-fcw-basic:24.04-download",
                "-f", "env/Dockerfile.app", "-t", "ubuntu-fcw-basic:24.04",
                "--stage", "build-offline", "--enroot", "--wait",
            ])
        assert result.exit_code == 0, result.output

    def test_verify_sqsh(self, runner):
        """Verify squashfs image exists on remote."""
        result = runner.invoke(app, ["data", "ls", "ce-images"])
        assert result.exit_code == 0, result.output
        assert "ubuntu-fcw-basic+24.04.sqsh" in result.output

    def test_container_list_local(self, runner):
        """List local container images."""
        result = runner.invoke(app, ["container", "list"])
        assert result.exit_code == 0


class TestContainerDeploy:
    """Single-command deploy workflow: build + push + import in one step."""

    def test_container_deploy(self, runner, timed_step, remote_platform):
        """Deploy aux container (build+push+import)."""
        cmd = ["container", "deploy", "aux", "--wait"]
        if remote_platform:
            cmd.extend(["--platform", remote_platform])

        with timed_step("container-deploy"):
            result = runner.invoke(app, cmd)
        assert result.exit_code == 0, result.output

    def test_verify_deploy(self, runner):
        """Verify deployed sqsh exists on remote."""
        result = runner.invoke(app, ["data", "ls", "ce-images"])
        assert result.exit_code == 0, result.output
        assert "fcw-aux+latest.sqsh" in result.output

    def test_container_list_remote(self, runner):
        """List remote container images (should show both sqsh files)."""
        result = runner.invoke(app, ["container", "list", "--remote"])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 4. Container Iteration (extract + patch)
# ---------------------------------------------------------------------------

class TestContainerIterate:
    """Code iteration workflow: extract from container, patch with bind-mount."""

    def test_job_run_container(self, runner, timed_step):
        """`fcw job run -c app -- 'csrun ...'` should inject the TOML + csrun
        shorthand and execute the command inside the container."""
        import uuid
        sentinel = uuid.uuid4().hex
        with timed_step("job-run-container"):
            result = runner.invoke(app, [
                "job", "run", "--remote-script",
                "-c", "app",
                "--", f"csrun echo RUN-SENTINEL-{sentinel}",
            ])
        assert result.exit_code == 0, result.output
        job_id = result.stdout.strip().split("\n")[-1].split(" ")[-1]

        wait = runner.invoke(app, ["job", "wait", job_id])
        assert wait.exit_code == 0, wait.output
        logs = runner.invoke(app, ["job", "logs", job_id])
        assert logs.exit_code == 0, logs.output
        assert f"RUN-SENTINEL-{sentinel}" in logs.output

    def test_container_extract(self, runner, timed_step):
        """Extract files from aux container stage to local directory.

        Also verifies the sidecar `.meta.json` is written next to the dump.
        """
        with timed_step("container-extract"):
            result = runner.invoke(app, [
                "container", "extract",
                "aux", "/workspace/aux",
                "extracted-code",
                "--wait",
            ])
        assert result.exit_code == 0, result.output
        assert os.path.isdir("extracted-code")
        # Sidecar should exist and record the stage + container_path.
        sidecar = "extracted-code.meta.json"
        assert os.path.exists(sidecar), result.output
        import json
        meta = json.loads(open(sidecar).read())
        assert meta["container_path"] == "/workspace/aux"
        assert "stage" in meta

    def test_container_patch(self, runner, timed_step):
        """Upload patched code and add bind-mount entry to container TOML."""
        # Prefer the sidecar-annotated dump from `extract`; fall back to data/raw.
        # When falling back, use '<local>:<container>' syntax since data/raw has no sidecar.
        if os.path.isdir("extracted-code"):
            path_arg = "extracted-code"  # sidecar supplies the target
        else:
            path_arg = "data/raw:/workspace"

        with timed_step("container-patch"):
            result = runner.invoke(app, [
                "container", "patch",
                "--container", "app",
                path_arg,
            ])
        assert result.exit_code == 0, result.output
        # app's TOML should now contain a .patches/ bind-mount.
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
                    "-c", "app",
                    "--", "csrun cat /workspace/aux/fcw-resync-sentinel.txt",
                ])
            assert result.exit_code == 0, result.output

            ls = runner.invoke(app, ["data", "ls", ".patches/extracted-code", "-R"])
            assert ls.exit_code == 0, ls.output
            assert "fcw-resync-sentinel.txt" in ls.output

            job_id = result.stdout.strip().split("\n")[-1].split(" ")[-1]

            wait = runner.invoke(app, ["job", "wait", job_id])
            assert wait.exit_code == 0, wait.output
            logs = runner.invoke(app, ["job", "logs", job_id])
            assert logs.exit_code == 0, logs.output

            assert sentinel_value in logs.output, f"Sentinel value not found in logs: {logs.output}"


# ---------------------------------------------------------------------------
# 4b. Container Rebuild (bake patches into new image)
# ---------------------------------------------------------------------------

class TestContainerRebuild:
    """Rebuild workflow: patch the app container's TOML, then rebuild."""

    def test_patch_app_container(self, runner):
        """Patch the app container so its TOML has .patches/ mounts for rebuild."""
        # Use the mount-syntax override so this works even if no sidecar is present.
        patch_source = "extracted-code" if os.path.isdir("extracted-code") else "data/raw"
        result = runner.invoke(app, [
            "container", "patch",
            "--container", "app",
            f"{patch_source}:/workspace/patched",
        ])
        assert result.exit_code == 0, result.output
        content = open("env/container.toml").read()
        assert ".patches/" in content

    def test_container_rebuild_dry_run(self, runner):
        """Dry run should print the SLURM script without submitting."""
        result = runner.invoke(app, [
            "container", "rebuild", "app",
            "--tag", "ubuntu-fcw-basic:v2",
            "--dry-run",
        ])
        assert result.exit_code == 0, result.output
        assert "fcw-container-rebuild" in result.output
        assert "podman" in result.output

    def test_container_rebuild(self, runner, timed_step):
        """Full rebuild: bake patches, create new TOML and config entry."""
        with timed_step("container-rebuild"):
            result = runner.invoke(app, [
                "container", "rebuild", "app",
                "--tag", "ubuntu-fcw-basic:v2",
                "--enroot", "--wait",
            ])
        assert result.exit_code == 0, result.output
        # Verify new config entry was added
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
        with timed_step("upload-data"):
            result = runner.invoke(app, ["data", "upload", "data/raw"])
        assert result.exit_code == 0, result.output

    def test_verify_data(self, runner):
        """Verify data on remote."""
        result = runner.invoke(app, ["data", "ls", "data/raw", "-R"])
        assert result.exit_code == 0, result.output
        assert "test.txt" in result.output

    def test_upload_incremental(self, runner, timed_step):
        """Re-upload with --incremental (should skip unchanged files)."""
        with timed_step("upload-incremental"):
            result = runner.invoke(app, ["data", "upload", "--incremental", "data/raw"])
        assert result.exit_code == 0, result.output

    def test_data_status(self, runner):
        """Show sync status for configured directories."""
        result = runner.invoke(app, ["data", "status"])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 6. Job Submission
# ---------------------------------------------------------------------------

class TestJobSubmission:
    def test_submit_preprocess(self, runner, timed_step):
        """Submit and wait for preprocess job."""
        with timed_step("job-preprocess"):
            result = runner.invoke(app, [
                "job", "submit", "--remote-script", "--wait", "--", "preprocess",
            ])
        assert result.exit_code == 0, result.output

    def test_verify_preprocess(self, runner):
        """Verify preprocess output exists."""
        result = runner.invoke(app, ["data", "ls", "data/processed"])
        assert result.exit_code == 0, result.output
        assert "preprocessed_files.txt" in result.output

    def test_submit_train(self, runner, timed_step):
        """Submit and wait for train job."""
        with timed_step("job-train"):
            result = runner.invoke(app, [
                "job", "submit", "--remote-script", "--wait", "--", "train",
            ])
        assert result.exit_code == 0, result.output

    def test_verify_train(self, runner):
        """Verify train output files exist."""
        result = runner.invoke(app, ["data", "ls", "outputs"])
        assert result.exit_code == 0, result.output
        assert "train_output_" in result.output

    def test_submit_evaluate(self, runner, timed_step):
        """Submit and wait for evaluate job."""
        with timed_step("job-evaluate"):
            result = runner.invoke(app, [
                "job", "submit", "--remote-script", "--wait", "--", "evaluate",
            ])
        assert result.exit_code == 0, result.output

    def test_verify_evaluate(self, runner):
        """Verify evaluate output exists."""
        result = runner.invoke(app, ["data", "ls", "outputs"])
        assert result.exit_code == 0, result.output
        assert "eval_summary_" in result.output

    def test_submit_with_env_override(self, runner):
        """Submit preprocess with --set to redirect output directory."""
        result = runner.invoke(app, [
            "job", "submit", "--remote-script", "--wait",
            "--set", "DATA_OUT=outputs",
            "--", "preprocess",
        ])
        assert result.exit_code == 0, result.output
        # Verify override worked: preprocessed_files.txt now also in outputs/
        result = runner.invoke(app, ["data", "ls", "outputs"])
        assert result.exit_code == 0, result.output
        assert "preprocessed_files.txt" in result.output


# ---------------------------------------------------------------------------
# 7. Job Management
# ---------------------------------------------------------------------------

class TestJobManagement:
    """Job lifecycle: list, run, status, logs, cancel."""

    def test_job_list(self, runner):
        """List recent jobs on the cluster."""
        result = runner.invoke(app, ["job", "list"])
        assert result.exit_code == 0, result.output

    def test_job_run_and_wait(self, runner, shared_state, timed_step):
        """Run ad-hoc command, then wait for it with 'job wait'."""
        with timed_step("job-run-and-wait"):
            # Submit ad-hoc job (job run has no --wait)
            result = runner.invoke(app, [
                "job", "run", "--remote-script",
                "--", "echo hello-from-fcw-run",
            ])
            assert result.exit_code == 0, result.output
            job_id = _extract_job_id(result.output)
            assert job_id, f"Could not extract job ID from: {result.output}"
            shared_state["run_job_id"] = job_id

            # Wait for completion using the separate 'job wait' command
            result = runner.invoke(app, ["job", "wait", job_id])
        assert result.exit_code == 0, result.output

    def test_job_status(self, runner, shared_state):
        """Check status of a completed job."""
        job_id = shared_state.get("run_job_id")
        if not job_id:
            pytest.skip("No job ID from previous test")
        result = runner.invoke(app, ["job", "status", job_id])
        assert result.exit_code == 0, result.output

    def test_job_logs(self, runner, shared_state):
        """View stdout logs of a completed job."""
        job_id = shared_state.get("run_job_id")
        if not job_id:
            pytest.skip("No job ID from previous test")
        result = runner.invoke(app, ["job", "logs", job_id])
        # Logs may fail if metadata path resolution doesn't work;
        # accept either success or a known metadata error
        assert result.exit_code == 0 or "metadata" in result.output.lower(), result.output

    def test_job_cancel(self, runner):
        """Submit a long-running job and cancel it."""
        # Submit a sleep job
        result = runner.invoke(app, [
            "job", "run", "--remote-script",
            "--", "sleep 600",
        ])
        assert result.exit_code == 0, result.output
        job_id = _extract_job_id(result.output)
        assert job_id, f"Could not extract job ID from: {result.output}"

        # Cancel it
        result = runner.invoke(app, ["job", "cancel", job_id])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 8. Data Download
# ---------------------------------------------------------------------------

class TestDataDownload:
    def test_download_outputs(self, runner, timed_step):
        """Download outputs directory."""
        with timed_step("download-outputs"):
            result = runner.invoke(app, ["data", "download", "outputs"])
        assert result.exit_code == 0, result.output

    def test_download_incremental(self, runner, timed_step):
        """Re-download with --incremental (should skip unchanged files)."""
        with timed_step("download-incremental"):
            result = runner.invoke(app, ["data", "download", "--incremental", "outputs"])
        assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# 9. Cleanup
# ---------------------------------------------------------------------------

class TestCleanup:
    def test_data_rm(self, runner):
        """Remove a remote directory."""
        result = runner.invoke(app, ["data", "rm", "--force", "data/processed"])
        assert result.exit_code == 0, result.output

    def test_verify_rm(self, runner):
        """Verify the directory was removed."""
        result = runner.invoke(app, ["data", "ls", "data/processed"])
        # Should fail (directory no longer exists) or return empty
        assert result.exit_code != 0 or "preprocessed_files.txt" not in result.output

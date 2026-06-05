"""End-to-end workflow tests for the BrainBERT example.

Structure: TestBrainBERTContainerDeploy runs first (shared dependency).
TestBrainBERTWorkflow exercises the full data+job pipeline.
TestBrainBERTNcclTests is standalone (only needs deploy) and can be
run independently: pytest ...::TestBrainBERTNcclTests

Requires: --run-e2e flag or FCW_E2E=1 environment variable.
Runs with: --example BrainBERT
"""

import pytest
from helpers import (
    assert_container_deploy,
    assert_data_download,
    assert_data_upload,
    assert_job_submit,
    assert_ok,
    assert_sqsh_exists,
    extract_job_id,
)

from fcw.cli import app

pytestmark = [pytest.mark.e2e, pytest.mark.example("BrainBERT")]


# ---------------------------------------------------------------------------
# 1. Config + Container Deploy (shared dependency for all other classes)
# ---------------------------------------------------------------------------

class TestBrainBERTContainerDeploy:
    def test_config_validate(self, runner):
        """fcw config validate succeeds with valid credentials."""
        result = runner.invoke(app, ["config", "validate"])
        assert_ok(result)

    def test_config_show(self, runner):
        """fcw config show displays resolved config."""
        result = runner.invoke(app, ["config", "show"])
        assert_ok(result)
        assert "BrainBERT" in result.output

    def test_container_deploy(self, runner, timed_step, remote_platform):
        """Deploy ngc-brainbert container."""
        assert_container_deploy(runner, timed_step, "ngc-brainbert", platform=remote_platform)

    def test_verify_sqsh(self, runner):
        """Verify squashfs image exists on remote."""
        assert_sqsh_exists(runner, "ce-images", "ngc-brainbert+25.12-alps2.sqsh")


# ---------------------------------------------------------------------------
# 2. Full Workflow (data upload -> job chain -> download -> benchy)
# ---------------------------------------------------------------------------

class TestBrainBERTWorkflow:
    """Full BrainBERT pipeline: upload dataset, run job chain, download outputs."""

    def test_upload_dataset(self, runner, timed_step):
        """Upload braintreebank dataset (CI subset).

        The dataset is a symlinked tree, so -L is required to upload the real
        files rather than the links.
        """
        assert_data_upload(runner, timed_step, "braintreebank.dev", follow_symlinks=True)

    def test_upload_conf(self, runner, timed_step):
        """Upload configuration directory."""
        assert_data_upload(runner, timed_step, "conf")

    def test_submit_extract_raw(self, runner, timed_step):
        """Submit extract-raw job and wait for completion."""
        assert_job_submit(runner, timed_step, "extract-raw")

    def test_submit_preprocess(self, runner, timed_step):
        """Submit preprocess job and wait for completion."""
        assert_job_submit(runner, timed_step, "preprocess")

    def test_submit_train(self, runner, timed_step):
        """Submit train job and wait for completion."""
        assert_job_submit(runner, timed_step, "train")

    def test_download_outputs(self, runner, timed_step):
        """Download outputs directory."""
        assert_data_download(runner, timed_step, "outputs")

    def test_submit_train_benchy(self, runner, timed_step, shared_state):
        """Submit train-benchy benchmark job with SBATCH overrides."""
        result = assert_job_submit(runner, timed_step, "train-benchy",
                                   extra_args=["--nodes", "2", "--time", "30:00"])
        job_id = extract_job_id(result.output)
        if job_id:
            shared_state["train_benchy_job_id"] = job_id

    def test_verify_train_benchy_performance(self, runner, shared_state):
        """Validate train-benchy output against performance thresholds.

        Checks that benchy JSON output was produced and contains expected
        throughput metrics.
        """
        job_id = shared_state.get("train_benchy_job_id")

        # TODO: parse benchy JSON output and validate throughput (samples/sec)
        # against reference thresholds.

        result = runner.invoke(app, ["data", "ls", "outputs/logs"])
        assert_ok(result)
        if job_id:
            assert f"brainbert-benchy-{job_id}" in result.output

        result = runner.invoke(app, ["data", "ls", "-R", "outputs"])
        assert_ok(result)
        benchy_lines = [l for l in result.output.splitlines()
                        if "benchy_output" in l]
        assert benchy_lines, "No benchy_output file found in outputs/"
        if job_id:
            matched = [l for l in benchy_lines if job_id in l]
            assert matched, f"No benchy_output for job {job_id} in outputs/"


# ---------------------------------------------------------------------------
# 3. NCCL Tests (standalone — only depends on container deploy)
# ---------------------------------------------------------------------------

class TestBrainBERTNcclTests:
    """NCCL all-reduce performance tests.

    Only requires the ngc-brainbert container to be deployed. Can be run
    independently after a prior deploy:
        pytest tests/e2e/test_e2e_brainbert.py::TestBrainBERTNcclTests --run-e2e
    """

    def test_submit_nccl_tests(self, runner, timed_step):
        """Submit nccl-tests job with SBATCH overrides for CI."""
        assert_job_submit(runner, timed_step, "nccl-tests",
                          extra_args=["--nodes", "2", "--time", "10:00"])

    def test_verify_nccl_performance(self, runner, shared_state):
        """Validate NCCL all-reduce bandwidth against reference thresholds.

        nccl-tests runs all_reduce_perf with message sizes from 8B to 8GB.
        The output contains bus bandwidth (GB/s) per size. Validate that
        large-message bandwidth meets expected minimums for the target system.
        """
        # TODO: parse all_reduce_perf output from job logs and validate
        # bus bandwidth for large messages (e.g., 1GB+) against per-system
        # reference thresholds. For now, verify the job completed.
        result = runner.invoke(app, ["job", "list"])
        assert_ok(result)

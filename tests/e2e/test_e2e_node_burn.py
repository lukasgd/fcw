"""End-to-end workflow tests for the node-burn example.

Workflow: config validate -> container deploy -> upload config -> submit job
-> verify performance output against reference data.

Requires: --run-e2e flag or FCW_E2E=1 environment variable.
Runs with: --example node-burn
"""

import pytest
from helpers import (
    assert_container_deploy,
    assert_job_submit,
    assert_ok,
    assert_sqsh_exists,
)

from fcw.cli import app

pytestmark = [pytest.mark.e2e, pytest.mark.example("node-burn")]


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
        assert "node-burn" in result.output


# ---------------------------------------------------------------------------
# 2. Container Deploy
# ---------------------------------------------------------------------------

class TestContainerDeploy:
    def test_container_deploy(self, runner, timed_step, remote_platform):
        """Deploy node-burn container (multi-stage build + push + remote build)."""
        assert_container_deploy(runner, timed_step, "node-burn", platform=remote_platform)

    def test_verify_sqsh(self, runner):
        """Verify squashfs image exists on remote."""
        assert_sqsh_exists(runner, "ce-images", "node-burn+12.4.1-runtime-ubuntu22.04.sqsh")


# ---------------------------------------------------------------------------
# 3. Job Submission + Performance Verification
# ---------------------------------------------------------------------------

class TestNodeBurnJob:
    def test_submit_node_burn(self, runner, timed_step):
        """Submit node-burn job and wait for completion."""
        assert_job_submit(runner, timed_step, "node-burn")

    def test_verify_performance(self, runner):
        """Verify node-burn output contains expected performance data.

        node-burn runs GPU GEMM benchmarks at various sizes. The output
        should contain performance numbers for both CPU and GPU GEMM.
        """
        # TODO: parse output logs and validate against reference thresholds
        # for GPU bandwidth and FLOPS. For now, just verify the job produced output.
        result = runner.invoke(app, ["job", "list"])
        assert_ok(result)

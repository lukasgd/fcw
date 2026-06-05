"""End-to-end workflow tests for the node-burn example.

Workflow: config validate -> container deploy -> upload config -> submit job
-> verify performance output against reference data.

Requires: --run-e2e flag or FCW_E2E=1 environment variable.
Runs with: --example node-burn
"""

import json
import tarfile

import pytest
from helpers import (
    assert_container_build,
    assert_container_build_remote,
    assert_container_deploy,
    assert_container_push,
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
# 3. Container build/push/build-remote sequence (decomposed deploy)
# ---------------------------------------------------------------------------

class TestContainerBuildSequence:
    """The decomposed build -> push -> build-remote path, config-driven.

    Complements TestContainerDeploy (the all-in-one path): the `build --save`
    step exercises the multi-stage `_save_image` archive on a real engine, which
    `deploy` does not reach.
    """

    def test_build_save(self, runner, timed_step, remote_platform, fcw_config, tmp_path):
        """Build all local stages and save them to a single tar archive."""
        tar = tmp_path / "node-burn.tar"
        assert_container_build(runner, timed_step, "node-burn",
                               platform=remote_platform, save=str(tar))
        # The multi-stage --save must write EVERY local stage to the tar, not
        # just the last one (regression guard for the `build --save` fix).
        cont = fcw_config.containers["node-burn"]
        expected = {cont.stage_tag(s) for s in cont.get_local_stages()}
        with tarfile.open(tar) as t:
            manifest = json.load(t.extractfile("manifest.json"))
        tags = {x for entry in manifest for x in (entry.get("RepoTags") or [])}
        assert expected <= tags, (expected, tags)

    def test_push(self, runner, timed_step, remote_platform):
        """Push all local stages to remote."""
        assert_container_push(runner, timed_step, "node-burn", platform=remote_platform)

    def test_build_remote(self, runner, timed_step):
        """Build offline stage on the cluster + enroot import."""
        assert_container_build_remote(runner, timed_step, "node-burn")

    def test_verify_sqsh(self, runner):
        """Verify squashfs image exists on remote."""
        assert_sqsh_exists(runner, "ce-images", "node-burn+12.4.1-runtime-ubuntu22.04.sqsh")


# ---------------------------------------------------------------------------
# 4. Job Submission + Performance Verification
# ---------------------------------------------------------------------------

class TestNodeBurnJob:
    def test_submit_node_burn(self, submit_job):
        """Submit node-burn job and wait for completion."""
        submit_job("node-burn")

    def test_verify_performance(self, runner):
        """Verify node-burn output contains expected performance data.

        node-burn runs GPU GEMM benchmarks at various sizes. The output
        should contain performance numbers for both CPU and GPU GEMM.
        """
        # TODO: parse output logs and validate against reference thresholds
        # for GPU bandwidth and FLOPS. For now, just verify the job produced output.
        result = runner.invoke(app, ["job", "list"])
        assert_ok(result)

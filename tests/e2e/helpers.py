"""Shared helper functions for e2e tests.

Thin wrappers around CLI invocations that handle timing and assertions.
"""

from __future__ import annotations

import os
import re
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from click.testing import Result
    from typer.testing import CliRunner

from fcw.cli import app
from fcw.core.config import load_config


def assert_ok(result: Result, args: list[str] | None = None) -> None:
    """Assert CLI result succeeded, with detailed diagnostics on failure."""
    if result.exit_code == 0:
        return
    cmd = f"fcw {' '.join(args)}" if args else "fcw <unknown>"
    msg = f"{cmd} failed (exit_code={result.exit_code})"
    if result.output:
        msg += f"\n--- output ---\n{result.output}"
    if result.exception:
        tb = "".join(traceback.format_exception(
            type(result.exception), result.exception, result.exception.__traceback__,
        ))
        msg += f"\n--- exception ---\n{tb}"
    raise AssertionError(msg)


def extract_job_id(output: str) -> str | None:
    """Extract a numeric job ID from CLI output."""
    for line in output.strip().split("\n"):
        line = line.strip()
        if line.isdigit():
            return line
        m = re.search(r"job (\d+)", line, re.IGNORECASE)
        if m:
            return m.group(1)
    return None


def invoke(runner: CliRunner, args: list[str], step_name: str | None = None,
           timed_step=None) -> object:
    """Invoke an fcw CLI command, optionally timed. Asserts exit_code == 0."""
    if timed_step and step_name:
        with timed_step(step_name):
            result = runner.invoke(app, args)
    else:
        result = runner.invoke(app, args)
    assert_ok(result, args)
    return result


def assert_container_build(runner, timed_step, name, *, stage=None, platform=None,
                           save=None, dockerfile=None, context=None):
    """Build a container image locally."""
    cmd = ["container", "build"]
    if stage:
        cmd.extend(["--stage", stage])
    if dockerfile:
        cmd.extend(["-f", dockerfile])
    if platform:
        cmd.extend(["--platform", platform])
    if save:
        cmd.extend(["--save", save])
    cmd.append(name)
    if context:
        cmd.extend(["--context", context])
    return invoke(runner, cmd, f"container-build-{name}", timed_step)


def assert_container_push(runner, timed_step, name, *, platform=None):
    """Push a container image to remote."""
    cmd = ["container", "push"]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.append(name)
    return invoke(runner, cmd, "container-push", timed_step)


def assert_container_build_remote(runner, timed_step, name, *, extra_args=None):
    """Build a container on the remote cluster."""
    cmd = ["container", "build-remote", name, "--enroot", "--wait", *(extra_args or [])]
    return invoke(runner, cmd, "container-build-remote", timed_step)


def assert_container_deploy(runner, timed_step, name, *, platform=None, stage_tars=None,
                            extra_args=None):
    """Deploy a container (build + push + remote build in one step).

    When *stage_tars* is set (engine-less consume mode), the local build+push is
    replaced by uploading pre-built per-stage tars from that directory followed by a
    remote build — no container engine required on this machine. Temporary.
    """
    if stage_tars:
        return _provision_from_stage_tars(runner, timed_step, name, stage_tars,
                                          extra_args=extra_args)
    cmd = ["container", "deploy", name, "--wait", *(extra_args or [])]
    if platform:
        cmd.extend(["--platform", platform])
    return invoke(runner, cmd, "container-deploy", timed_step)


def _stage_tar_name(stage_tag: str) -> str:
    """Canonical per-stage tar filename (matches container.py:752 / :1091)."""
    return stage_tag.replace(":", "+").replace("/", "+") + ".tar"


def _provision_from_stage_tars(runner, timed_step, name, stage_tars, *, extra_args=None):
    """Engine-less provisioning: push pre-built per-stage tars, then build-remote.

    Uses the legacy engine-free ``push <tar>`` path; the trailing slash on ``--to``
    is required because that path uploads to ``os.path.dirname(--to)``.
    """
    config = load_config("fcw.yaml")
    cont = config.containers[name]
    images_dir = config.resolve_container_images_dir(cont)
    for stage in cont.get_local_stages():
        tar = os.path.join(stage_tars, _stage_tar_name(cont.stage_tag(stage)))
        invoke(runner, ["container", "push", tar, "--to", images_dir + "/"],
               f"container-push-tar-{stage}", timed_step)
    return invoke(runner,
                  ["container", "build-remote", name, "--enroot", "--wait", *(extra_args or [])],
                  "container-build-remote", timed_step)


def save_stage_tars(runner, out_dir):
    """Producer slice (needs an engine): build + save every container's local stages.

    Writes one per-stage tar per container into *out_dir*, named per the
    container.py:752 / :1091 contract so the engine-less consumer and build-remote
    locate them. Temporary engine-less-e2e affordance.
    """
    config = load_config("fcw.yaml")
    os.makedirs(out_dir, exist_ok=True)
    for name, cont in config.containers.items():
        for stage in cont.get_local_stages():
            out = os.path.join(out_dir, _stage_tar_name(cont.stage_tag(stage)))
            invoke(runner, ["container", "build", name, "--stage", stage, "--save", out],
                   f"prepare-{name}-{stage}", None)


def assert_job_submit(runner, timed_step, job_name, *, step_name=None,
                      remote_script=False, extra_args=None):
    """Submit a job by config name and wait for completion."""
    cmd = ["job", "submit"]
    if remote_script:
        cmd.append("--remote-script")
    cmd.append("--wait")
    if extra_args:
        cmd.extend(extra_args)
    cmd.extend(["--", job_name])
    return invoke(runner, cmd, step_name or f"job-{job_name}", timed_step)


def assert_job_run(runner, timed_step, command, *, step_name="job-run",
                   remote_script=False, container=None):
    """Run an ad-hoc command via `fcw job run`."""
    cmd = ["job", "run"]
    if remote_script:
        cmd.append("--remote-script")
    if container:
        cmd.extend(["--container", container])
    cmd.extend(["--", command])
    return invoke(runner, cmd, step_name, timed_step)


def assert_data_upload(runner, timed_step, path, *, incremental=False, follow_symlinks=False):
    """Upload a data directory."""
    cmd = ["data", "upload"]
    if incremental:
        cmd.append("--incremental")
    if follow_symlinks:
        cmd.append("--follow-symlinks")
    cmd.append(path)
    step = f"upload-{path}" if not incremental else f"upload-incremental-{path}"
    return invoke(runner, cmd, step, timed_step)


def assert_data_download(runner, timed_step, path, *, incremental=False):
    """Download a data directory."""
    cmd = ["data", "download"]
    if incremental:
        cmd.append("--incremental")
    cmd.append(path)
    step = f"download-{path}" if not incremental else f"download-incremental-{path}"
    return invoke(runner, cmd, step, timed_step)


def assert_remote_ls_contains(runner, path, expected):
    """Assert that `fcw data ls <path>` output contains expected string."""
    args = ["data", "ls", path]
    result = runner.invoke(app, args)
    assert_ok(result, args)
    assert expected in result.output, f"Expected '{expected}' in ls output: {result.output}"
    return result


def assert_remote_ls_not_contains(runner, path, expected):
    """Assert that `fcw data ls <path>` output does NOT contain expected string."""
    result = runner.invoke(app, ["data", "ls", path])
    assert result.exit_code != 0 or expected not in result.output


def assert_file_contains_tokens(local_path, tokens, expected_marker_count=None):
    """Assert a downloaded file contains every token, with order independence.

    Optionally assert the number of FCW-DATA marker lines equals
    *expected_marker_count* (one per source file that was concatenated).
    """
    with open(local_path) as f:
        content = f.read()
    for tok in tokens:
        assert tok in content, f"Expected token '{tok}' in {local_path}, got:\n{content}"
    if expected_marker_count is not None:
        markers = [ln for ln in content.splitlines() if ln.startswith("FCW-DATA-")]
        assert len(markers) == expected_marker_count, (
            f"Expected {expected_marker_count} FCW-DATA markers in {local_path}, "
            f"found {len(markers)}"
        )


def assert_sqsh_exists(runner, ce_images_path, sqsh_name):
    """Verify a .sqsh file exists on remote."""
    args = ["data", "ls", ce_images_path]
    result = runner.invoke(app, args)
    assert_ok(result, args)
    assert sqsh_name in result.output, f"Expected '{sqsh_name}' in: {result.output}"

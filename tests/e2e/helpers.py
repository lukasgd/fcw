"""Shared helper functions for e2e tests.

Thin wrappers around CLI invocations that handle timing and assertions.
"""

from __future__ import annotations

import re
import traceback
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from click.testing import Result
    from typer.testing import CliRunner

from fcw.cli import app


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


def assert_container_build_remote(runner, timed_step, name, **kwargs):
    """Build a container on the remote cluster."""
    cmd = ["container", "build-remote", name, "--enroot", "--wait"]
    return invoke(runner, cmd, "container-build-remote", timed_step)


def assert_container_deploy(runner, timed_step, name, *, platform=None):
    """Deploy a container (build + push + remote build in one step)."""
    cmd = ["container", "deploy", name, "--wait"]
    if platform:
        cmd.extend(["--platform", platform])
    return invoke(runner, cmd, "container-deploy", timed_step)


def assert_job_submit(runner, timed_step, job_name, *, step_name=None,
                      remote_script=True, extra_args=None):
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
                   remote_script=True, container=None):
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


def assert_sqsh_exists(runner, ce_images_path, sqsh_name):
    """Verify a .sqsh file exists on remote."""
    args = ["data", "ls", ce_images_path]
    result = runner.invoke(app, args)
    assert_ok(result, args)
    assert sqsh_name in result.output, f"Expected '{sqsh_name}' in: {result.output}"

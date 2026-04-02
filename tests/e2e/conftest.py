"""E2E test fixtures requiring real FirecREST access."""

import os
import shutil
import sys
import uuid
from contextlib import contextmanager

import pytest

import firecrest

from fcw.core import load_config, get_client, get_async_client, get_system, get_account
sys.path.insert(0, os.path.dirname(__file__))
from perf import TimingCollector, load_thresholds, timed_step as _timed_step  # noqa: E402


# Session-level failure tracking
_session_failed = False


def pytest_configure(config):
    """Create the timing collector early so it can be shared with terminal summary."""
    config._perf_collector = TimingCollector()


def pytest_runtest_makereport(item, call):
    """Track whether any test in the session has failed."""
    global _session_failed
    if call.when == "call" and call.excinfo is not None:
        _session_failed = True


@pytest.fixture(scope="session")
def example_workdir(request, tmp_path_factory):
    """Copy the selected example to a temp dir and chdir there for the session."""
    if not os.environ.get("FIRECREST_SCRATCH"):
        pytest.skip("FIRECREST_SCRATCH not set (remote scratch path, e.g. /iopsstor/scratch/cscs/user)")

    example_name = request.config.getoption("--example")
    repo_root = os.path.join(os.path.dirname(__file__), "..", "..")
    source = os.path.join(repo_root, "examples", example_name)
    if not os.path.isdir(source):
        pytest.skip(f"Example '{example_name}' not found at {source}")

    tmp_dir = tmp_path_factory.mktemp(f"e2e-{example_name}")
    workdir = tmp_dir / example_name
    shutil.copytree(source, workdir)

    # Set a unique run ID so ${FCW_<EXAMPLE>_RUN_ID} in fcw.yaml resolves to a fresh remote dir.
    # If already set by the user, reuse it (allows re-running against the same remote dir).
    run_id_var = f"FCW_{example_name.upper().replace('-', '_')}_RUN_ID"
    generated_id = False
    if run_id_var not in os.environ:
        os.environ[run_id_var] = uuid.uuid4().hex[:8]
        generated_id = True

    run_id = os.environ[run_id_var]
    print(f"\nE2E: {run_id_var}={run_id}")

    old_cwd = os.getcwd()
    os.chdir(workdir)

    # Load config to show resolved remote workdir
    config = load_config(os.path.join(str(workdir), "fcw.yaml"))
    print(f"E2E: Remote workdir: {config.workdir.remote}")

    yield workdir

    os.chdir(old_cwd)

    # Conditional remote cleanup
    cleanup = request.config.getoption("--cleanup-remote")
    if cleanup and not _session_failed:
        try:
            client = get_client()
            system = get_system()
            print(f"\nE2E: Cleaning up remote workdir: {config.workdir.remote}")
            client.rm(system, config.workdir.remote)
        except Exception as e:
            print(f"E2E: Remote cleanup failed: {e}")
    elif _session_failed:
        print(f"\nE2E: Tests failed. To re-run against the same remote dir:")
        print(f"  {run_id_var}={run_id} pytest tests/ --run-e2e -v")

    if generated_id:
        os.environ.pop(run_id_var, None)


@pytest.fixture(autouse=True, scope="session")
def _chdir_to_workdir(example_workdir):
    """Ensure CWD is the temp workdir for all e2e tests."""
    pass


@pytest.fixture(scope="session")
def fcw_config(example_workdir):
    """Load the example project config from the temp workdir."""
    config_path = os.path.join(str(example_workdir), "fcw.yaml")
    return load_config(config_path)


@pytest.fixture(scope="session")
def system():
    """Get the target system from environment."""
    return get_system()


@pytest.fixture(scope="session")
def account():
    """Get the SLURM account from environment."""
    return get_account()


@pytest.fixture(scope="session")
def client():
    """Get a real FirecREST v2 sync client."""
    return get_client()


@pytest.fixture(scope="session")
def async_client():
    """Get a real FirecREST v2 async client."""
    return get_async_client()


@pytest.fixture(scope="session")
def remote_workdir(fcw_config):
    """Get the remote workdir from config."""
    return fcw_config.workdir.remote


@pytest.fixture
def runner():
    """Get a Typer CLI test runner."""
    from typer.testing import CliRunner

    return CliRunner()


@pytest.fixture(scope="session")
def shared_state():
    """Session-scoped dict for sharing state (e.g. job IDs) between tests."""
    return {}


@pytest.fixture(scope="session")
def remote_platform(client, system):
    """Detect remote system platform for cross-architecture builds."""
    from fcw.commands.container import _detect_remote_platform

    return _detect_remote_platform(client, system)


# ---------------------------------------------------------------------------
# Performance timing fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def perf_collector(request):
    """Session-level timing collector, shared with terminal summary hook."""
    return request.config._perf_collector


@pytest.fixture(scope="session")
def perf_thresholds(system):
    """Load performance thresholds for the target system."""
    config_path = os.path.join(os.path.dirname(__file__), "perf_thresholds.yaml")
    if not os.path.exists(config_path):
        return {}
    return load_thresholds(config_path, system)


@pytest.fixture
def timed_step(request, perf_collector, perf_thresholds):
    """Context-manager factory for timing steps within a test.

    Usage::

        def test_upload_data(self, runner, timed_step):
            with timed_step("upload-data"):
                result = runner.invoke(app, ["data", "upload", "data/raw"])
            assert result.exit_code == 0, result.output
    """
    test_name = request.node.nodeid

    @contextmanager
    def _step(step_name: str):
        threshold = perf_thresholds.get(step_name)
        with _timed_step(perf_collector, test_name, step_name, threshold_s=threshold):
            yield

    return _step


@pytest.fixture(scope="session", autouse=True)
def _check_perf_violations(request, perf_collector):
    """After all tests, fail if --check-perf is active and violations exist."""
    yield
    check = request.config.getoption("--check-perf", default=False) or os.environ.get(
        "FCW_CHECK_PERF"
    )
    if not check:
        return
    violations = perf_collector.violations()
    if violations:
        names = ", ".join(v.step_name for v in violations)
        pytest.fail(
            f"Performance check failed: {len(violations)} step(s) exceeded thresholds: {names}",
            pytrace=False,
        )


def pytest_terminal_summary(terminalreporter, config):
    """Print timing summary and check for threshold violations."""
    collector = getattr(config, "_perf_collector", None)
    if collector is None or not collector.timings:
        return

    terminalreporter.section("E2E Step Timings")
    for t in collector.timings:
        threshold_info = ""
        if t.threshold_s is not None:
            exceeded = t.elapsed_s > t.threshold_s
            marker = "FAIL" if exceeded else "OK"
            threshold_info = f"  [{marker}: threshold {t.threshold_s:.0f}s]"
        terminalreporter.line(f"  {t.step_name:.<40s} {t.elapsed_s:>8.1f}s{threshold_info}")

    violations = collector.violations()
    if violations:
        terminalreporter.section("Performance Threshold Violations")
        for v in violations:
            pct = (v.elapsed_s / v.threshold_s - 1) * 100
            terminalreporter.line(
                f"  {v.step_name}: {v.elapsed_s:.1f}s exceeded threshold of "
                f"{v.threshold_s:.0f}s (+{v.elapsed_s - v.threshold_s:.1f}s / +{pct:.0f}%)"
            )

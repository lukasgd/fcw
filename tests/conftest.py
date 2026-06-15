"""Shared test fixtures."""

import os
import textwrap

import pytest
import yaml

from fcw.core.config import load_config


SAMPLE_CONFIG_YAML = textwrap.dedent("""\
    project: test-project

    workdir:
      remote: /scratch/user/test-project
      local: .

    directories:
      data/raw:
        type: in
      data/processed:
        type: out
      outputs:
        type: out
      config:
        type: in

    containers:
      app:
        file: ./Dockerfile
        tag: my-fcw-app:latest
        remote_path: ./ce-images/
        toml: ./env/container.toml
      aux:
        file: ./Dockerfile.aux
        tag: fcw-aux:latest

    jobs:
      preprocess:
        script: slurm/preprocess.sh
        container: app
        env_paths:
          DATA_IN: data/raw
          DATA_OUT: data/processed
      train:
        script: slurm/train.sh
        container: app
        time: "12:00:00"
        nodes: 1
        env_paths:
          DATA_DIR: data/processed
          OUTPUT_DIR: outputs
        env:
          EPOCHS: "10"
""")


@pytest.fixture
def sample_config_yaml():
    return SAMPLE_CONFIG_YAML


@pytest.fixture
def sample_config(tmp_path):
    """Write sample config to tmp_path and return loaded FcwConfig."""
    config_path = tmp_path / "fcw.yaml"
    config_path.write_text(SAMPLE_CONFIG_YAML)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return load_config(str(config_path))
    finally:
        os.chdir(old_cwd)


def pytest_addoption(parser):
    parser.addoption("--run-e2e", action="store_true", default=False, help="Run e2e tests")
    parser.addoption("--example", default="basic", help="Example project for e2e tests")
    parser.addoption("--cleanup-remote", action="store_true", default=False,
                     help="Delete remote workdir after successful e2e run")
    parser.addoption("--check-perf", action="store_true", default=False,
                     help="Fail e2e tests if step timings exceed thresholds")
    parser.addoption("--max-node-hours", type=float, default=None,
                     help="Cap each e2e job at this many node-hours (nodes x walltime) "
                          "via SBATCH overrides; unset = use the job's configured walltime")
    parser.addoption("--remote-script", action="store_true", default=False,
                     help="Use `fcw job submit/run --remote-script` for e2e job submission "
                          "(uploads the script before sbatch; workaround for systems where "
                          "slurmrestd inline-script submission fails, e.g. lys)")
    # TODO(temporary): remove with the engine-less e2e affordance. Lets a client
    # without a container engine run the suite by consuming pre-built stage tars.
    parser.addoption("--stage-tars", default=None,
                     help="[temporary] Directory of pre-built per-stage image tars; the "
                          "container phase runs engine-free, pushing these instead of building")
    parser.addoption("--prepare-stage-tars", default=None,
                     help="[temporary] Producer slice: build each container's local stages "
                          "and save them as per-stage tars into this directory (needs an engine)")


def pytest_configure(config):
    # Resolve path options to absolute while cwd is still the invocation dir, so they
    # survive example_workdir chdir'ing into a temp dir (relative paths would break).
    for opt in ("stage_tars", "prepare_stage_tars"):
        val = getattr(config.option, opt)
        if val:
            setattr(config.option, opt, os.path.abspath(val))

    config.addinivalue_line("markers", "example(name): select which example project this test targets")
    config.addinivalue_line("markers", "needs_engine: requires a local container engine (skipped under --stage-tars)")
    config.addinivalue_line("markers", "engineless_only: only runs in engine-less consume mode (requires --stage-tars)")


def pytest_collection_modifyitems(config, items):
    if not (config.getoption("--run-e2e") or os.environ.get("FCW_E2E")):
        skip_e2e = pytest.mark.skip(reason="need --run-e2e or FCW_E2E=1 to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)
        return

    selected_example = config.getoption("--example")
    for item in items:
        example_markers = list(item.iter_markers("example"))
        if example_markers:
            example_name = example_markers[0].args[0]
            if example_name != selected_example:
                item.add_marker(pytest.mark.skip(
                    reason=f"example '{example_name}' not selected (--example {selected_example})"
                ))

    # Engine-less consume mode (temporary): with --stage-tars the client has no
    # container engine, so skip engine-only tests; without it, skip the consume-only
    # provisioning tests that replace them.
    stage_tars = config.getoption("--stage-tars")
    skip_engine = pytest.mark.skip(reason="engine-less mode (--stage-tars): engine-only test skipped")
    skip_engineless = pytest.mark.skip(reason="needs --stage-tars (engine-less consume mode)")
    for item in items:
        if stage_tars and "needs_engine" in item.keywords:
            item.add_marker(skip_engine)
        if not stage_tars and "engineless_only" in item.keywords:
            item.add_marker(skip_engineless)

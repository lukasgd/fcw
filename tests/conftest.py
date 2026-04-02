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
        env:
          DATA_IN: data/raw
          DATA_OUT: data/processed
      train:
        script: slurm/train.sh
        container: app
        time: "12:00:00"
        nodes: 1
        env:
          DATA_DIR: data/processed
          OUTPUT_DIR: outputs
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


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-e2e") or os.environ.get("FCW_E2E"):
        return
    skip_e2e = pytest.mark.skip(reason="need --run-e2e or FCW_E2E=1 to run")
    for item in items:
        if "e2e" in item.keywords:
            item.add_marker(skip_e2e)

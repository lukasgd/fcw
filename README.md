# FirecREST Container Workflows

A command-line tool for orchestrating HPC container workflows via [FirecREST](https://docs.cscs.ch/access/firecrest/) built on [PyFirecREST](https://github.com/eth-cscs/pyfirecrest).

## Features

- **Container deployment**: Build, deploy, and iterate on container images (mirror bind-mounted patches and rebuild images when stable)
- **Data transfer**: Directory mirroring with continuous upload/download using direction enforcement
- **Job management**: Declarative definition of SLURM jobs and flexible CLI for submission 
- **FUSE mount**: Mount remote storage as local filesystem over FirecREST (optional)

## Installation

```bash
pip install fcw  # add [dev] to install with testing utilities (incl. performance)

# With FUSE support (requires libfuse3-dev)
pip install fcw[fuse]
```

## Shell completion

`fcw` ships with tab completion for commands, sub-commands, and options. Install it
once for your shell (bash/zsh/fish/powershell), then restart the shell:

```bash
fcw --install-completion      # detects your shell and installs
fcw --show-completion bash    # print the script instead of installing
```

After installation, `fcw <TAB>` completes command groups and `fcw job <TAB>` completes
job verbs.

## Quick start


Set up FirecREST credentials, e.g. for Clariden:

```bash
export FIRECREST_URL="https://api.cscs.ch/ml/firecrest/v2"
export AUTH_TOKEN_URL="https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"
export FIRECREST_SYSTEM="clariden"
export FIRECREST_ACCOUNT="<account>"
export FIRECREST_CLIENT_ID="<client_id>"
export FIRECREST_CLIENT_SECRET="<client_secret>"
```

Initialize a project:

```bash
fcw config init
fcw config validate
```

Interactively add/remove data directories, containers and jobs to configure your project using

```bash
fcw config directory add ...
fcw config container add ...
fcw config job add ...
```

## Configuration

Example `fcw.yaml`:

```yaml
project: my-fcw-app

workdir:
  remote: ${FIRECREST_SCRATCH}/my-fcw-app
  local: .

directories:  # dataflow types: in/out/both (relative to remote job)
  data/raw:
    type: in
  data/processed:
    type: out
  configs:
    type: in
  outputs:
    type: out

containers:  # using multistage Dockerfiles (download and build-offline by default)
  app-main:
    file: ./env/Dockerfile.main
    tag: my-fcw-app-main:26.03
    remote_path: ce-images/
    toml: ./env/app-main-26.03.toml  # optional, user-editable enroot environment
    platform: linux/arm64  # optional, for cross-arch builds (auto-detect if omitted)
  app-prep:
    file: ./env/Dockerfile.prep
    tag: my-fcw-app-prep:26.01
    remote_path: ce-images/
    toml: ./env/app-prep-26.01.toml
    platform: linux/arm64

jobs:  # Job definitions
       # at submit time, fcw inlines the TOML and resolves the image path automatically.
       # env_paths: path-valued vars — relative values are expanded to
       #   ${workdir.remote}/<path>.
       # env: literal vars — passed through untouched.
       # Both are injected as shell defaults (export VAR="${VAR:-value}"), so pre-set
       # env vars take precedence. A CLI --set override of a declared env_paths key
       # is resolved against ${workdir.remote}; any other --set value is literal.
  preprocess:
    script: slurm/preprocess.sh
    container: app-prep  # references a container from the containers section
    env_paths:
      DATA_IN: data/raw
      DATA_OUT: data/processed

  train:
    script: slurm/train.sh
    container: app-main
    time: "12:00:00"
    nodes: 2
    env_paths:  # evaluated relative to ${workdir.remote}
      DATA_DIR: data/processed
      CONFIG_DIR: configs
      OUTPUT_DIR: outputs
    env:
      EPOCHS: "100"  # literal (not interpreted as a path)

  evaluate:
    script: slurm/evaluate.sh
    container: app-main
    env_paths:
      MODEL_DIR: outputs
```

## Usage

### Data transfer

```bash
# Upload input data
fcw data upload data/raw

# Download outputs with incremental sync
fcw data download outputs --incremental --watch

# List remote directory
fcw data ls outputs -R
```

### Job submission

Jobs are submitted using the `--` separator pattern: SBATCH options before `--`, 
script/job name after.

The hot verbs `submit`, `run`, `logs`, `wait`, and `cancel` are also available at the
top level, so `fcw submit train.sh` is shorthand for `fcw job submit train.sh`. The
full verb set (including `status` and `list`) lives under `fcw job`.

```bash
# Single job submission (top-level shorthand; `fcw job submit ...` also works)
fcw submit train.sh           # explicit script path
fcw submit train              # use jobs.train config from fcw.yaml

# Override SBATCH options
fcw job submit --time 12:00:00 --nodes 4 -- train.sh

# Chain jobs with SLURM dependencies
JOB1=$(fcw job submit preprocess.sh)
fcw job submit --dependency afterok:$JOB1 -- train.sh

# Set additional environment variables after --
fcw job submit train --set CONFIG=configs/exp1.yaml --set EPOCHS=100

# Run individual command
fcw job run 'nvidia-smi'
fcw job run --time 01:00:00 --nodes 2 -- 'python train.py'

# Monitor jobs
fcw job logs $JOB1 --follow
fcw job wait $JOB1
```

### Container Management

#### Initial deployment

The build process is distributed across machines - a download stage built on the client that collects the base image(s) and dependencies and a build-offline stage on the remote cluster to build the final image from the dependencies/base image(s).

This can be run end-to-end with the command `deploy`, which builds local stages, pushes them, and submits a remote build job according to config in `fcw.yaml`:
```bash
fcw container deploy app-main --wait
```

For more control, the same workflow as explicit steps:
```bash
fcw container build app-main
fcw container push app-main
fcw container build-remote app-main --enroot --wait
```

For customizing the build, these commands allow overriding Dockerfile, tag, platform, build args, etc. from the CLI (takes priority over `fcw.yaml`).

When the client and remote cluster have different CPU architectures
(e.g., building on x86_64 for an arm64 cluster), `fcw` handles this automatically: `container build` and `container deploy` detect the remote system's architecture via FirecREST and pass `--platform` to podman/docker (set `platform: linux/arm64` in the container config in `fcw.yaml` to skip auto-detection). Furthermore, the remote build step verifies the image architecture matches the compute node.

#### Iterative code development

For quick iteration without rebuilding the full container:

```bash
# extract code from a container build stage locally
# (stage defaults to 'download'; override with --stage)
fcw container extract app-main /workspace/BrainBERT ./code
```

`extract` also writes a sidecar `./code.meta.json` recording the build stage and
in-container path. Later commands use this to map patches back to the stage
they came from.

Edit `./code` locally. Test by bind-mounting the patched code (no rebuild):
```bash
fcw container patch --container app-main ./code
```
Mount target defaults to the sidecar's `container_path`; override with
`./code:/some/other/path` if needed. The container's TOML
(`containers.app-main.toml` in `fcw.yaml`) is updated in place and uploaded.
Multiple dumps can be patched at once:
```bash
fcw container patch -c app-main ./code ./configs
```

Run a test job with the patched container:
```bash
fcw job submit --container app-main -- slurm/test.sh
```

Repeat edit/patch/test until stable. Then bake the accumulated patches into
a new image:
```bash
fcw container rebuild app-main --tag app-main:v2 --enroot --wait
```
`rebuild` reads the `.patches/` bind-mounts from the container's TOML,
groups them by stage via the uploaded sidecars, applies each group to its
stage's image on the remote (loading all local stages, `podman cp` + commit
for patched ones), then rebuilds the remote stage. A new TOML without patch
mounts is written next to the original, the rebuilt container is registered
in `fcw.yaml` (e.g. `app-main-v2`), and per-stage tars are saved under the
new tag so further rebuilds (`app-main:v2 -> app-main:v3`) work the same way.

For rebuilds without a pre-existing TOML, pass dumps directly (Mode B):
```bash
fcw container rebuild app-main --tag app-main:v2 \
    --dump ./code --dump ./configs --wait
```

Now you can re-run the same job against the rebuilt container image. The example script appended below illustrates this workflow.

### FUSE Mount (experimental)

This is an experimental feature and requires installation with FUSE support. Mount directory from remote with:
```bash
fcw mount start outputs ./local-outputs
```
This allows working with files locally, e.g.
```bash
tail -f ./local-outputs/train.log
```
Note that for expensive filesystem operations, continuous synchronization is recommended over FUSE-mounting due to better performance.

When done, unmount with
```
fcw mount stop ./local-outputs
```

## Example Projects

- **[basic](examples/basic/)** — Minimal example demonstrating a fully fcw-managed pipeline. See the [e2e workflow](examples/basic/e2e_workflow.md).
- **[node-burn](examples/node-burn/)** — Benchmarking of GEMM operations on CPU and GPU on an HPC cluster analogous to the [CSCS ReFrame test-suite](https://github.com/eth-cscs/cscs-reframe-tests). Demonstrates multi-stage container builds with different build-time vs run-time base images.
- **[BrainBERT](examples/BrainBERT/)** — End-to-end pre-training of a Transformer model for intra-cranial EEG data. Includes multi-stage container build, data transfer and preprocessing, distributed training, and benchmarking (I/O, communication and training throughput). See the [fcw workflow guide](examples/BrainBERT/fcw_e2e_workflow.md).

## Example: Full Training Workflow

```bash
#!/bin/bash -l
set -e

# Upload input data and experiment configs
fcw data upload data/raw
fcw data upload configs

# Build and deploy container (first time)
fcw container deploy app --wait

# Run preprocessing
JOB_PREP=$(fcw job submit --time 01:00:00 -- slurm/preprocess.sh)

# Sync configs continuously in background (picks up edits during the run)
fcw data upload configs --incremental --watch &
SYNC_PID=$!

# Run multiple training experiments (all depend on preprocessing)
JOB_T1=$(fcw job submit --dependency afterok:$JOB_PREP -- train --set CONFIG=configs/exp1.yaml)
JOB_T2=$(fcw job submit --dependency afterok:$JOB_PREP -- train --set CONFIG=configs/exp2.yaml)
JOB_T3=$(fcw job submit --dependency afterok:$JOB_PREP -- train --set CONFIG=configs/exp3.yaml)

# Evaluate all (depends on all training jobs)
fcw job submit --dependency afterok:$JOB_T1:$JOB_T2:$JOB_T3 -- slurm/evaluate.sh

# Monitor outputs
fcw data download outputs --watch --incremental

kill $SYNC_PID
```

## Example: Code Iteration Workflow

```bash
#!/bin/bash -l
# Fast iteration on code without rebuilding full container

# One-time setup: extract code (writes ./code.meta.json sidecar)
fcw container extract my-fcw-app /workspace/BrainBERT ./code

# Edit loop
while true; do
    # Edit ./code locally with your favorite editor...
    read -p "Press Enter to test changes..."

    # Upload and add bind-mount to containers.my-fcw-app.toml
    fcw container patch --container my-fcw-app ./code

    # Run test job
    fcw job submit --container my-fcw-app --time 00:30:00 -- slurm/test.sh
done

# When satisfied, bake accumulated patches into a new image
fcw container rebuild my-fcw-app --tag my-fcw-app:v2 --enroot --wait
```


## Tests

### Linting & type checking

```bash
ruff check src/
mypy src/fcw/
```

### Unit tests

```bash
pytest
pytest tests/test_job.py::TestApplySbatchOverrides -v   # single test
```

### End-to-end tests

e2e tests require FirecREST credentials (`FIRECREST_URL`, `FIRECREST_CLIENT_ID`,
`FIRECREST_CLIENT_SECRET`, `AUTH_TOKEN_URL`) and `FIRECREST_SCRATCH` to be set.

```bash
# Run e2e tests for a given example
pytest tests/ --run-e2e --example basic -vv       # default
pytest tests/ --run-e2e --example node-burn -vv
pytest tests/ --run-e2e --example BrainBERT -vv

# Or via env var
FCW_E2E=1 pytest tests/ --example BrainBERT
```

Each example has its own test file (`tests/e2e/test_e2e_<name>.py`) and performance thresholds (`examples/<name>/e2e_perf_thresholds.yaml`).

Run a specific test class (in particular, NCCL tests without the full BrainBERT workflow):

```bash
pytest tests/e2e/test_e2e_brainbert.py::TestBrainBERTNcclTests --run-e2e --example BrainBERT
```

#### Clients without a container engine

A client with no podman/docker can still run the full e2e suite by consuming pre-built
per-stage image tars produced on a machine that *does* have an engine. The suite splits into
a producer slice (`--prepare-stage-tars`, builds and saves each container's local stages) and
an engine-free consumer slice (`--stage-tars`, which skips the engine-only tests and instead
pushes the tars and runs `build-remote`):

```bash
# On an engine-equipped box: build and save the local stage tars (absolute path required)
pytest tests/e2e/test_e2e_prepare.py --run-e2e --example basic --prepare-stage-tars "$PWD/tars-basic"

# Transfer tars-basic/ to the engine-less client, then consume them (no podman/docker needed)
pytest tests/e2e --run-e2e --example basic --stage-tars "$PWD/tars-basic" -v
```

See [`tests/e2e/README.md`](tests/e2e/README.md) for the full three-phase (prepare →
transfer → consume) workflow and caveats.

### Performance tests

Every e2e step is timed and compared against per-system thresholds defined in
`examples/<name>/e2e_perf_thresholds.yaml`. Timing results are always printed
in the terminal summary. To **fail the test run** when thresholds are exceeded:

```bash
pytest tests/ --run-e2e --example BrainBERT --check-perf

# Or via env var
FCW_CHECK_PERF=1 pytest tests/ --run-e2e --example BrainBERT
```

To run only the performance-focused tests (training throughput, NCCL bandwidth,
GPU GEMM benchmarks):

```bash
# BrainBERT: NCCL all-reduce bandwidth
pytest tests/e2e/test_e2e_brainbert.py::TestBrainBERTNcclTests --run-e2e --example BrainBERT

# BrainBERT: training throughput (train-benchy)
pytest tests/e2e/test_e2e_brainbert.py::TestBrainBERTWorkflow -k "train_benchy" --run-e2e --example BrainBERT

# node-burn: GPU GEMM benchmarks
pytest tests/e2e/test_e2e_node_burn.py::TestNodeBurnJob --run-e2e --example node-burn
```

### Remote cleanup

By default, remote workdirs are preserved after a run (useful for debugging).
To clean up automatically after a successful run:

```bash
pytest tests/ --run-e2e --cleanup-remote
```

On failure, the test output prints the `FCW_<EXAMPLE>_RUN_ID` so you can re-run
against the same remote directory without re-uploading data.
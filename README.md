# fcw - FirecREST Workflow CLI

A command-line tool for orchestrating HPC workflows via [FirecREST](https://github.com/eth-cscs/pyfirecrest).

## Features

- **Data Transfer**: Upload/download with directory type enforcement (`in`/`out`/`both`)
- **Job Management**: Submit jobs from TOML with SBATCH overrides via `--` separator
- **Container Management**: Build, deploy, and iterate on container images (fast iteration with bind-mounts and image rebuild when stable)
- **FUSE Mount**: Mount remote storage as local filesystem (optional)

## Installation

```bash
pip install fcw

# With FUSE support (requires libfuse3-dev)
pip install fcw[fuse]
```

## Quick Start

1. Set up FirecREST credentials:

```bash
export FIRECREST_URL="https://api.cscs.ch/firecrest/v2"
export FIRECREST_CLIENT_ID="your-client-id"
export FIRECREST_CLIENT_SECRET="your-client-secret"
export AUTH_TOKEN_URL="https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"
export FIRECREST_SYSTEM="clariden"
export FIRECREST_ACCOUNT="your-account"
```

2. Initialize a project:

```bash
fcw config init
fcw config validate
```

3. Edit `fcw.yaml` to configure your project.

## Configuration

Example `fcw.yaml`:

```yaml
project: my-hpc-app

workdir:
  remote: /scratch/${USER}/my-project
  local: .

# Directory types: in (upload only), out (download only), both (bidirectional)
directories:
  data/raw:
    type: in
  data/processed:
    type: out
  outputs:
    type: out
  code:
    type: both

containers:
  app:
    file: ./env/Dockerfile
    tag: myapp:latest
    remote_path: images/myapp.sqsh

jobs:
  preprocess:
    script: slurm/preprocess.sh
    env:
      DATA_IN: data/raw
      DATA_OUT: data/processed

  train:
    script: slurm/train.sh
    time: "12:00:00"
    nodes: 1
    env:
      DATA_DIR: data/processed
      OUTPUT_DIR: outputs

  evaluate:
    script: slurm/evaluate.sh
    env:
      MODEL_DIR: outputs
```

## Usage

### Data Transfer

```bash
# Upload input data
fcw data upload data/raw

# Download outputs with incremental sync
fcw data download outputs --incremental --watch

# List remote directory
fcw data ls outputs -R
```

### Job Submission

Jobs are submitted using the `--` separator pattern: SBATCH options before `--`, 
script/job name after.

```bash
# Simple submission (script path or config job name)
fcw job submit train.sh
fcw job submit train                    # Uses jobs.train.script from fcw.yaml

# Override SBATCH options (applied to script)
fcw job submit --time 24:00:00 --nodes 4 -- train.sh

# Chain jobs with dependencies
JOB1=$(fcw job submit preprocess.sh)
fcw job submit --dependency afterok:$JOB1 -- train.sh

# Set environment variables
fcw job submit train --set CONFIG=exp1.yaml --set EPOCHS=100

# Ad-hoc command
fcw job run 'nvidia-smi'
fcw job run --time 01:00:00 --nodes 2 -- 'python train.py'

# Monitor jobs
fcw job logs $JOB1 --follow
fcw job wait $JOB1
```

### Container Management

#### Initial Build & Deploy

For multi-stage Dockerfiles (download + build-offline pattern):

```bash
# Build download stage locally (fetches dependencies, requires network)
fcw container build --stage download -f env/Dockerfile.prod-multistage --build-arg BASE_IMAGE=ubuntu:24.04 -t myapp:download .

# Push download image to remote
fcw container push myapp:download

# Build offline stage on the cluster and import as enroot squashfs
fcw container build-remote myapp:download \
    -f env/Dockerfile.prod-multistage -t myapp:latest \
    --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \
    --enroot --wait
```

#### Code Iteration Workflow

For fast iteration without rebuilding the full container:

```bash
# 1. Extract code from container for local editing
fcw container extract myapp:download /workspace/BrainBERT ./code

# 2. Edit ./code locally...

# 3a. Quick iteration: bind-mount patched code (no rebuild)
fcw container patch ./code /workspace/BrainBERT --toml env/container.toml
# Then: srun --environment env/container.toml python train.py

# 3b. Bake changes: patch + rebuild (when satisfied with changes)
fcw container update ./code myapp:download /workspace/BrainBERT \
    --tag myapp:v2 --rebuild --dockerfile env/Dockerfile.prod-multistage \
    --build-arg BASE_IMAGE=ubuntu:24.04 --enroot --wait
```

### FUSE Mount (Optional)

```bash
# Mount remote storage
fcw mount start outputs ./local-outputs --read-only

# Work with files locally
tail -f ./local-outputs/train.log

# Unmount
fcw mount stop ./local-outputs
```

## Example: Full Training Workflow

```bash
#!/bin/bash
set -e

# Upload input data
fcw data upload data/raw

# Build and deploy container (first time)
fcw container build --stage download -t myapp:download .
fcw container push myapp:download
fcw container build-remote myapp:download \
    -f env/Dockerfile.prod-multistage -t myapp:latest \
    --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \
    --enroot --wait

# Run preprocessing
JOB_PREP=$(fcw job submit --time 01:00:00 -- slurm/preprocess.sh)

# Run multiple training experiments (all depend on preprocessing)
JOB_T1=$(fcw job submit --dependency afterok:$JOB_PREP -- train --set CONFIG=exp1.yaml)
JOB_T2=$(fcw job submit --dependency afterok:$JOB_PREP -- train --set CONFIG=exp2.yaml)
JOB_T3=$(fcw job submit --dependency afterok:$JOB_PREP -- train --set CONFIG=exp3.yaml)

# Evaluate all (depends on all training jobs)
fcw job submit --dependency afterok:$JOB_T1:$JOB_T2:$JOB_T3 -- slurm/evaluate.sh

# Monitor outputs
fcw data download outputs --watch --incremental
```

## Example: Code Iteration Workflow

```bash
#!/bin/bash
# Fast iteration on code without rebuilding full container

# One-time setup: extract code
fcw container extract myapp:download /workspace/BrainBERT ./code

# Edit loop
while true; do
    # Edit ./code locally with your favorite editor...
    read -p "Press Enter to test changes..."
    
    # Upload and configure bind-mount
    fcw container patch ./code /workspace/BrainBERT --toml env/dev.toml
    
    # Run test job
    fcw job submit --time 00:30:00 -- slurm/test.sh
done

# When satisfied, bake changes into new image
fcw container update ./code myapp:download /workspace/BrainBERT \
    --tag myapp:v2 --rebuild --dockerfile env/Dockerfile.prod-multistage \
    --build-arg BASE_IMAGE=ubuntu:24.04 --enroot --wait
```

## License

MIT

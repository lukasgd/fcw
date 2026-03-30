# fcw - FirecREST Workflow CLI

A command-line tool for orchestrating HPC workflows via [FirecREST](https://github.com/eth-cscs/pyfirecrest).

## Features

- **Container Management**: Build, deploy, and iterate on container images (fast iteration with mirrored, bind-mounted patches and image rebuild when stable)
- **Data Transfer**: Directory mirroring with continuous upload/download using type enforcement (`in`/`out`/`both`)
- **Job Management**: Submit jobs from TOML with SBATCH overrides via `--` separator
- **FUSE Mount**: Mount remote storage as local filesystem over FirecREST (optional, untested)

## Installation

```bash
pip install fcw

# With FUSE support (requires libfuse3-dev)
pip install fcw[fuse]
```

## Quick Start


1. Set up FirecREST credentials:

```bash
export FIRECREST_URL="https://api.cscs.ch/ml/firecrest/v2"
export AUTH_TOKEN_URL="https://auth.cscs.ch/auth/realms/firecrest-clients/protocol/openid-connect/token"
export FIRECREST_SYSTEM="clariden"
export FIRECREST_ACCOUNT="<account>"
export FIRECREST_CLIENT_ID="<client_id>"
export FIRECREST_CLIENT_SECRET="<client_secret>"
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
project: my-app

workdir:
  remote: /scratch/${USER}/my-project
  local: .

# Directory types: in/out/both relative to HPC job (upload/download/both)
directories:
  data/raw:
    type: in
  data/processed:
    type: out
  outputs:
    type: out
  code:
    type: both
  configs:
    type: in

containers:
  app:
    file: ./env/Dockerfile
    tag: my-fcw-app:latest
    remote_path: ce-images/my-fcw-app.sqsh

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
      CONFIG_DIR: configs

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
fcw job submit train --set CONFIG=configs/exp1.yaml --set EPOCHS=100

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
fcw container build --stage download -f env/Dockerfile.prod-multistage --build-arg BASE_IMAGE=ubuntu:24.04 -t my-fcw-app:download .

# Push download image to remote
fcw container push my-fcw-app:download

# Build offline stage on the cluster and import as enroot squashfs
fcw container build-remote my-fcw-app:download \
    -f env/Dockerfile.prod-multistage -t my-fcw-app:latest \
    --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \
    --enroot --wait
```

#### Code Iteration Workflow

For fast iteration without rebuilding the full container:

```bash
# 1. Extract code from container for local editing
fcw container extract my-fcw-app:download /workspace/BrainBERT ./code

# 2. Edit ./code locally...

# 3a. Quick iteration: bind-mount patched code (no rebuild)
fcw container patch ./code /workspace/BrainBERT --toml env/container.toml
# Then: srun --environment env/container.toml python train.py

# 3b. Bake changes: patch + rebuild (when satisfied with changes)
fcw container update ./code my-fcw-app:download /workspace/BrainBERT \
    --tag my-fcw-app:v2 --rebuild --dockerfile env/Dockerfile.prod-multistage \
    --build-arg BASE_IMAGE=ubuntu:24.04 --enroot --wait
```

### FUSE Mount (Optional, Untested)

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

# Upload input data and experiment configs
fcw data upload data/raw
fcw data upload configs

# Build and deploy container (first time)
fcw container build --stage download -t my-fcw-app:download .
fcw container push my-fcw-app:download
fcw container build-remote my-fcw-app:download \
    -f env/Dockerfile.prod-multistage -t my-fcw-app:latest \
    --stage build-offline --build-arg BASE_IMAGE=ubuntu:24.04 \
    --enroot --wait

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
#!/bin/bash
# Fast iteration on code without rebuilding full container

# One-time setup: extract code
fcw container extract my-fcw-app:download /workspace/BrainBERT ./code

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
fcw container update ./code my-fcw-app:download /workspace/BrainBERT \
    --tag my-fcw-app:v2 --rebuild --dockerfile env/Dockerfile.prod-multistage \
    --build-arg BASE_IMAGE=ubuntu:24.04 --enroot --wait
```

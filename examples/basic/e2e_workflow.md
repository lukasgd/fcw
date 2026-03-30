# End-to-End Training Workflow

Full workflow for running the test project on an HPC cluster via fcw + FirecREST.

## Prerequisites

- `fcw` installed (`pip install -e ".[dev]"`)
- Environment variables set: `FIRECREST_URL`, `FIRECREST_CLIENT_ID`, `FIRECREST_CLIENT_SECRET`, `AUTH_TOKEN_URL`, `FIRECREST_SYSTEM`, `FIRECREST_ACCOUNT`
- Container runtime available (podman or docker)
- Working directory: `examples/basic/`

## Step 0: Setup Remote Directories

Create the required remote directory structure before running any jobs:

```bash
fcw job run --time 00:05:00 -- "mkdir -p data/processed outputs logs slurm"
```

Upload env files (container.toml) and slurm scripts:

```bash
fcw data upload env
fcw data upload slurm
```

**Verify:** Remote directories exist and env/container.toml is accessible.

## Step 1: Validate Configuration

```bash
fcw config validate
fcw config show
```

**Verify:** Config resolves correctly, credentials are valid, remote system is reachable. Output should show "All checks passed!".

## Step 2: Container Build & Deploy

### 2a. Build download stage locally

```bash
fcw container build --stage download -t ubuntu-fcw-basic:24.04-download -f env/Dockerfile.app .
```

**Verify:** Local image `ubuntu-fcw-basic:24.04-download` exists (`podman images` or `docker images`).

### 2b. Push to remote

```bash
fcw container push ubuntu-fcw-basic:24.04-download
```

**Verify:** Tar file uploaded to `${workdir.remote}/ce-images/` on the remote system:
```bash
fcw data ls ce-images
```

### 2c. Build offline stage on cluster + enroot import

```bash
fcw container build-remote ubuntu-fcw-basic:24.04-download \
    -f env/Dockerfile.app -t ubuntu-fcw-basic:24.04 \
    --stage build-offline --enroot --wait
```

**Verify:** `ubuntu-fcw-basic+24.04.sqsh` exists at `${workdir.remote}/ce-images/`:
```bash
fcw data ls ce-images
```

## Step 3: Upload Data

```bash
fcw data upload data/raw
```

**Verify:**
```bash
fcw data ls data/raw -R
```
Should show `test.txt` and `test_1.txt` through `test_6.txt` on the remote side.

## Step 4: Submit Jobs

### 4a. Preprocess

```bash
fcw job submit --wait -- preprocess
```

**Verify:** Job completes successfully. `data/processed/preprocessed_files.txt` exists on remote:
```bash
fcw data ls data/processed
```

### 4b. Train

```bash
fcw job submit --wait -- train
```

Resources: 2 nodes, 4 tasks/node, 4 GPUs/node.

**Verify:** Job completes. Output files `train_output_<jobid>_<rank>.txt` exist in `outputs/` (one per rank, 8 total):
```bash
fcw data ls outputs
```

### 4c. Evaluate

```bash
fcw job submit --wait -- evaluate
```

**Verify:** Job completes. `eval_summary_<jobid>.txt` exists in `outputs/`:
```bash
fcw data ls outputs
```

## Step 5: Download Results

```bash
fcw data download outputs
```

**Verify:** Local `outputs/` directory contains:
- `train_output_*_*.txt` (8 files, one per rank)
- `eval_summary_*.txt` (1 file)

## Step 6: Review Logs

```bash
fcw job list
fcw job logs <job_id>
```

**Verify:** All three jobs (preprocess, train, evaluate) show COMPLETED state.

## Notes

- All SLURM scripts use `srun --environment ./env/container.toml` which requires the enroot squashfs image from Step 2.
- The `env/container.toml` references `${CE_IMAGES_DIR}` — this must be set via job env vars in `fcw.yaml` (e.g., `CE_IMAGES_DIR: ce-images`).
- The `logs/`, `data/processed/`, and `outputs/` directories must exist on the remote workdir before job submission (Step 0).
- Job environment variables (`DATA_IN`, `DATA_OUT`, `DATA_DIR`, `OUTPUT_DIR`, `MODEL_DIR`, `CE_IMAGES_DIR`) are resolved from `fcw.yaml` job definitions and injected into the submitted script.
- Directory type enforcement: `data/raw` is `in` (upload only), `data/processed` and `outputs` are `out` (download only), `code` is `both`.
- Global options (`--system`, `--account`) can be set via environment variables (`FIRECREST_SYSTEM`, `FIRECREST_ACCOUNT`) instead of passing them on every command.

## System-Specific Notes

### lys

- **`--remote-script` required**: slurmrestd inline script submission + pyxis SPANK plugin causes `srun --environment` to segfault. Use `fcw job submit --remote-script` to upload the script to the remote filesystem first, then submit with `script_remote_path`. This is a workaround — inline scripts work on other systems.
- **`ignore_chown_errors` for podman**: The `firecr02` service account lacks subuid/subgid entries, so podman overlay storage needs `ignore_chown_errors = "true"` in `[storage.options.overlay]`. This is already configured in the generated storage.conf template.
- **Pyxis plugin**: Container support uses the pyxis SPANK plugin (`/usr/lib64/slurm/spank_pyxis.so`), not native SLURM `JobContainerType`. The `--environment` flag on `srun` is a pyxis option.

## Bugs Found & Fixed During E2E

1. **`config show`** crashed on `job_config.after` attribute that doesn't exist on `JobConfig` — removed the "After" column.
2. **`config validate`** had leftover `debugpy` code that hung waiting for debugger — removed.
3. **`build-remote` script** — `podman system reset -f` failed before `XDG_RUNTIME_DIR` was set up. Fix: set up XDG_RUNTIME_DIR before podman commands, re-create after cleanup.
4. **`build-remote` script** — podman couldn't resolve short image names loaded via `podman load`. Fix: use image ID (from `podman image inspect`) as `DOWNLOAD_IMAGE` build-arg, and `--pull=never`.
5. **`fcw.yaml`** — jobs using `srun --environment ./env/container.toml` need `CE_IMAGES_DIR` in their env config.

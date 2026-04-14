# E2E Test Suite

Automated end-to-end tests covering the full fcw HPC workflow: config validation, container build/deploy/iterate, data transfer, SLURM job submission, and cleanup.

Tests run sequentially and form a dependency chain — each phase builds on the previous one.

## Prerequisites

- `fcw` installed: `pip install -e ".[dev]"`
- Container runtime: podman or docker
- Environment variables set:

  | Variable | Description |
  |----------|-------------|
  | `FIRECREST_URL` | FirecREST API endpoint |
  | `FIRECREST_CLIENT_ID` | OAuth client ID |
  | `FIRECREST_CLIENT_SECRET` | OAuth client secret |
  | `AUTH_TOKEN_URL` | OAuth token endpoint |
  | `FIRECREST_SYSTEM` | Target cluster |
  | `FIRECREST_ACCOUNT` | SLURM account |
  | `FIRECREST_SCRATCH` | Remote scratch path (e.g., `/iopsstor/scratch/cscs/$USER`) |

## Running the Tests

```bash
# Full suite (from repo root)
pytest tests/ --run-e2e -v

# Or via environment variable
FCW_E2E=1 pytest tests/ -v

# Specific example project (default: basic)
pytest tests/ --run-e2e --example basic -v

# Clean up remote workdir after a successful run
pytest tests/ --run-e2e --cleanup-remote -v
```

Each run creates a fresh remote directory `${FIRECREST_SCRATCH}/fcw-basic-<uuid>`. To reuse an existing one (e.g., to re-run after a partial failure):

```bash
FCW_BASIC_RUN_ID=<hex-id> pytest tests/ --run-e2e -v
```

The run ID is printed at the start of each run. If tests fail, the output includes a ready-to-copy re-run command.

## Test Inventory (36 tests)

### 1. Config Validation (2 tests)

| Test | Command | Verifies |
|------|---------|----------|
| `test_config_validate` | `fcw config validate` | FirecREST credentials are valid and API is reachable. |
| `test_config_show` | `fcw config show` | Resolved config displays correctly (env vars expanded). |

### 2. Remote Setup (3 tests)

| Test | Command | Verifies |
|------|---------|----------|
| `test_setup_remote_dirs` | *(direct API: `client.mkdir`)* | Creates remote directory tree: `data/raw`, `data/processed`, `outputs`, `logs`, `env`, `ce-images`. |
| `test_upload_env` | `fcw data upload env` | Uploads `env/` (Dockerfiles, container.toml) — needed by SLURM jobs referencing `./env/container.toml`. |
| `test_upload_slurm` | `fcw data upload slurm` | Uploads `slurm/` (job scripts) — needed for `--remote-script` submission. |

### 3. Multi-Stage Container Build (6 tests)

Tests the advanced workflow: build download stage locally, push tar to cluster, build offline stage on cluster, enroot import to sqsh.

| Test | Command | Verifies |
|------|---------|----------|
| `test_container_build_local` | `fcw container build --stage download -t ubuntu-fcw-basic:24.04-download -f env/Dockerfile.app .` | Builds the `download` stage locally (has network access, copies project into image). |
| `test_container_build_save` | `fcw container build --stage download ... --save test-image.tar .` | Builds and exports image to tar file. Confirms tar exists, then cleans up. |
| `test_container_push` | `fcw container push ubuntu-fcw-basic:24.04-download` | Exports image to tar, uploads to `ce-images/` on remote via FirecREST. |
| `test_container_build_remote` | `fcw container build-remote ... --stage build-offline --enroot --wait` | SLURM job: loads pushed tar into podman, builds offline stage, enroot imports to `ce-images/ubuntu-fcw-basic+24.04.sqsh`. |
| `test_verify_sqsh` | `fcw data ls ce-images` | Sqsh file exists on remote — multi-stage build produced the expected artifact. |
| `test_container_list_local` | `fcw container list` | Lists local container images. |

### 4. Single-Command Deploy (3 tests)

Tests the all-in-one workflow: build + push + enroot import in one command.

| Test | Command | Verifies |
|------|---------|----------|
| `test_container_deploy` | `fcw container deploy aux --wait` | Looks up `aux` container in config, builds download stage locally, pushes tar, submits SLURM job to build offline stage + enroot import. Produces `ce-images/fcw-aux+latest.sqsh`. |
| `test_verify_deploy` | `fcw data ls ce-images` | Sqsh file exists on remote. |
| `test_container_list_remote` | `fcw container list --remote` | Lists remote sqsh files across all configured `remote_path` directories. |

### 5. Container Iteration (2 tests)

Tests the code iteration workflow: extract code from container, modify locally, push back as bind-mount overlay.

| Test | Command | Verifies |
|------|---------|----------|
| `test_container_extract` | `fcw container extract aux /workspace/aux extracted-code --wait` | SLURM job: loads the stage image, `podman cp`s `/workspace/aux` out, archives it. Downloads and extracts locally. Writes sidecar `extracted-code.meta.json`. |
| `test_container_patch` | `fcw container patch --container app extracted-code` (or `data/raw:/workspace`) | Uploads the dump; mount target comes from the sidecar or from `<local>:<container>` override. Adds a `.patches/` entry to `containers.app.toml`. |

### 6. Data Upload (4 tests)

| Test | Command | Verifies |
|------|---------|----------|
| `test_upload_data` | `fcw data upload data/raw` | Uploads test data files. Respects directory type enforcement (`data/raw` is type `in`). |
| `test_verify_data` | `fcw data ls data/raw -R` | `test.txt` visible on remote. |
| `test_upload_incremental` | `fcw data upload --incremental data/raw` | Re-upload skips unchanged files (uses `.fcw/sync/` timestamp markers). |
| `test_data_status` | `fcw data status` | Reports sync status for all configured directories. |

### 7. Job Submission (7 tests)

All jobs use `--remote-script` (uploads script before sbatch) and `--wait` (polls until complete, checks final SLURM state).

| Test | Command | Verifies |
|------|---------|----------|
| `test_submit_preprocess` | `fcw job submit --remote-script --wait -- preprocess` | Runs inside enroot container. Cats `data/raw/*` into `data/processed/preprocessed_files.txt`. |
| `test_verify_preprocess` | `fcw data ls data/processed` | `preprocessed_files.txt` exists. |
| `test_submit_train` | `fcw job submit --remote-script --wait -- train` | Distributed: 2 nodes, 4 tasks/node. Each rank writes `train_output_<jobid>_rank<N>.txt` to `outputs/`. |
| `test_verify_train` | `fcw data ls outputs` | `train_output_` files exist. |
| `test_submit_evaluate` | `fcw job submit --remote-script --wait -- evaluate` | Lists model dir, writes `eval_summary_<jobid>.txt` to `outputs/`. |
| `test_verify_evaluate` | `fcw data ls outputs` | `eval_summary_` exists. |
| `test_submit_with_env_override` | `fcw job submit --remote-script --wait --set DATA_OUT=outputs -- preprocess` | Overrides `DATA_OUT` to redirect output to `outputs/`. Relative path resolved to absolute. |

### 8. Job Management (5 tests)

| Test | Command | Verifies |
|------|---------|----------|
| `test_job_list` | `fcw job list` | Lists recent SLURM jobs in a table. |
| `test_job_run_and_wait` | `fcw job run --remote-script -- echo hello` + `fcw job wait <id>` | Submits ad-hoc command, then waits for it via separate `job wait`. |
| `test_job_status` | `fcw job status <id>` | Queries status of the completed ad-hoc job. |
| `test_job_logs` | `fcw job logs <id>` | Retrieves stdout of the completed ad-hoc job. |
| `test_job_cancel` | `fcw job run ... -- sleep 600` + `fcw job cancel <id>` | Submits a long-running job, then cancels it. |

### 9. Data Download (2 tests)

| Test | Command | Verifies |
|------|---------|----------|
| `test_download_outputs` | `fcw data download outputs` | Downloads `outputs/` from remote. Respects directory type enforcement (`outputs` is type `out`). |
| `test_download_incremental` | `fcw data download --incremental outputs` | Re-download skips unchanged files. |

### 10. Cleanup (2 tests)

| Test | Command | Verifies |
|------|---------|----------|
| `test_data_rm` | `fcw data rm --force data/processed` | Deletes remote directory. `--force` skips confirmation. |
| `test_verify_rm` | `fcw data ls data/processed` | Directory no longer exists or is empty. |

## Test Dependencies

```
Setup dirs (3) --> Upload env/slurm (4,5)
                        |
    Build local (6) --> Push (8) --> Build remote (9) --> Verify sqsh (10)
                                                              |
    Deploy aux (12) --> Verify deploy (13)           All jobs (21-27)
         |                                                    |
    Extract (15) --> Patch (16)                         Verify outputs (22,24,26)
                                                              |
                                                         Download (33,34)
                                                              |
                                                         Cleanup (35,36)
```

**Failure cascade:** If push (8) fails, build-remote (9) has no base image, no sqsh is created, and all container-based SLURM jobs (21-27) fail. If env/slurm upload (4,5) fails, `--remote-script` jobs can't find the scripts.

## Fixtures (conftest.py)

| Fixture | Scope | Description |
|---------|-------|-------------|
| `example_workdir` | session | Copies `examples/basic/` to temp dir, sets unique run ID, changes CWD. Optionally cleans up remote dir after success. |
| `fcw_config` | session | Loaded `FcwConfig` from the temp workdir. |
| `system` | session | Target system from `FIRECREST_SYSTEM`. |
| `account` | session | SLURM account from `FIRECREST_ACCOUNT`. |
| `client` | session | Sync FirecREST v2 client. |
| `remote_workdir` | session | Resolved remote path (e.g., `/iopsstor/scratch/cscs/user/fcw-basic-<id>`). |
| `runner` | function | Typer `CliRunner` for invoking CLI commands. |
| `shared_state` | session | Dict for passing state (e.g., job IDs) between tests. |

## Debugging a Failed Run

1. Note the run ID from the test output: `E2E: FCW_BASIC_RUN_ID=<hex>`
2. Re-run with the same ID: `FCW_BASIC_RUN_ID=<hex> pytest tests/ --run-e2e -v`
3. Check SLURM job logs on remote: `fcw job logs <job_id>`
4. List remote files: `fcw data ls <path> -R`
5. Check job state: `fcw job status <job_id>`

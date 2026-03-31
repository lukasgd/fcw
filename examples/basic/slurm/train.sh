#!/bin/bash -l

#SBATCH --job-name train
#SBATCH --time 12:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 4
#SBATCH --gpus-per-node 4

set -euxo pipefail

srun -ul --environment ${FCW_CONTAINER_TOML} bash -c "
    echo 'Rank \${SLURM_PROCID} on node \${SLURM_NODEID}'
    wc ${DATA_DIR}/* > ${OUTPUT_DIR}/train_output_\${SLURM_JOB_ID}_\${SLURM_PROCID}.txt
"
#!/bin/bash -l

#SBATCH --job-name evaluate
#SBATCH --time 12:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 288
#SBATCH --gpus-per-node 4

set -euxo pipefail

srun -ul --environment ./env/container.toml bash -c "
    echo 'Evaluating model outputs in ${MODEL_DIR}'
    ls -lh ${MODEL_DIR}/
    wc -l ${MODEL_DIR}/* > ${MODEL_DIR}/eval_summary_\${SLURM_JOB_ID}.txt
    echo 'Evaluation complete'
"

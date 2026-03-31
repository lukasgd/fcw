#!/bin/bash -l

#SBATCH --job-name preprocess
#SBATCH --time 12:00:00
#SBATCH --output logs/%x-%j.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 288
#SBATCH --gpus-per-node 4

set -euxo pipefail

srun -ul --environment ${FCW_CONTAINER_TOML} bash -c "
    cat ${DATA_IN}/* > ${DATA_OUT}/preprocessed_files.txt
"
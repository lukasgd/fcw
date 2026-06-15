#!/bin/bash -l

#SBATCH --job-name preprocess
#SBATCH --time 00:15:00
#SBATCH --output logs/%x-%j.out
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 288
#SBATCH --gpus-per-node 4

set -euxo pipefail

mkdir -p ${DATA_OUT}

srun -ul --environment ${FCW_CONTAINER_TOML} bash -c "
    cat ${DATA_IN}/*.txt > ${DATA_OUT}/preprocessed_files.txt
"
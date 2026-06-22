#!/bin/bash -l
#SBATCH --job-name node-burn
#SBATCH --nodes 4
#SBATCH --ntasks-per-node 4
#SBATCH --gres=gpu:4
#SBATCH --output logs/%x-%j.out
#SBATCH --time=1:00:00

set -euxo pipefail

hostname

srun -l --environment ${FCW_CONTAINER_TOML} burn -cgemm,1000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -cgemm,3000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -cgemm,10000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -cgemm,30000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -ggemm,1000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -ggemm,3000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -ggemm,10000 -d10 --batch

srun -l --environment ${FCW_CONTAINER_TOML} burn -ggemm,30000 -d10 --batch


#!/bin/bash

# Parameters
#SBATCH --cpus-per-gpu=24
#SBATCH --error=/gpfs/mskmind_ess/boehmk/data-prep/ncit/slurm.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --job-name=run
#SBATCH --mem=96GB
#SBATCH --nodes=1
#SBATCH --open-mode=append
#SBATCH --output=/gpfs/mskmind_ess/boehmk/data-prep/ncit/slurm.out
#SBATCH --signal=USR2@120
#SBATCH --time=1200
#SBATCH --priority=HIGH

# command
srun --unbuffered --output /gpfs/mskmind_ess/boehmk/data-prep/ncit/slurm.out --error /gpfs/mskmind_ess/boehmk/data-prep/ncit/slurm.err /gpfs/mskmind_ess/boehmk/miniforge3/envs/pykeen/bin/python /gpfs/mskmind_ess/boehmk/data-prep/ncit/main.py
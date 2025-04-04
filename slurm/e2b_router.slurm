#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --requeue

echo "Starting job"
set -x -e

source ~/.bashrc
source openr1/bin/activate

srun python scripts/e2b_router.py
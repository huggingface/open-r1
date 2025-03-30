#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH --mem=16g
#SBATCH --time=72:00:00
#SBATCH --cpus-per-task=16
#SBATCH -o logs/dataset_proc-%j-%x.out
#SBTACH --requeue

echo "Starting job"
module load cuda/12.4
set -x -e

source ~/.bashrc
source openr1/bin/activate

srun python scripts/e2b_router.py
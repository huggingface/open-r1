#!/bin/bash

#SBATCH --partition=hopper-cpu
#SBATCH --mem=16g
#SBATCH --cpus-per-task=16
#SBATCH --output=/fsx/open-r1/logs/morph_router/%x-%j.out
#SBATCH --err=/fsx/open-r1/logs/morph_router/%x-%j.err
#SBATCH --requeue
#SBATCH --time=7-00:00:00


echo "Starting job"
set -x -e

source ~/.bashrc
source openr1/bin/activate

srun python scripts/morph_router.py --port 8001 --max_num_sandboxes 20

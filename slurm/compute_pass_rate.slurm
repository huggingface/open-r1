#!/bin/bash

#SBATCH --job-name=open-r1-compute-pass-rate
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=01-00:00:00
#SBATCH --requeue

# example usage: sbatch slurm/dataset_filter.slurm recipes/dataset_filtering/filter_dapo.yaml 0 500

set -x -e

source ~/.bashrc
source openr1/bin/activate

python scripts/pass_rate_filtering/compute_pass_rate.py --config $1 --dataset_start_index $2 --dataset_end_index $3
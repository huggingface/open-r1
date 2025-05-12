#!/bin/bash

#SBATCH --job-name=open-r1-dataset-filtering
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --time=02-00:00:00

echo "Starting job"
set -x -e

source ~/.bashrc
source openr1/bin/activate

python scripts/filter_dataset.py --config $1
#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=/fsx/h4/logs/%x-%j.out
#SBATCH --err=/fsx/h4/logs/%x-%j.err
#SBATCH --requeue

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!


# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.4


set -x -e

source ~/.bashrc
source openr1/bin/activate
echo "START TIME: $(date)"

accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml --num_processes=8 scripts/remote_grpo.py \
    --config recipes/SmolLM2-1.7B-Instruct/grpo/config_v05.19.yaml \
    --push_to_hub false \
    --output_dir data/remote-grpo
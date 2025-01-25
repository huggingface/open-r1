#!/bin/bash
#SBATCH --job-name=open-r1-evaluate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod 
#SBATCH --time=01:59:00
#SBATCH --output=./logs/evaluate/%x-%j.out
#SBATCH --err=./logs/evaluate/%x-%j.err

# Usage: sbatch slurm/evaluate.slurm deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B aime24

set -x -e

source ~/.bashrc
conda activate openr1
module load cuda/12.1
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"


NUM_GPUS=8
MODEL=$1
TASK=$2
MODEL_ARGS="pretrained=$MODEL,dtype=float16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
OUTPUT_DIR=data/evals/$MODEL


# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

# Specific configuration optimized for the Hugging Face Compute Cluster
# Be ye warned this may not work on other clusters!
module load cuda/12.1

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
    --output-dir $OUTPUT_DIR 


echo "END TIME: $(date)"

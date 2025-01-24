#!/bin/bash
#SBATCH --job-name=default
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod 
#SBATCH --qos=high
#SBATCH --time=01:59:00
#SBATCH --output=/fsx/elie_bakouch/open-r1/logs/%x-%j.out 
#SBATCH --err=/fsx/elie_bakouch/open-r1/logs/%x-%j.err

set -x -e

# DON'T FORGET TO MODIFY THE HEADER WITH YOUR PATH 
# @todo:eliebak maybe with use submitit at some point

source ~/.bashrc
conda activate openr1
module load cuda/12.1
echo "START TIME: $(date)"
echo "PYTHON ENV: $(which python)"

MODEL=Qwen2.5-1.5B-Instruct
TASK=sft
PRECISION=v00.00
ACCELERATOR=deepspeed_zero3

# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
# Due to conflicts between Accelerate's DeepSpeed configs and Transformers' TrainingArguments, we need to parse the gradient accumulation steps from the config file to ensure they match
CONFIG_FILE=recipes/$MODEL/$TASK/config_$PRECISION.yaml

echo "CONFIG_FILE: $CONFIG_FILE"
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')


# Loop through the arguments and find the one with "--gradient_accumulation_steps"
for arg in "${ARGS[@]}"; do
    if [[ "$arg" == "--gradient_accumulation_steps="* ]]; then
        # Extract the value after the equals sign
        GRAD_ACC_STEPS="${arg#*=}"
        break  # Exit the loop once we find the desired argument
    fi
done

echo "Gradient accumulation steps: $GRAD_ACC_STEPS"
# so processes know who to talk to
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
MASTER_PORT=6000

export CMD=" \
    scripts/training/$TASK.py $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "

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

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
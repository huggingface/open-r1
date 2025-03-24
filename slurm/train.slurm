#!/bin/bash
#SBATCH --job-name=open-r1-sft
#SBATCH --ntasks-per-node=1
#SBATCH --exclusive
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod  # Adjust this for your cluster
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --requeue

# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/12.4
set -x -e

source ~/.bashrc
source openr1/bin/activate
echo "START TIME: $(date)"

MODEL=$1
TASK=$2
CONFIG_SUFFIX=$3
ACCELERATOR=$4
OPTIONAL_ARGS=$5
CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')
MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Distributed configuration
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
NODELIST=($(scontrol show hostnames $SLURM_JOB_NODELIST))
MASTER_ADDR=${NODELIST[0]}  # First node for main process
MASTER_PORT=6000
TRAIN_NODES=("${NODELIST[@]}")

USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
fi
# if using vllm
if [[ "$USE_VLLM" == "true" ]]; then
     TRAIN_NODES=("${NODELIST[@]:0:$((NUM_NODES - 1))}")
     VLLM_NODE=${NODELIST[-1]} # Last node
     TP=$(python scripts/get_tensor_parallel_size.py --model_name $MODEL --revision $REVISION --default_tp $GPUS_PER_NODE)
     WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
     NUM_NODES=$((NUM_NODES - 1))
     srun --nodes=1 --ntasks=1 --nodelist=$VLLM_NODE trl vllm-serve --model $MODEL --revision $REVISION --tensor_parallel_size $TP &

     OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host=$VLLM_NODE"
fi

# force crashing on nccl issues like hanging broadcast
export NCCL_ASYNC_ERROR_HANDLING=1
# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=COLL
# export NCCL_SOCKET_NTHREADS=1
# export NCCL_NSOCKS_PERTHREAD=1
# export CUDA_LAUNCH_BLOCKING=1

export CMD=" \
    src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS
    "

export LAUNCHER="HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info TRANSFORMERS_VERBOSITY=info accelerate launch \
    --config_file recipes/accelerate_configs/$ACCELERATOR.yaml  \
    --gradient_accumulation_steps $GRAD_ACC_STEPS \
    --num_machines $NUM_NODES \
    --num_processes $WORLD_SIZE \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank $SLURM_PROCID \
    --rdzv_backend=c10d \
    --max_restarts 1 \
    --role \$(hostname -s): \
    --tee 3 \
    "
# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --nodes=$NUM_NODES \
    --ntasks=$NUM_NODES \
    --nodelist=$TRAIN_NODES
    "
clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

echo "END TIME: $(date)"
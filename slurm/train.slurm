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
# Be ye warned this may not work on other clusters!
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

# Training setup
NUM_NODES=$SLURM_NNODES
GPUS_PER_NODE=8
WORLD_SIZE=$(($NUM_NODES*$GPUS_PER_NODE))
# Due to conflicts between Accelerate's DeepSpeed configs and Transformers' TrainingArguments, we need to parse the gradient accumulation steps from the config file to ensure they match
CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml
GRAD_ACC_STEPS=$(grep 'gradient_accumulation_steps' $CONFIG_FILE | awk '{print $2}')

# Split the string into individual arguments
IFS=' ' read -ra ARGS <<< "$OPTIONAL_ARGS"

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

# Get all hostnames in allocation
HOSTNAMES=($(scontrol show hostnames $SLURM_JOB_NODELIST))
NUM_HOSTS=${#HOSTNAMES[@]}
LAST_NODE=${HOSTNAMES[-1]}
LAST_NODE_IP=$(getent hosts $LAST_NODE | awk '{ print $1 }')

# Default: all nodes are training nodes
TRAIN_NODES=("${HOSTNAMES[@]}")

# Default to not using vLLM but override if the config file specifies it
USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
fi

if [[ "$USE_VLLM" == "true" ]]; then
    # We allocate 1 node for vLLM inference, so we need to subtract that from the training nodes
    WORLD_SIZE=$((WORLD_SIZE - GPUS_PER_NODE))
    NUM_NODES=$((NUM_NODES - 1))
    MODEL=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
    REVISION=$(grep 'model_revision:' $CONFIG_FILE | awk '{print $2}')

    # Some models like Qwen have a number of attention heads that is not divisible by the default tensor parallel size, so we need to adjust that
    TP=$(python scripts/get_tensor_parallel_size.py --model_name $MODEL --revision $REVISION --default_tp $GPUS_PER_NODE)


    OPTIONAL_ARGS="$OPTIONAL_ARGS --vllm_server_host=$LAST_NODE_IP"

    # Launch vLLM server in background on the last node
    srun --nodes=1 \
        --nodelist=$LAST_NODE \
        --gres=gpu:$GPUS_PER_NODE \
        --exclusive \
        --ntasks=1 \
        --job-name=${SLURM_JOB_NAME}-vllm-server \
        bash -c "source ~/.bashrc && source openr1/bin/activate && trl vllm-serve --model $MODEL --revision $REVISION --tensor_parallel_size $TP --host $LAST_NODE_IP" &

    # Add a wait for the vLLM server to initialize with health checks
    echo "Waiting for vLLM server to initialize at http://$LAST_NODE_IP:8000..."
    TIMEOUT=900  # 15 minutes
    START_TIME=$(date +%s)

    while true; do
        if curl -s -o /dev/null -w "%{http_code}" "http://$LAST_NODE_IP:8000/health" >/dev/null 2>&1; then
            echo "vLLM server is ready at http://$LAST_NODE_IP:8000"
            break
        fi

        CURRENT_TIME=$(date +%s)
        if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
            echo "Error: vLLM server failed to start within $TIMEOUT seconds"
            exit 1
        fi

        echo "Still waiting for vLLM server to initialize... ($(($CURRENT_TIME - $START_TIME)) seconds elapsed)"
        sleep 30
    done

    # Remove last node from training
    TRAIN_NODES=("${HOSTNAMES[@]:0:$((NUM_HOSTS - 1))}")
fi

echo "Training nodes: ${TRAIN_NODES[@]}"


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

# srun error handling:
# --wait=60: wait 60 sec after the first task terminates before terminating all remaining tasks
# --kill-on-bad-exit=1: terminate a step if any task exits with a non-zero exit code
SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    "

# clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1

clear; srun $SRUN_ARGS --jobid $SLURM_JOB_ID \
    --overlap \
    --nodelist=$(IFS=, ; echo "${TRAIN_NODES[*]}") \
    bash -c "$LAUNCHER --role \$SLURMD_NODENAME: $CMD" 2>&1


echo "END TIME: $(date)"
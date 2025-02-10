#!/bin/bash
#SBATCH --job-name=r1-vllm
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1

set -exuo pipefail

MODEL_PATH="deepseek-ai/DeepSeek-R1"
CONDA_ENV="vllm7"
SERVER_PORT=8000
RAY_PORT=6379
RAY_DASHBOARD_PORT=8265

while getopts "m:e:h" opt; do
    case $opt in
        m) MODEL_PATH="$OPTARG" ;;
        e) CONDA_ENV="$OPTARG" ;;
        h|?) echo "Usage: sbatch $0 [-m MODEL_PATH] [-e CONDA_ENV]"; exit 1 ;;
    esac
done

# Environment setup
module load cuda/12.1
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate conda env $CONDA_ENV"; exit 1; }

# Get nodes information
NODES=($(scontrol show hostnames "$SLURM_JOB_NODELIST"))
HEAD_NODE="${NODES[0]}"
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "Head node: $HEAD_NODE ($HEAD_NODE_IP)"

# Start Ray head node
echo "Starting Ray head node at $HEAD_NODE"
srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" \
    ray start --head \
    --node-ip-address="$HEAD_NODE_IP" \
    --port=$RAY_PORT \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=$RAY_DASHBOARD_PORT \
    --block &

sleep 10

# Start Ray worker nodes
WORKER_COUNT=$((SLURM_JOB_NUM_NODES - 1))
for ((i = 1; i <= WORKER_COUNT; i++)); do
    WORKER_NODE="${NODES[$i]}"
    echo "Starting Ray worker $i at $WORKER_NODE"
    srun --nodes=1 --ntasks=1 -w "$WORKER_NODE" \
        ray start --address "$HEAD_NODE_IP:$RAY_PORT" \
        --block &
    sleep 5
done

echo "Waiting for Ray cluster to initialize..."
sleep 60

# Start vLLM server
echo "Starting vLLM server..."
RAY_ADDRESS="http://$HEAD_NODE_IP:$RAY_DASHBOARD_PORT" ray job submit \
    --working-dir src/open_r1 \
    --no-wait \
    --job-id vllm-server \
    -- vllm serve "$MODEL_PATH" \
        --tensor-parallel-size 8 \
        --pipeline-parallel-size 4 \
        --gpu-memory-utilization 0.90 \
        --max-model-len 32768 \
        --max-num-batched-tokens 262144 \
        --max-num-seqs 128 \
        --max-seq-len-to-capture 32768 \
        --enable-chunked-prefill true \
        --preemption-mode recompute \
        --swap-space 128 \
        --trust-remote-code \
        --distributed-executor-backend ray

# Wait for server with timeout
TIMEOUT=3600  # 1h
START_TIME=$(date +%s)
echo "Waiting for vLLM server (http://$HEAD_NODE_IP:$SERVER_PORT)..."

while true; do
    if curl -s -o /dev/null -w "%{http_code}" "http://$HEAD_NODE_IP:$SERVER_PORT/health" >/dev/null 2>&1; then
        echo "Server is ready at http://$HEAD_NODE_IP:$SERVER_PORT"
        break
    fi

    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
        echo "Error: Server failed to start within $TIMEOUT seconds"
        exit 1
    fi

    echo "Still waiting... ($(($CURRENT_TIME - $START_TIME)) seconds elapsed)"
    sleep 60
done

echo "Checking available models..."
curl "http://$HEAD_NODE_IP:$SERVER_PORT/v1/models"
sleep 10

echo "Executing sanity check..."
curl "http://$HEAD_NODE_IP:$SERVER_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"default\",
        \"prompt\": \"<｜begin▁of▁sentence｜><｜User｜>hi, how are you?<｜Assistant｜>\",
        \"max_tokens\": 2048,
        \"temperature\": 0.6
    }"

# Keep the job running with health checks
while true; do
    if ! curl -s -o /dev/null "http://$HEAD_NODE_IP:$SERVER_PORT/health"; then
        echo "Error: Server health check failed"
        exit 1
    fi
    sleep 300
done
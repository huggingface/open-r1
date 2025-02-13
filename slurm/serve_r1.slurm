#!/bin/bash
#SBATCH --job-name=r1-server
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1

set -exuo pipefail

MODEL_PATH="deepseek-ai/DeepSeek-R1"
CONDA_ENV="sglang124"
ROUTER_ADDRESS=""
SERVER_PORT=39877
DIST_PORT=45000

# TODO: Adjust these variables to your cluster configuration
export OUTLINES_CACHE_DIR=/scratch/serve_r1/ocache/
export TRITON_HOME=/scratch/serve_r1/triton/
export GLOO_SOCKET_IFNAME="enp71s0"
export NCCL_SOCKET_IFNAME="enp71s0"

while getopts "m:e:r:h" opt; do
    case $opt in
        m) MODEL_PATH="$OPTARG" ;;
        e) CONDA_ENV="$OPTARG" ;;
        r) ROUTER_ADDRESS="$OPTARG" ;;
        h|?) echo "Usage: sbatch $0 [-m MODEL_PATH] [-e CONDA_ENV] [-r ROUTER_ADDRESS]"; exit 1 ;;
    esac
done

# TODO: Environment setup, adjust to your cluster configuration
module load cuda/12.4
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate conda env $CONDA_ENV"; exit 1; }

FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
FIRST_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$FIRST_NODE" hostname --ip-address)

# Launch servers synchronously across all nodes
# (--max-running-requests=56 is rough estimate to avoid too many evicted/preempted 16k-long requests)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    bash -c "python -m sglang.launch_server \
        --model-path '$MODEL_PATH' \
        --tp 16 \
        --dist-init-addr '$FIRST_NODE_IP:$DIST_PORT' \
        --nnodes 2 \
        --node-rank \$SLURM_PROCID \
        --port '$SERVER_PORT' \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-running-requests 56 \
        --context-length 32768" &

# Wait for server with timeout
TIMEOUT=3600  # 1h, but model loading should take ~30min
START_TIME=$(date +%s)
echo "Waiting for SGLang server (http://$FIRST_NODE_IP:$SERVER_PORT)..."

while true; do
    if curl -s -o /dev/null -w "%{http_code}" "http://$FIRST_NODE_IP:$SERVER_PORT/health" >/dev/null 2>&1; then
        echo "Server is ready at http://$FIRST_NODE_IP:$SERVER_PORT"
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

# Register with router only if address was provided
if [ -n "$ROUTER_ADDRESS" ]; then
    echo "Registering with router at $ROUTER_ADDRESS..."
    curl -X POST "http://$ROUTER_ADDRESS/add_worker?url=http://$FIRST_NODE_IP:$SERVER_PORT" || true
    sleep 10
fi

echo "Checking available models..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/models"
sleep 10

echo "Executing sanity check..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"default\",
        \"prompt\": \"<｜begin▁of▁sentence｜><｜User｜>hi, how are you?<｜Assistant｜>\",
        \"max_tokens\": 2048,
        \"temperature\": 0.6
    }"

# Keep the job running with health checks
while true; do
    if ! curl -s -o /dev/null "http://$FIRST_NODE_IP:$SERVER_PORT/health"; then
        echo "Error: Server health check failed"
        exit 1
    fi
    sleep 300
done
#!/bin/bash
#SBATCH --job-name=deepseek-r1-generation
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --gpus-per-node=8
#SBATCH --output=./logs/%x-%j.out
#SBATCH --err=./logs/%x-%j.err
#SBATCH --time=08:00:00

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --hf-dataset)
            HF_DATASET="$2"
            shift 2
            ;;
        --hf-dataset-config)
            HF_DATASET_CONFIG="$2"
            shift 2
            ;;
        --hf-dataset-split)
            HF_DATASET_SPLIT="$2"
            shift 2
            ;;
        --prompt-column)
            PROMPT_COLUMN="$2"
            shift 2
            ;;
        --model)
            MODEL="$2"
            shift 2
            ;;
        --temperature)
            TEMPERATURE="$2"
            shift 2
            ;;
        --top-p)
            TOP_P="$2"
            shift 2
            ;;
        --max-new-tokens)
            MAX_NEW_TOKENS="$2"
            shift 2
            ;;
        --num-generations)
            NUM_GENERATIONS="$2"
            shift 2
            ;;
        --hf-output-dataset)
            HF_OUTPUT_DATASET="$2"
            shift 2
            ;;
        --private)
            PRIVATE="true"
            shift
            ;;
        *)
            echo "Unknown parameter: $1"
            exit 1
            ;;
    esac
done

if [ -z "$MODEL" ] || [ -z "$HF_DATASET" ]; then
    echo "Error: --model and --hf-dataset are required parameters"
    exit 1
fi

# Set default values for optional parameters
HF_DATASET_SPLIT=${HF_DATASET_SPLIT:-"train"}
PROMPT_COLUMN=${PROMPT_COLUMN:-"prompt"}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-8192}
NUM_GENERATIONS=${NUM_GENERATIONS:-1}
PRIVATE=${PRIVATE:-"false"}

# Print all input arguments
echo "Input arguments:"
echo "MODEL: $MODEL"
echo "HF_DATASET: $HF_DATASET"
echo "HF_DATASET_CONFIG: $HF_DATASET_CONFIG"
echo "HF_DATASET_SPLIT: $HF_DATASET_SPLIT"
echo "PROMPT_COLUMN: $PROMPT_COLUMN"
echo "TEMPERATURE: $TEMPERATURE"
echo "TOP_P: $TOP_P"
echo "MAX_NEW_TOKENS: $MAX_NEW_TOKENS"
echo "NUM_GENERATIONS: $NUM_GENERATIONS"
echo "HF_OUTPUT_DATASET: $HF_OUTPUT_DATASET"
echo "PRIVATE: $PRIVATE"
echo "-------------------"

set -ex

module load cuda/12.1

export LD_LIBRARY_PATH=.venv/lib/python3.11/site-packages/nvidia/nvjitlink/lib

echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"

source .venv/bin/activate

# Getting the node names
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# Get the IP address of the head node
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

# Start Ray head node
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "Starting HEAD at $head_node"
srun --nodes=1 --ntasks=1 -w "$head_node" \
    ray start --head --node-ip-address="$head_node_ip" --port=$port \
    --dashboard-host=0.0.0.0 \
    --dashboard-port=8265 \
    --block &

# Give some time to head node to start...
sleep 10

# Start Ray worker nodes
worker_num=$((SLURM_JOB_NUM_NODES - 1))

# Start from 1 (0 is head node)
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun --nodes=1 --ntasks=1 -w "$node_i" \
        ray start --address "$ip_head" \
        --block &
    sleep 5
done

# Give some time to the Ray cluster to gather info
echo "Waiting a bit for Ray cluster to gather node info..."
sleep 60

# Run vllm
RAY_ADDRESS="http://$head_node_ip:8265" ray job submit \
    --working-dir src/open_r1 \
    --no-wait \
    -- vllm serve $MODEL \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --gpu-memory-utilization=0.85 \
    --max-model-len 16384 \
    --enable-chunked-prefill \
    --trust-remote-code \
    --distributed-executor-backend ray

# wait for vllm to load the model
echo "Waiting for vLLM (http://$head_node_ip:8000) server to be up..."

# wait for vllm to load and serve the model
while true; do
    if curl -s -o /dev/null -w "%{http_code}" http://$head_node_ip:8000 >/dev/null 2>&1; then
        echo "Received response from http://$head_node_ip:8000"
        break
    else
        echo "Still waiting... (Press Ctrl+C to cancel)"
        sleep 60
    fi
done

echo "Checking available models..."
curl http://$head_node_ip:8000/v1/models

echo "Executing sanity check..."
curl http://$head_node_ip:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"$MODEL\",
        \"prompt\": \"<｜begin▁of▁sentence｜><｜User｜>hi, how are you?<｜Assistant｜>\",
        \"max_tokens\": 2048,
        \"temperature\": 0.6
    }"

# Finally submit the job to the cluster
echo "Submitting job to ray cluster..."
RAY_ADDRESS="http://$head_node_ip:8265" ray job submit \
    --working-dir src/open_r1 \
    -- python -u generate.py \
    --model "$MODEL" \
    --hf-dataset "$HF_DATASET" \
    ${HF_DATASET_CONFIG:+--hf-dataset-config "$HF_DATASET_CONFIG"} \
    --hf-dataset-split "$HF_DATASET_SPLIT" \
    --prompt-column "$PROMPT_COLUMN" \
    ${TEMPERATURE:+--temperature "$TEMPERATURE"} \
    ${TOP_P:+--top-p "$TOP_P"} \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --num-generations "$NUM_GENERATIONS" \
    ${HF_OUTPUT_DATASET:+--hf-output-dataset "$HF_OUTPUT_DATASET"} \
    ${PRIVATE:+--private} \
    --vllm-server-url "http://$head_node_ip:8000/v1"
#!/bin/bash
#SBATCH --job-name=agent-sft-single
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --qos=high
#SBATCH --partition=hopper-prod
#SBATCH --output=./logs/%x-%j.out
#SBATCH --error=./logs/%x-%j.err
#SBATCH --requeue
#SBATCH --time=3-00:00:00

if [[ "$*" == *"--help"* ]]; then
  echo "Usage: sbatch slurm/train_single.slurm [options]"
  echo "Options:"
  echo "  --model MODEL            Model name"
  echo "  --task TASK              Task name (e.g. sft, grpo)"
  echo "  --config SUFFIX          Configuration suffix (e.g. demo, v00.00)"
  echo "  --args \"ARGS\"          Optional arguments to pass to the training script"
  exit 0
fi

# Specific configuration optimized for the Hugging Face Compute Cluster
module load cuda/12.4
set -x -e

source ~/.bashrc
source openr1/bin/activate
START_TIME=$(date +%s)
echo "START TIME: $(date)"

# Default values
MODEL=""
TASK=""
CONFIG_SUFFIX=""
OPTIONAL_ARGS=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)
      MODEL="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --config)
      CONFIG_SUFFIX="$2"
      shift 2
      ;;
    --args)
      OPTIONAL_ARGS="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

export HF_HOME="/fsx/aymeric/.cache/"

# Validate required arguments
if [[ -z "$MODEL" || -z "$TASK" || -z "$CONFIG_SUFFIX" ]]; then
  echo "Error: Missing required arguments"
  echo "Run with --help for usage information"
  exit 1
fi

CONFIG_FILE=recipes/$MODEL/$TASK/config_$CONFIG_SUFFIX.yaml

# Extract model info from config (for potential vLLM usage)
MODEL_PATH=$(grep 'model_name_or_path:' $CONFIG_FILE | awk '{print $2}')
REVISION=$(grep 'model_revision:' $CONFIG_FILE | head -n 1 | awk '{print $2}')

# Check if vLLM is needed (though unlikely for single GPU training)
USE_VLLM="false"
if [[ -f "$CONFIG_FILE" ]] && grep -qE '^\s*use_vllm:\s*true' "$CONFIG_FILE"; then
    USE_VLLM="true"
    echo "Warning: vLLM usage detected in single GPU setup. This may not be optimal."
fi

# Set up environment for better debugging
export CUDA_LAUNCH_BLOCKING=1
export TRANSFORMERS_VERBOSITY=info

# Build command
export CMD="src/open_r1/$TASK.py --config $CONFIG_FILE $OPTIONAL_ARGS"

echo "Running command: python $CMD"
echo "Config file: $CONFIG_FILE"
echo "Model: $MODEL_PATH"
echo "Revision: $REVISION"

# Run training directly with python (no accelerate needed for single GPU)
python $CMD 2>&1

END_TIME=$(date +%s)
echo "END TIME: $(date)"
ELAPSED_SECONDS=$((END_TIME - START_TIME))
HOURS=$((ELAPSED_SECONDS / 3600))
MINUTES=$(( (ELAPSED_SECONDS % 3600) / 60 ))
SECONDS=$((ELAPSED_SECONDS % 60))
echo "TOTAL JOB TIME: ${HOURS}h ${MINUTES}m ${SECONDS}s (${ELAPSED_SECONDS} seconds)"
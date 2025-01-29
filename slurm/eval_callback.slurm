#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --partition=hopper-prod
#SBATCH --output=./logs/evaluate/%x-%j.out
#SBATCH --err=./logs/evaluate/%x-%j.err
#SBATCH --requeue

set -x -e
source ~/.bashrc
conda activate openr1

TASK_NAME=$1
TASKS=$2
MODEL_ID=$3
MODEL_REVISION=$4
# Optional args
[ -z "$5"] && TENSOR_PARALLEL=False || TENSOR_PARALLEL=$5
[ -z "$6"] && TRUST_REMOTE_CODE=False || TRUST_REMOTE_CODE=$6
# $7 is reserved for system_prompt, see line 51
NUM_GPUS=$(nvidia-smi -L | wc -l)

# Set Whether to use tensor parallelism or data parallelism
if [ "$TENSOR_PARALLEL" = "True" ]; then
    # use TP to shard model across NUM_GPUS
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
else
    MODEL_ARGS="pretrained=$MODEL_ID,revision=$MODEL_REVISION,trust_remote_code=$TRUST_REMOTE_CODE,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
fi

LM_EVAL_REPO_ID="open-r1/open-r1-eval-leaderboard"
MODEL_NAME=$(echo $MODEL_ID | sed 's/\//_/g') # replaces / with _
DETAILS_REPO_ID="open-r1//details-$MODEL_NAME"
OUTPUT_DIR="eval_results/$MODEL_ID/$MODEL_REVISION/$TASK_NAME"
# We need this flag since we run this script from training jobs that use DeepSpeed and the env vars get progated which causes errors during evaluation
ACCELERATE_USE_DEEPSPEED=false
# Enable fast downloads
HF_HUB_ENABLE_HF_TRANSFER=1

echo "Running lighteval script ..."
echo "Eval results will be saved to $OUTPUT_DIR"
# Check if "custom" is a substring of TASKS
if [[ $TASKS == *"custom"* ]]; then
    echo "Custom task detected. Running custom task evaluation script ..."
    lighteval vllm $MODEL_ARGS $TASKS \
    --custom-tasks "src/open_r1/evaluate.py" \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details \
    ${7:+--system-prompt "$7"}
else
    lighteval vllm $MODEL_ARGS $TASKS \
    --use-chat-template \
    --output-dir $OUTPUT_DIR \
    --save-details \
    ${7:+--system-prompt "$7"}
fi

OUTPUT_FILEPATHS=$(find $OUTPUT_DIR/results/ -type f \( -name "*.json" \))
for filepath in $OUTPUT_FILEPATHS; do
    echo "Uploading $filepath to Hugging Face Hub..."
    filename=$(basename -- "$filepath")
    huggingface-cli upload --repo-type space --private $LM_EVAL_REPO_ID $filepath $OUTPUT_DIR/$filename
done

echo "Uploading details to Hugging Face Hub..."
DETAILS_FILEPATHS=$(find $OUTPUT_DIR/details/ -type f \( -name "*.parquet" \))
echo "DETAILS_FILEPATHS: $DETAILS_FILEPATHS"
TIMESTAMP=$(date +"%Y-%m-%dT%H-%M-%S")
python src/open_r1/utils/upload_details.py --data_files $DETAILS_FILEPATHS --hub_repo_id $DETAILS_REPO_ID --config_name $MODEL_REVISION.$TASK_NAME.$TIMESTAMP
    
echo "Cleaning up ..."
rm -rf $OUTPUT_DIR

echo "Done!"
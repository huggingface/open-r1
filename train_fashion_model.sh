#!/bin/bash

# Set up environment variables
export MODEL_NAME="deepseek-ai/deepseek-llm-1.3b-base"
export OUTPUT_DIR="data/DeepSeek-R1-Fashion-model"
export DATASET_PATH="data/fashion-dataset/fashion_dataset.json"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Run the training
accelerate launch --config_file=recipes/DeepSeek-R1-Fashion/accelerate_config.yaml src/open_r1/sft.py \
    --model_name_or_path $MODEL_NAME \
    --dataset_paths $DATASET_PATH \
    --learning_rate 2.0e-5 \
    --num_train_epochs 3 \
    --max_seq_length 2048 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing \
    --bf16 \
    --logging_steps 5 \
    --eval_strategy steps \
    --eval_steps 50 \
    --save_strategy steps \
    --save_steps 100 \
    --output_dir $OUTPUT_DIR \
    --report_to none

echo "Training completed. Model saved to $OUTPUT_DIR"

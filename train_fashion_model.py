#!/usr/bin/env python

import os
import json
import subprocess
import tempfile
from datasets import Dataset

# Set up environment variables
MODEL_NAME = "deepseek-ai/deepseek-llm-1.3b-base"
OUTPUT_DIR = "data/DeepSeek-R1-Fashion-model"
DATASET_PATH = "data/fashion-dataset/fashion_dataset.json"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the dataset
with open(DATASET_PATH, 'r') as f:
    raw_data = json.load(f)

# Extract the relevant fields from the conversations
train_data = []

for item in raw_data:
    # Extract the system prompt
    system_prompt = item["conversation"]["system"]
    
    # Combine the user and assistant messages into a single text string
    messages = item["conversation"]["messages"]
    conversation = []
    
    for msg in messages:
        if msg["role"] == "user":
            conversation.append({"role": "user", "content": msg["content"]})
        elif msg["role"] == "assistant":
            conversation.append({"role": "assistant", "content": msg["content"]})
    
    train_data.append({
        "id": item["id"],
        "conversations": conversation,
        "system": system_prompt
    })

# Create or update the JSONL train file
train_jsonl_path = 'data/fashion-dataset-train.jsonl'
with open(train_jsonl_path, 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

print(f"Dataset converted and saved to '{train_jsonl_path}'")

# Since we're having issues with the command-line arguments, let's create a config file
config_path = 'data/train_config.json'
config = {
    "model_name_or_path": MODEL_NAME,
    "output_dir": OUTPUT_DIR,
    "dataset_name": "json",
    "dataset_kwargs": {"data_files": train_jsonl_path},
    "dataset_text_field": "conversations",
    "learning_rate": 2.0e-5,
    "num_train_epochs": 3,
    "max_seq_length": 2048,
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "gradient_checkpointing": True,
    "bf16": True,
    "logging_steps": 5,
    "eval_strategy": "steps",
    "eval_steps": 50,
    "save_strategy": "steps",
    "save_steps": 100,
    "report_to": "none"
}

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)

print(f"Training config saved to '{config_path}'")

# Construct the command with simpler arguments
cmd = [
    "accelerate", "launch",
    "--config_file=recipes/DeepSeek-R1-Fashion/accelerate_config.yaml",
    "src/open_r1/sft.py",
    "--model_name_or_path", MODEL_NAME,
    "--dataset_name", "json",
    "--dataset_kwargs", '{"data_files":"data/fashion-dataset-train.jsonl"}',
    "--dataset_text_field", "conversations",
    "--learning_rate", "2.0e-5",
    "--num_train_epochs", "3",
    "--max_seq_length", "2048",
    "--per_device_train_batch_size", "1",
    "--gradient_accumulation_steps", "4",
    "--gradient_checkpointing",
    "--bf16",
    "--logging_steps", "5",
    "--eval_strategy", "steps",
    "--eval_steps", "50",
    "--save_strategy", "steps",
    "--save_steps", "100",
    "--output_dir", OUTPUT_DIR,
    "--report_to", "none"
]

# Print the command for debugging
print("Running command:", " ".join(cmd))

# Run the command
subprocess.run(cmd)

print(f"Training completed. Model saved to {OUTPUT_DIR}")

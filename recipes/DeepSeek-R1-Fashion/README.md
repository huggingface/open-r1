# DeepSeek-R1-Fashion

This recipe provides configuration files and instructions for training a fashion-specialized version of DeepSeek-R1. The model is fine-tuned to provide high-quality fashion advice, outfit recommendations, and style guidance.

## Training Process

The training process consists of two main steps:

1. **Supervised Fine-Tuning (SFT)**: Fine-tune the base DeepSeek-R1 model on a fashion dataset
2. **Group Relative Policy Optimization (GRPO)**: Further refine the model with reinforcement learning

## Data Preparation

Before training, you need to prepare a fashion dataset. You can use the provided script to generate synthetic fashion conversations:

```bash
python recipes/DeepSeek-R1-Fashion/generate_fashion_dataset.py --output-path data/fashion-dataset --num-samples 10000
```

For the GRPO phase, you'll need query data:

```bash
# Create a directory for fashion queries
mkdir -p data/fashion-queries-dataset

# Example of creating a simple query dataset
python -c "
from datasets import Dataset
import json

queries = [
    'What should I wear to a summer wedding?',
    'How do I style a denim jacket?',
    'What are the current fashion trends?',
    # Add more fashion queries here
]

ds = Dataset.from_dict({'query': queries})
ds.to_json('data/fashion-queries-dataset/fashion_queries.jsonl')
"
```

## Training Commands

### 1. Supervised Fine-Tuning (SFT)

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/DeepSeek-R1-Fashion/sft/config_fashion.yaml
```

### 2. GRPO Training

```bash
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/DeepSeek-R1-Fashion/grpo/config_fashion.yaml
```

## Evaluation

After training, evaluate your fashion model using:

```bash
MODEL=your-username/DeepSeek-R1-Fashion
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:32768,temperature:0.6,top_p:0.95}"
OUTPUT_DIR=data/evals/$MODEL

# Fashion style evaluation
TASK=fashion_style
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

## Configuration Details

### SFT Configuration

The SFT configuration (`config_fashion.yaml`) uses the following key settings:

- Base model: DeepSeek-R1
- Learning rate: 5e-5
- Training epochs: 1
- Max sequence length: 16384
- Batch size: 16

### GRPO Configuration

The GRPO configuration includes:

- Base model: Your SFT-trained fashion model
- Learning rate: 1e-6
- Reward functions:
  - accuracy: Checks factual correctness
  - format: Ensures proper output formatting
  - tag_count: Maintains proper usage of think/answer tags
  - fashion_relevance: Custom reward for fashion-specific quality

## Customization

You can customize the configurations by:

1. Adjusting training parameters in the config files
2. Modifying the system prompt to better match your fashion use case
3. Using different reward weights in the GRPO phase
4. Adding custom reward functions for fashion-specific evaluation

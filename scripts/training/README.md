# Scripts to Train and Evaluate Chat Models

## GRPO

```
accelerate launch scripts/training/grpo.py \
    --output_dir Qwen2.5-0.5B-GRPO \
    --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
    --dataset_name AI-MO/NuminaMath-TIR \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --logging_steps 10
```
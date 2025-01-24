# Scripts to Train and Evaluate Chat Models

## GRPO

```
accelerate launch scripts/training/grpo.py --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct --output_dir Qwen2.5-0.5B-GRPO --dataset_name AI-MO/NuminaMath-TIR
```
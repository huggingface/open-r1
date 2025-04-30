# How to train the DataAgent-7B model


For the Qwen model
```bash
sbatch --job-name=train-data-agent-qwen --nodes=1 slurm/train.slurm --model DataAgent-Qwen-7B --task sft --config v00.00 --accelerator zero3
```

For the Llama model
```bash
sbatch --job-name=train-data-agent-llama --nodes=1 slurm/train.slurm --model DataAgent-Llama-8B --task sft --config v00.00 --accelerator zero3
```



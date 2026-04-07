Launch:
```bash
sbatch --nodes=1 slurm/train.slurm --model SmolLM2-1.7B-Instruct --task sft --config agent --accelerator zero3
```
Refers to the config  recipes/SmolLM2-1.7B-Instruct/sft/config_agent.yaml
zero3 is one of the accelerate configs in recipes/accelerate_configs


### VLM training

Launch in multi GPU:
```bash
sbatch --qos=high --nodes=1 slurm/train.slurm --model Qwen2.5-VL-3B-Instruct --task sft --config agent --accelerator zero3
```

🛑 For me the above fails because of NCCL issues, I launch it in single-GPU mode as follows:
```bash
sbatch slurm/trainsingle.slurm --model Qwen2.5-VL-3B-Instruct --task sft --config agent
```

The config is located under recipes/Qwen2.5-VL-3B-Instruct/sft/config_agent.yaml

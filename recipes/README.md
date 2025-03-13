# Post-training recipes

## OlympicCoder

To train the OlympicCoder models, run:

```
# 7B
sbatch --nodes=1 slurm/train.slurm OlympicCoder-7B sft v00.00 zero3

# 32B
sbatch --nodes=16 slurm/train.slurm OlympicCoder-32B sft v00.00 fsdp
```

Note that we found it necessary to switch to FSDP1 and paged AdamW 8-bit for the 32B model in order to fit the largest possible context size.
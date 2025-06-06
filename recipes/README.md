# Post-training recipes

## OpenR1 Distill 7B

To train the OpenR1 Distill 7B model, run:

```
sbatch --nodes=1 slurm/train.slurm --model OpenR1-Distill-7B --task sft --config distill --accelerator zero3
```

## OlympicCoder

To train the OlympicCoder models, run:

```
# 7B
sbatch --nodes=1 slurm/train.slurm --model OlympicCoder-7B --task sft --config v00.00 --accelerator zero3

# 32B
sbatch --nodes=16 slurm/train.slurm --model OlympicCoder-32B --task sft --config v00.00 --accelerator fsdp
```

Note that we found it necessary to switch to FSDP1 and paged AdamW 8-bit for the 32B model in order to fit the largest possible context size.
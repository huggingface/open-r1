# Instructions to train Qwen-R1

We build the **Qwen-R1** by doing `SFT` on [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k) and then `GRPO` on [NuminaMath-TIR](https://huggingface.co/datasets/AI-MO/NuminaMath-TIR).

## Setup

Follow the installation instructions in https://github.com/huggingface/open-r1/tree/main?tab=readme-ov-file## Installation.

## Training

We support training models with either DDP or DeepSpeed ZeRO-2 and ZeRO-3. To switch between methods, simply change the path to the `recipes` YAML config in `accelerate_configs`.

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

You can find the configuration files for different model sizes in this folder and specify the path to the configuration file in the commands below.

```shell
# SFT
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/sft/config_full.yaml

# GRPO
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/grpo.py --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml
```

# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is work in progress, let's build it together!*

## Overview
The goal of this repo is to build the missing pieces of the R1 pipeline such that everybody can reproduce and build on top of it. The project is simple by design and mostly consists of:

- `src/open_r1` contains the script to train and evaluate models as well a generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset
    - `sft.py`: simple SFT of a model on a dataset
    - `evaluate.py`: evaluates a model on the R1 benchmarks
    - `generate.py`: use a model to generate syntehtic data
- `Makefile` contains an easy to run command for each step in the R1 pipeline leveraging the scipts above.

## Installation

To run the code in this project, first, create a Python virtual environment using e.g. Conda:

```shell
conda create -n openr1 python=3.11 && conda activate openr1
```

Next, install vLLM:

```shell
pip install vllm==0.6.6.post1

# For HF (cluster only has CUDA 12.1)
pip install vllm==0.6.6.post1 --extra-index-url https://download.pytorch.org/whl/cu121
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
pip install -e ".[dev]"
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Evaluating models (internal)
For small models use `--data_parallel=$NUM_GPUS`, for large models shard with `--tensor_parallel=$NUM_GPUS`
Example for evaluating `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B `
```
NUM_GPUS=1
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
MODEL_ARGS="pretrained=$MODEL_ID,dtype=bfloat16,data_parallel=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24 # or math
OUTPUT_DIR=evals/$MODEL

lighteval $MODEL_ARGS $TASK --use-chat-template --custom-tasks src/open_r1/eval/$TASK.py --output-dir $OUTPUT_DIR --system-prompt="Please reason step by step, and put your final answer within \boxed{}."
```


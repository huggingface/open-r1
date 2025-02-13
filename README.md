# Open R1

*A fully open reproduction of DeepSeek-R1. This repo is a work in progress, let's build it together!*

**Table of Contents**  
1. [Overview](#overview)  
2. [Plan of attack](#plan-of-attack)  
3. [Installation](#installation)  
4. [Training models](#training-models)  
   - [SFT](#sft)  
   - [GRPO](#grpo)  
5. [Evaluating models](#evaluating-models)  
6. [Reproducing Deepseek's evaluation results](#reproducing-deepseeks-evaluation-results)  
7. [Data generation](#data-generation)  
   - [Generate data from a smol distilled R1 model](#generate-data-from-a-smol-distilled-r1-model)  
   - [Generate data from DeepSeek-R1](#generate-data-from-deepseek-r1)  
8. [Contributing](#contributing)

## Overview

The goal of this repo is to build the missing pieces of the R1 pipeline such that everybody can reproduce and build on top of it. The project is simple by design and mostly consists of:


- `src/open_r1`: contains the scripts to train and evaluate models as well as generate synthetic data:
    - `grpo.py`: trains a model with GRPO on a given dataset.
    - `sft.py`: performs a simple SFT of a model on a dataset.
    - `evaluate.py`: evaluates a model on the R1 benchmarks.
    - `generate.py`: generates synthetic data from a model using [Distilabel](https://github.com/argilla-io/distilabel).
- `Makefile`: contains easy-to-run commands for each step in the R1 pipeline leveraging the scripts above.

### Plan of attack

We will use the DeepSeek-R1 [tech report](https://github.com/deepseek-ai/DeepSeek-R1) as a guide, which can roughly be broken down into three main steps:

* Step 1: replicate the R1-Distill models by distilling a high-quality corpus from DeepSeek-R1.
* Step 2: replicate the pure RL pipeline that DeepSeek used to create R1-Zero. This will likely involve curating new, large-scale datasets for math, reasoning, and code.
* Step 3: show we can go from base model to RL-tuned via multi-stage training.

<center>
    <img src="assets/plan-of-attack.png" width="500">
</center>


## Installation

> [!CAUTION]
> Libraries rely on CUDA 12.4. If you see errors related to segmentation faults, double check the version your system is running with `nvcc --version`.

To run the code in this project, first, create a Python virtual environment using e.g. `uv`.
To install `uv`, follow the [UV Installation Guide](https://docs.astral.sh/uv/getting-started/installation/).


```shell
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip --link-mode=copy
```

Next, install vLLM:

```shell
uv pip install vllm==0.7.2 --link-mode=copy
```

This will also install PyTorch `v2.5.1` and it is **very important** to use this version since the vLLM binaries are compiled for it. You can then install the remaining dependencies for your specific use case via `pip install -e .[LIST OF MODES]`. For most contributors, we recommend:

```shell
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e ".[dev]" --link-mode=copy
```

Next, log into your Hugging Face and Weights and Biases accounts as follows:

```shell
huggingface-cli login
wandb login
```

Finally, check whether your system has Git LFS installed so that you can load and push models/datasets to the Hugging Face Hub:

```shell
git-lfs --version
```

If it isn't installed, run:

```shell
sudo apt-get install git-lfs
```

## Training models

We support training models with either DDP or DeepSpeed (ZeRO-2 and ZeRO-3). For example, to run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

```shell
# Train via command line
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name HuggingFaceH4/Bespoke-Stratos-17k \
    --learning_rate 2.0e-5 \
    --num_train_epochs 1 \
    --packing \
    --max_seq_length 4096 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing \
    --bf16 \
    --output_dir data/Qwen2.5-1.5B-Open-R1-Distill

# Train via YAML config
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

Currently, the following tasks are supported:

* Supervised Fine-Tuning `sft`
* Group Relative Policy Optimization `grpo`

> [!TIP]
> If you scale up/down the number of GPUs, we recommend also scaling up the per-device batch size or number of gradient accumulation steps to keep the global batch size constant.

By default, these scripts will push each model to your Hugging Face Hub username, i.e. `{username}/{model_name}-{task}`. You can override the parameters in each YAML config by appending them to the command as follows: 

```shell
# Change batch size, number of epochs etc
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --per_device_train_batch_size=1 --num_train_epochs=5
```

If you also wish to override the Weights and Biases default settings, you can do so as follows:

```shell
accelerate launch --config_file recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
    --wandb_entity huggingface --wandb_project open-r1 --run_name Qwen2.5-1.5B-GRPO
```

> [!NOTE]
> The training commands below are configured for a node of 8 x H100s (80GB). For different hardware and topologies, you may need to tune the batch size and number of gradient accumulation steps.

### SFT

To run SFT on a dataset distilled from DeepSeek-R1 with reasoning traces such as [Bespoke-Stratos-17k](https://huggingface.co/datasets/bespokelabs/Bespoke-Stratos-17k), run:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero3.yaml \
    src/open_r1/sft.py \
    --config recipes/Qwen2.5-1.5B-Instruct/sft/config_demo.yaml
```

### GRPO

To train via the GRPO trainer, we use one GPU to run vLLM for faster generation and the remaining GPUs for training. For example, one a node with 8 GPUs, use the `recipes/accelerate_configs/zero2.yaml` config and then overwrite `num_processes` to run on 7 devices:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-1.5B-Instruct/grpo/config_demo.yaml
```

We provide a minimal reproducible experiment using GRPO for mathematical reasoning, referencing the approach from [SimpleRL-Reason](https://hkust-nlp.notion.site/simplerl-reason) which uses a 7B model trained on 8K examples. Running this on 8 H100 80G GPU takes about 3 hours:

```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml
```

Our final [model](https://huggingface.co/Dongwei/Qwen-2.5-7B_Base_Math_smalllr), while using different learning rates, loss functions and reward structures, achieves 69.4% accuracy on MATH-500, demonstrating a 17%+ improvement over the base model.

### Launching jobs on a Slurm cluster

If you have access to a Slurm cluster, we provide a `slurm/train.slurm` script that will automatically queue training jobs for you. Here's how you can use it:

```shell
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm {model_name} {task} {config_suffix} {accelerator}
```

Here `{model_name}` and `{task}` are defined as above, while `{config_suffix}` refers to the specific config and `{accelerator}` refers to the choice of 🤗 Accelerate config in `recipes/accelerate_configs`. If you wish to override the default config parameters, you can provide them by appending a space-separated string like `'--arg1=value1 --arg2=value2'`. Here's a concrete example to run SFT on 1 node of 8 GPUs:

```shell
# Launch on Slurm and override default hyperparameters
sbatch --job-name=open_r1 --nodes=1 slurm/train.slurm Qwen2.5-1.5B-Instruct sft demo zero3 '--per_device_train_batch_size=1 --num_train_epochs=5'
```

You can scale the number of nodes by increasing the `--nodes` flag.

> [!NOTE]
> The configuration in `slurm/train.slurm` is optimised for the Hugging Face Compute Cluster and may require tweaking to be adapted to your own compute nodes.

## Evaluating models

We use `lighteval` to evaluate models, with custom tasks defined in `src/open_r1/evaluate.py`. For models which fit on a single GPU, run:

```shell
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8"
OUTPUT_DIR=data/evals/$MODEL

# AIME 2024
TASK=aime24
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# MATH-500
TASK=math_500
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR

# GPQA Diamond
TASK=gpqa:diamond
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

> [!IMPORTANT]
> You must set `max_model_length=32768` in the `vllm` command to align with the `generation_size` we define per eval. Without this, `lighteval` will throw an error.

To increase throughput across multiple GPUs, use _data parallel_ as follows:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,data_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

For large models which require sharding across GPUs, use _tensor parallel_ and run:

```shell
NUM_GPUS=8
MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,tensor_parallel_size=$NUM_GPUS,max_model_length=32768,gpu_memory_utilisation=0.8"
TASK=aime24
OUTPUT_DIR=data/evals/$MODEL

export VLLM_WORKER_MULTIPROC_METHOD=spawn
lighteval vllm $MODEL_ARGS "custom|$TASK|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR 
```

You can also launch an evaluation with `make evaluate`, specifying the model, task, and optionally the parallelism technique and number of GPUs.

To evaluate on a single GPU:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
```

To use Data Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
```

To use Tensor Parallelism:

```shell
make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8
```

## Reproducing Deepseek's evaluation results

> [!NOTE]
> The DeepSeek-R1 paper uses sampling with a temperature of 0.6, a top-p value of 0.95, and 64 responses per query to estimate `pass@1`. Below, we report the results from greedy decoding, which likely explains the small 1-3σ discrepancies between our results and theirs.

### MATH-500

We are able to reproduce Deepseek's reported results on the MATH-500 benchmark within ~1-3 standard deviations:

| Model                         | MATH-500 (🤗 LightEval) | MATH-500 (DeepSeek Reported) |
|:------------------------------|:-----------------------:|:----------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |          81.2           |             83.9             |
| DeepSeek-R1-Distill-Qwen-7B   |          91.8           |             92.8             |
| DeepSeek-R1-Distill-Qwen-14B  |          94.2           |             93.9             |
| DeepSeek-R1-Distill-Qwen-32B  |          95.0           |             94.3             |
| DeepSeek-R1-Distill-Llama-8B  |          85.4           |             89.1             |
| DeepSeek-R1-Distill-Llama-70B |          93.4           |             94.5             |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=$NUM_GPUS"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|math_500|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

Alternatively, you can launch Slurm jobs as follows:

```shell
python scripts/run_benchmarks.py --model-id={model_id}  --benchmarks math_500
```

### GPQA Diamond

We are able to reproduce Deepseek's reported results on the GPQA Diamond benchmark within ~1-3 standard deviations:

| Model                         | GPQA Diamond (🤗 LightEval) | GPQA Diamond (DeepSeek Reported) |
|:------------------------------|:---------------------------:|:--------------------------------:|
| DeepSeek-R1-Distill-Qwen-1.5B |            33.3             |               33.8               |
| DeepSeek-R1-Distill-Qwen-7B   |            48.4             |               49.1               |
| DeepSeek-R1-Distill-Qwen-14B  |            55.6             |               59.1               |
| DeepSeek-R1-Distill-Qwen-32B  |            58.6             |               62.1               |
| DeepSeek-R1-Distill-Llama-8B  |            51.0             |               49.0               |
| DeepSeek-R1-Distill-Llama-70B |            65.2             |               65.2               |

To reproduce these results use the following command:

```shell
NUM_GPUS=1 # Set to 8 for 32B and 70B models
MODEL=deepseek-ai/{model_name}
MODEL_ARGS="pretrained=$MODEL,dtype=bfloat16,max_model_length=32768,gpu_memory_utilisation=0.8,tensor_parallel_size=$NUM_GPUS"
OUTPUT_DIR=data/evals/$MODEL

lighteval vllm $MODEL_ARGS "custom|gpqa:diamond|0|0" \
    --custom-tasks src/open_r1/evaluate.py \
    --use-chat-template \
    --output-dir $OUTPUT_DIR
```

```shell
python scripts/run_benchmarks.py --model-id={model_id}  --benchmarks gpqa
```

## Data generation

### Generate data from a smol distilled R1 model

The following example can be run in 1xH100. 
First install the following dependencies:

```shell
uv pip install "distilabel[vllm]>=1.5.2"
```

Now save the following snippet into a file named `pipeline.py` and run it with `python pipeline.py`. It will generate 4 outputs for each of the 10 examples (change the username for the repository to your org/user name):

```python
from datasets import load_dataset
from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps.tasks import TextGeneration


prompt_template = """\
You will be given a problem. Please reason step by step, and put your final answer within \boxed{}:
{{ instruction }}"""

dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train").select(range(10))

model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"  # Exchange with another smol distilled r1

with Pipeline(
    name="distill-qwen-7b-r1",
    description="A pipeline to generate data from a distilled r1 model",
) as pipeline:

    llm = vLLM(
        model=model_id,
        tokenizer=model_id,
        extra_kwargs={
            "tensor_parallel_size": 1,
            "max_model_len": 8192,
        },
        generation_kwargs={
            "temperature": 0.6,
            "max_new_tokens": 8192,
        },
    )
    prompt_column = "problem"
    text_generation = TextGeneration(
        llm=llm, 
        template=prompt_template,
        num_generations=4,
        input_mappings={"instruction": prompt_column} if prompt_column is not None else {}
    )


if __name__ == "__main__":
    distiset = pipeline.run(dataset=dataset)
    distiset.push_to_hub(repo_id="username/numina-deepseek-r1-qwen-7b")
```

Take a look at the sample dataset at [HuggingFaceH4/numina-deepseek-r1-qwen-7b](https://huggingface.co/datasets/HuggingFaceH4/numina-deepseek-r1-qwen-7b).


### Generate data from DeepSeek-R1

To run the bigger DeepSeek-R1, we used 2 nodes, each with 8×H100 GPUs using the slurm file present in this repo at `slurm/generate.slurm`. First, install the dependencies:

(for now we need to install the vllm dev wheel that [fixes the R1 cuda graph capture](https://github.com/vllm-project/vllm/commits/221d388cc5a836fa189305785ed7e887cea8b510/csrc/moe/moe_align_sum_kernels.cu))
```shell
pip install https://wheels.vllm.ai/221d388cc5a836fa189305785ed7e887cea8b510/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl --extra-index-url https://download.pytorch.org/whl/cu121

uv pip install "distilabel[vllm,ray,openai]>=1.5.2"
```

And then run the following command:

```shell
sbatch slurm/generate.slurm \
    --hf-dataset AI-MO/NuminaMath-TIR \
    --temperature 0.6 \
    --prompt-column problem \
    --model deepseek-ai/DeepSeek-R1 \
    --hf-output-dataset username/r1-dataset
```

> [!NOTE]  
> While the job is running, you can setup an SSH tunnel through the cluster login node to access the Ray dashboard from your computer running `ssh -L 8265:ray_ip_head_node:8265 <login_node>`, then browsing `http://localhost:8265`

## Contributing

Contributions are welcome. Please refer to https://github.com/huggingface/open-r1/issues/23.

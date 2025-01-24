 # Open R1

 ## Installation instructions

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
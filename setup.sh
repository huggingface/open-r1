#!/bin/bash

REQUIRED_PREFIX="12.1"
CUDA_FULL_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)

# Only stable for CUDA 12.1 right now.
if [[ "$CUDA_FULL_VERSION" != ${REQUIRED_PREFIX}* ]]; then
  echo "Error: CUDA version starting with $REQUIRED_PREFIX is required, but found CUDA version $CUDA_FULL_VERSION."
  exit 1
fi

echo "---"
echo "Using CUDA $CUDA_FULL_VERSION"

echo "---"
echo "Setting up vllm and nvJitLink."
pip install "vllm>=0.7.1" --extra-index-url https://download.pytorch.org/whl/cu121 2>&1
export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH

echo "---"
echo "Installing dependencies."
pip install -U -e ".[dev]"

echo "---"
echo "Finished setup. GRPO example (expects 8 accelerators):

    HF_HUB_ENABLE_HF_TRANSFER=1 ACCELERATE_LOG_LEVEL=info accelerate launch \\
        --config_file recipes/accelerate_configs/zero3.yaml \\
        --num_processes=7 \\
        src/open_r1/grpo.py \\
        --config recipes/qwen/Qwen2.5-1.5B-Instruct/grpo/confg_full.yaml
"
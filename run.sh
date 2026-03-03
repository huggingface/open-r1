CUDA_VISIBLE_DEVICES=0 python -m trl.scripts.vllm_serve \
    --model KAKA22/CodeRM-8B \
    --tensor_parallel_size 1 \
    --gpu_memory_utilization 0.85 \
    --max_model_len 4608 \
    --port 8000 \
    --enable_prefix_caching true

export VLLM_WORKER_MULTIPROC_METHOD=spawn

CUDA_VISIBLE_DEVICES=1,2,3 \
ACCELERATE_LOG_LEVEL=info \
    accelerate launch --config_file recipes/coderm/zero3-3gpu.yaml \
    src/open_r1/grpo.py --config recipes/coderm/coderm-config-server.yaml \
    --report_to swanlab
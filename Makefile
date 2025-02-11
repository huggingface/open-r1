.PHONY: style quality

# make sure to test the local checkout in scripts and not the pre-installed one (don't use quotes!)
export PYTHONPATH = src

check_dirs := src tests

style:
	ruff format --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort $(check_dirs) setup.py

quality:
	ruff check --line-length 119 --target-version py310 $(check_dirs) setup.py
	isort --check-only $(check_dirs) setup.py
	flake8 --max-line-length 119 $(check_dirs) setup.py

test:
	pytest -sv tests/

# Evaluation

evaluate:
	$(eval PARALLEL_ARGS := $(if $(PARALLEL),$(shell \
		if [ "$(PARALLEL)" = "data" ]; then \
			echo "data_parallel_size=$(NUM_GPUS)"; \
		elif [ "$(PARALLEL)" = "tensor" ]; then \
			echo "tensor_parallel_size=$(NUM_GPUS)"; \
		fi \
	),))
	$(if $(filter tensor,$(PARALLEL)),export VLLM_WORKER_MULTIPROC_METHOD=spawn &&,) \
	MODEL_ARGS="pretrained=$(MODEL),dtype=bfloat16,$(PARALLEL_ARGS),max_model_length=32768,gpu_memory_utilisation=0.8" && \
	lighteval vllm $$MODEL_ARGS "custom|$(TASK)|0|0" \
		--custom-tasks src/open_r1/evaluate.py \
		--use-chat-template \
		--system-prompt="Please reason step by step, and put your final answer within \boxed{}." \
		--output-dir data/evals/$(MODEL)

# Example usage:
# Single GPU:
#   make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24
# Data parallel:
#   make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=data NUM_GPUS=8
# Tensor parallel:
#   make evaluate MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-32B TASK=aime24 PARALLEL=tensor NUM_GPUS=8

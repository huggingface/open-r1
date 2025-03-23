

```bash
# Run on pod:
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install

apt install -y tmux
pip install uv
uv venv openr1 --python 3.11 && source openr1/bin/activate && uv pip install --upgrade pip
uv pip install vllm==0.7.2
uv pip install setuptools && uv pip install flash-attn --no-build-isolation
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
huggingface-cli login
wandb login

tmux new -s openr1

source openr1/bin/activate
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/zero2.yaml \
    --num_processes=7 src/open_r1/grpo.py \
    --config recipes/Qwen2.5-Math-7B/grpo/config_simple_rl.yaml

# On a new ec2 instance, follow: https://gist.github.com/aidando73/23bbfa534a01ebefc4b2ef505ca6b464#file-stop_pod_after_delay-md
```
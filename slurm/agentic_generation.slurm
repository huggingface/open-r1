#!/bin/bash
#SBATCH --job-name=agentic-r1
#SBATCH --partition=hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=64
#SBATCH --exclusive
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=7-00:00:00
set -exuo pipefail

source ~/.bashrc
source openr1/bin/activate

python scripts/generate_agent_traces.py \
    --output-file "data/codeforces_agentic_generations.jsonl" \
    --prompt-column "prompt" \
    --uuid-column "contestId" \
    --api-addr "10.53.86.164:39876" \
    --num-generations 5 \
    --max-tokens 8096 \
    --max-concurrent 64
#!/bin/bash
#SBATCH --job-name=r1-router
#SBATCH --partition=hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1875m
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=30-00:00:00
#SBATCH --requeue

set -exuo pipefail

# TODO: Adjust these variables to your cluster configuration
CONDA_ENV="sglang124"
ROUTER_PORT=39876

trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' SIGUSR1

while getopts "e:h" opt; do
    case $opt in
        e) CONDA_ENV="$OPTARG" ;;
        h|?) echo "Usage: sbatch $0 [-e CONDA_ENV]"; exit 1 ;;
    esac
done

# TODO: Environment setup, adjust to your cluster configuration
source ~/.bashrc
source "$CONDA_PREFIX/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV" || { echo "Failed to activate conda env $CONDA_ENV"; exit 1; }

python -m sglang_router.launch_router \
    --port "$ROUTER_PORT" \
    --host 0.0.0.0 \
    --worker-startup-timeout-secs 300

# Keep the job running with health checks
while true; do
    if ! curl -s -o /dev/null "http://localhost:$ROUTER_PORT/health"; then
        echo "Error: Router health check failed"
        exit 1
    fi
    sleep 300
done
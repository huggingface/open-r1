#!/bin/bash
#SBATCH --job-name=morph_router
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/morph_router_%j.out
#SBATCH --error=logs/morph_router_%j.err

# Load necessary modules
module load anaconda

# Activate the conda environment
source activate openr1

# Ensure the logs directory exists
mkdir -p logs

# Start the MorphCloud router service
echo "Starting MorphCloud router service on port 8001"
python scripts/morph_router.py --port 8001 --max_num_sandboxes 20

# The router will continue running until the job is terminated
echo "MorphCloud router service exited"
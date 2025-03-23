#!/bin/bash

# this simple script will launch a bunch of piston workers on the HF science cluster

N_INSTANCES=${1:-5}  # Default to 5 instances

for i in $(seq 1 $N_INSTANCES); do
    # Find random (hopefully) available port
    PORT=$(comm -23 <(seq 2000 10000 | sort) <(ss -tan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n1)
    
    # the job name format is important for the code to then be able to get a list of workers. `piston-worker-<port>`
    sbatch \
        --job-name="piston-worker-$PORT" \
        --export=ALL,PORT=$PORT \
        slurm/piston/launch_single_piston.sh
done
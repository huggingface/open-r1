#!/bin/bash
#SBATCH --job-name=piston_worker
#SBATCH --output=/fsx/open-r1/logs/piston/worker-logs/%x-%j.out
#SBATCH --error=/fsx/open-r1/logs/piston/worker-logs/%x-%j.out  # Redirect error logs to .out
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1950M
#SBATCH --partition=hopper-cpu
#SBATCH --time=48:00:00

# sometimes if a bunch of workers start at the same time pyxis dies
sleep $(( RANDOM % 20 ))

# mounting the packages folder lets us not have to manually install the package on each instance
# we use 63b5654156a89c5a2ad281aface21416615d62ec056d88efe8fcd307ce73575a as the latest image requires isolate, which does not work on the HF science cluster (cgroups incompatibility)
# feel free try with the latest image
# the code you see below increases the very constrained piston default limits, and sets the repo url to the one hosting our IOI package
srun --container-mounts=/fsx/guilherme/ioi2024/piston_files/packages:/piston/packages --container-image "ghcr.io#engineer-man/piston:sha256:63b5654156a89c5a2ad281aface21416615d62ec056d88efe8fcd307ce73575a" \
    bash -c "
    export PISTON_COMPILE_TIMEOUT=60000
    export PISTON_RUN_TIMEOUT=60000
    export PISTON_OUTPUT_MAX_SIZE=1000000000
    export PISTON_MAX_FILE_SIZE=1000000000
    export PISTON_DISABLE_NETWORKING=true
    export PISTON_REPO_URL=https://github.com/guipenedo/piston/releases/download/pkgs/index

    sed -i '/app.use(body_parser.urlencoded/c\    app.use(body_parser.urlencoded({ extended: true, limit: \"512mb\" }));' src/index.js
    sed -i '/app.use(body_parser.json/c\    app.use(body_parser.json({ limit: \"512mb\" }));' src/index.js

    # Start server in background
    node src
    "

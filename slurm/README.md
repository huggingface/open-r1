## Serving DeepSeek-R1 on 2x8 H100 SLURM nodes with SGLang 

1. Set up the environment (adjust for your cuda version):
```bash
conda create -n sglang124 python=3.11
conda activate sglang124

pip install torch=2.5.1 --index-url https://download.pytorch.org/whl/cu124

pip install sgl-kernel --force-reinstall --no-deps
pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
```

2. Run the server and wait for the model to load:
```bash
sbatch slurm/serve_r1.slurm -m "/fsx/deepseek-r1-checkpoint" -e "sglang124"
```

3. Run the data generation script:
```bash
python scripts/generate_reasoning.py \
    --dataset-name "AI-MO/NuminaMath-1.5" \
    --output-file "numinamath_r1_generations.jsonl" \
    --prompt-column "problem" \
    --uuid-column "problem" \
    --api-addr "<SGLANG_SERVER_ADDRESS>:39877" \
    --num-generations 2 \
    --max-tokens 16384 \
    --max-concurrent 200
```
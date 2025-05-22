# Generate agent traces

## Step 1: Install (setup the environment)

```bash
make install
```

```bash
source openr1/bin/activate
uv pip install -e ".[smolagents,jupyter]"
```

## Step 2: Start the R1 server

for the `serve_r1.slurm` file do not forget to add the router address

```bash
sbatch slurm/serve_router.slurm
sbatch slurm/serve_r1.slurm
```

## Step 3: Generate traces

This takes ~3 days to complete.

```bash
sbatch slurm/agentic_generation.slurm 
```

## Step 4: Process the traces and upload dataset to the hub

This is done in a jupyter notebook for ease of use during development.

Follow the instructions in eda.ipynb to process the traces into a training dataset.
The notebook filters the failed generation traces then it upload the dataset to the hub for later use.

**TODO:**
- filter the traces to keep traces that pass the test cases
- filter by length of the generation, so traces that converge quickly are favoured.

**Remarks:**
Right now, the `generate_agent_traces.py` file seems to be buggy, it does not generate a single correct trace.By correct, I mean a trace that passes the test cases.

The dataset can be found at https://huggingface.co/datasets/baptistecolle/codeforces-agentic-generations

## Step 5: Train on the traces and upload the model to the hub

```bash
sbatch --nodes=1 --time=8:00:00 slurm/train.slurm Qwen2.5-1.5B-Instruct sft demo_agentic_trace zero3 '--per_device_train_batch_size=1 --num_train_epochs=5'
```

The trainedmodel can be found at https://huggingface.co/baptistecolle/Qwen2.5-1.5B-Open-R1-Distill-Agentic-Trace

## Step 6: Test the model
first need to fix the generate_agent_traces.py file before testing the model I believe (see: `generate_agent_traces.py` file is not working)
**TODO:** create some custom metrics in lighteval for the agentic traces.

# TODOs:
- **The `generate_agent_traces.py` file is not working**: most of the generation of the traces fails, and furthermore based on the eda (exploratory data analysis) none of the generated traces acutally pass the test cases, indeed almost all traces end with `Error:\\nReached max steps.` so none of the generated traces actually solve the test cases

# Current status
- The pipeline is present, now we just need to debug it to increase performance.
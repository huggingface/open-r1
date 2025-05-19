# Pass rate filtering

We provide support to filter datasets by generating and computing pass rate on veriable tasks

See `scripts/pass_rate_filtering/compute_pass_rate.py` and `scripts/pass_rate_filtering/launch_filtering.sh` (hardcoded for DAPO at the moment)

By default the script chunks the dataset, merge can be run using the following snippet (example for DAPO) :

from datasets import load_dataset, concatenate_datasets

name = "open-r1/DAPO-Math-17k-Processed-R1-Distill-Qwen-Math-7B-Merges-v00.02-v01.02-0.3-0.7-filter"

```python
gen_datasets = []
filt_datasets = []
for start in range(0,17400,200):
    end = start + 200
    if start == 17200:
        end = 17398
    gen_config_name = f"gen-{start}-{end}"
    gen_dataset = load_dataset(name, gen_config_name, revision="gen",  split="train")
    gen_datasets.append(gen_dataset)
    
    filt_config_name = f"filt-0.1-0.6-{start}-{end}"
    filt_dataset = load_dataset(name, filt_config_name, revision="pass_rate",  split="train")
    filt_datasets.append(filt_dataset)
    
gen_dataset = concatenate_datasets(gen_datasets)
gen_dataset.push_to_hub(name, config_name="gen", split="train")
print(gen_dataset)

filt_dataset = concatenate_datasets(filt_datasets)
filt_dataset.push_to_hub(name, config_name="default", split="train")

print(filt_dataset)
```
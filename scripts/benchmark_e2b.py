# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Benchmark script for the code_reward function with E2B.

This script measures the performance of the code_reward function with varying numbers
of samples and parallelization levels.

Each sample is a CodeForces problem with a gold standard solution that is executed against a set of public test cases.
"""

from datasets import load_dataset
from open_r1.rewards import code_reward
import time
from tqdm.auto import tqdm

from dotenv import load_dotenv
load_dotenv()

def benchmark_code_reward(example):
    start_time = time.time()
    test_completions = [[{"content": example["gold_standard_solution"]}]]
    reward_kwargs = {"verification_info": [example["verification_info"]]}
    rewards = code_reward(test_completions, **reward_kwargs)
    end_time = time.time()
    example["test_reward"] = rewards[0]
    example["reward_time"] = end_time - start_time
    return example

if __name__ == "__main__":
    parallel_dict = {
        16:[1,4,16],
        64:[4,16, 64],
        256:[16, 64, 96], # cap at 96 as PRO account is limited to 100
    }
    # Store results for table formatting
    results = []
    
    for num_samples in tqdm([16, 64,256], desc="Benchmarking samples"):
        for num_parallel in parallel_dict[num_samples]:
            code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated")
            code_dataset = code_dataset["train"].shuffle(seed=42).select(range(num_samples))

            test_completions = [[{"content": example["gold_standard_solution"]}] for example in code_dataset]
            reward_kwargs = {"verification_info": [example["verification_info"] for example in code_dataset]}

            start_time = time.time()
            rewards = code_reward(test_completions, num_parallel=num_parallel, **reward_kwargs)
            execution_time = time.time() - start_time
            
            # Calculate some statistics about rewards
            mean_reward = sum(rewards) / len(rewards)
            min_reward = min(rewards)
            max_reward = max(rewards)
            
            # Store results
            results.append({
                "num_samples": num_samples,
                "num_parallel": num_parallel,
                "execution_time": execution_time,
                "mean_reward": mean_reward,
                "min_reward": min_reward,
                "max_reward": max_reward
            })
    
    print("\n## Benchmark Results\n")
    print("| Sample Size | Parallelization | Execution Time (s) | Mean Reward | Min Reward | Max Reward |")
    print("|:-----------:|:---------------:|------------------:|:-----------:|:-----------:|:-----------:|")
    
    for result in results:
        print(f"| {result['num_samples']:^11} | {result['num_parallel']:^15} | {result['execution_time']:17.2f} | {result['mean_reward']:^11.4f} | {result['min_reward']:^11.4f} | {result['max_reward']:^11.4f} |")
    
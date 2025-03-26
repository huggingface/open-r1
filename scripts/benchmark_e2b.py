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

from datasets import load_dataset
from open_r1.rewards import code_reward
import time

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
        256:[16, 64, 96], # cap at 96 at PRO account is limited to 100
    }
    for num_samples in [16,64,256]:
        for num_parallel in parallel_dict[num_samples]:
            code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated")
            code_dataset = code_dataset["train"].shuffle(seed=42).select(range(num_samples))

            test_completions = [[{"content": example["gold_standard_solution"]}] for example in code_dataset]
            reward_kwargs = {"verification_info": [example["verification_info"] for example in code_dataset]}

            start_time = time.time()
            rewards = code_reward(test_completions, num_parallel=num_parallel, **reward_kwargs)
            print(num_samples, num_parallel, f"Total time: {time.time() - start_time}")
            print(rewards)
            print("#"*80)
    
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


import unittest

from datasets import load_dataset

from open_r1.rewards import code_reward, ioi_code_reward


class TestCodeRewards(unittest.TestCase):
    def test_python_code_reward(self):
        # requires E2B, see the README.md file
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested")
        NUM_SAMPLES = 20
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": sample["gold_standard_solution"]}] for sample in samples]
        reward_kwargs = {"verification_info": [sample["verification_info"] for sample in samples]}
        rewards = code_reward(test_completions, **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES

    def test_ioi_code_reward(self):
        # This slow test case requires spinning up a bunch (I tested with ~64) of piston workers, see docs here
        # slurm/piston/README.md
        code_dataset = load_dataset("open-r1/ioi-reward-test-dataset")
        NUM_SAMPLES = 16
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": f"```cpp\n{sample['sample_solution']}```"}] for sample in samples]
        keys = [key for key in samples[0] if key not in ["prompt", "completion"]]
        reward_kwargs = {key: [example[key] for example in samples] for key in keys}
        rewards = ioi_code_reward(test_completions, **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES


if __name__ == "__main__":
    unittest.main()

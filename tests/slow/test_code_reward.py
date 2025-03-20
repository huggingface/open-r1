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

from open_r1.rewards import code_reward


class TestCodeRewards(unittest.TestCase):
    def test_code_reward(self):
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python-10k")
        NUM_SAMPLES = 20
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": sample["gold_standard_solution"]}] for sample in samples]
        reward_kwargs = {"verification_info": [sample["verification_info"] for sample in samples]}
        rewards = code_reward(test_completions, **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES


if __name__ == "__main__":
    unittest.main()

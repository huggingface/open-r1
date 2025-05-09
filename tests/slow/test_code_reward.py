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

from e2b_code_interpreter.models import Execution, ExecutionError
from open_r1.rewards import code_reward, ioi_code_reward
from open_r1.utils.routed_morph import RoutedMorphSandbox
from open_r1.utils.routed_sandbox import RoutedSandbox


class TestCodeRewards(unittest.TestCase):
    def test_python_code_reward(self):
        # requires E2B, see the README.md file
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")
        NUM_SAMPLES = 20
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": sample["gold_standard_solution"]}] for sample in samples]
        reward_kwargs = {"verification_info": [sample["verification_info"] for sample in samples]}
        rewards = code_reward(test_completions, **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES

    def test_e2b_router(self):
        # run router locally: python scripts/e2b_router.py
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")
        NUM_SAMPLES = 128
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": sample["gold_standard_solution"]}] for sample in samples]
        reward_kwargs = {"verification_info": [sample["verification_info"] for sample in samples]}
        rewards = code_reward(test_completions, e2b_router_url="0.0.0.0:8000", **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES

    def test_e2b_router_parallel(self):
        # run router locally: python scripts/e2b_router.py
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")

        BATCH_SIZE = 32
        NUM_SAMPLES = 256

        def batch_code_reward(examples):
            test_completions = [[{"content": solution}] for solution in examples["gold_standard_solution"]]
            reward_kwargs = {
                "verification_info": [verification_info for verification_info in examples["verification_info"]]
            }
            rewards = code_reward(test_completions, e2b_router_url="0.0.0.0:8000", **reward_kwargs)
            assert rewards == [1.0] * BATCH_SIZE
            return examples

        code_dataset = code_dataset["train"].select(range(NUM_SAMPLES))
        code_dataset = code_dataset.map(
            batch_code_reward,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=4,
            load_from_cache_file=False,
        )

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

    def test_e2b_router_run_code_success(self):
        # run router locally: python scripts/e2b_router.py
        routed_sandbox = RoutedSandbox(router_url="localhost:8000")
        scripts = [
            "print('hello from integration test')",
            "result = 2 + 2\nprint(result)",
        ]

        results = routed_sandbox.run_code(scripts)

        assert len(results) == 2

        for result in results:
            assert isinstance(result, Execution)
            # assert result.exit_code == 0
            assert result.error is None
            assert "hello" in result.logs["stdout"][0] or "4" in result.logs["stdout"][0]

    def test_e2b_router_run_code_with_error(self):
        # run router locally: python scripts/e2b_router.py

        routed_sandbox = RoutedSandbox(router_url="localhost:8000")
        scripts = ["print('this is fine')", "print('unterminated string"]

        results = routed_sandbox.run_code(scripts)

        assert len(results) == 2

        # First one should be okay
        # assert results[0].exit_code == 0 # Execution object has no attribute 'exit_code'
        assert results[0].error is None
        assert "this is fine" in results[0].logs["stdout"][0]

        # Second one should have a syntax error

        # assert results[1].exit_code != 0 # Execution object has no attribute 'exit_code'
        assert results[1].error is not None
        assert isinstance(results[1].error, ExecutionError)
        assert "SyntaxError" in results[1].error.name

    def test_python_code_reward_morph(self):
        # requires MorphCloud, see the README.md file
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")
        NUM_SAMPLES = 20
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": sample["gold_standard_solution"]}] for sample in samples]
        reward_kwargs = {
            "verification_info": [sample["verification_info"] for sample in samples],
            "provider_type": "morph",
        }
        rewards = code_reward(test_completions, **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES

    def test_morph_router(self):
        # run router locally: python scripts/morph_router.py --port 8001 --max_num_sandboxes 20
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")
        NUM_SAMPLES = 32
        samples = code_dataset["train"].select(range(NUM_SAMPLES))
        test_completions = [[{"content": sample["gold_standard_solution"]}] for sample in samples]
        reward_kwargs = {
            "verification_info": [sample["verification_info"] for sample in samples],
            "provider_type": "morph",
            "morph_router_url": "0.0.0.0:8001",
        }
        rewards = code_reward(test_completions, **reward_kwargs)
        print(rewards)
        assert rewards == [1.0] * NUM_SAMPLES

    def test_morph_router_parallel(self):
        # run router locally: python scripts/morph_router.py --port 8001 --max_num_sandboxes 20
        code_dataset = load_dataset("open-r1/verifiable-coding-problems-python_decontaminated-tested-shuffled")

        BATCH_SIZE = 32
        NUM_SAMPLES = 256

        def batch_code_reward(examples):
            test_completions = [[{"content": solution}] for solution in examples["gold_standard_solution"]]
            reward_kwargs = {
                "verification_info": [verification_info for verification_info in examples["verification_info"]],
                "provider_type": "morph",
                "morph_router_url": "0.0.0.0:8001",
            }
            rewards = code_reward(test_completions, **reward_kwargs)
            assert rewards == [1.0] * BATCH_SIZE
            return examples

        code_dataset = code_dataset["train"].select(range(NUM_SAMPLES))
        code_dataset = code_dataset.map(
            batch_code_reward,
            batched=True,
            batch_size=BATCH_SIZE,
            num_proc=4,
            load_from_cache_file=False,
        )

    def test_morph_router_run_code_success(self):
        # run router locally: python scripts/morph_router.py --port 8001 --max_num_sandboxes 20

        routed_sandbox = RoutedMorphSandbox(router_url="localhost:8001")
        scripts = [
            "print('hello from morph integration test')",
            "result = 2 + 2\nprint(result)",
        ]

        results = routed_sandbox.run_code(scripts)

        assert len(results) == 2

        for result in results:
            assert result.exception_str is None
            assert "hello" in result.text or "4" in result.text

    def test_morph_router_run_code_with_error(self):
        # run router locally: python scripts/morph_router.py --port 8001 --max_num_sandboxes 20

        routed_sandbox = RoutedMorphSandbox(router_url="localhost:8001")
        scripts = ["print('this is fine with morph')", "print('unterminated string"]

        results = routed_sandbox.run_code(scripts)

        assert len(results) == 2

        # First one should be okay
        assert results[0].exception_str is None
        assert "this is fine with morph" in results[0].text

        # Second one should have a syntax error
        assert "SyntaxError" in results[1].text


if __name__ == "__main__":
    unittest.main()

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
from dataclasses import asdict

from datasets import DatasetDict, load_dataset

from open_r1.configs import DatasetConfig, DatasetMixtureConfig, ScriptArguments
from open_r1.utils.data import get_dataset


class TestGetDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset_name = "trl-internal-testing/zen"
        cls.dataset_config = "conversational_preference"
        cls.ref_dataset = load_dataset(cls.dataset_name, cls.dataset_config)

    def test_dataset_and_config_name(self):
        args = ScriptArguments(dataset_name=self.dataset_name, dataset_config=self.dataset_config)
        dataset = get_dataset(args)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertIn("train", dataset)
        self.assertEqual(len(dataset["train"]), len(self.ref_dataset["train"]))

    def test_unweighted_mixture(self):
        """Mix train and test splits of the same dataset."""
        dataset_configs = [
            DatasetConfig(id=self.dataset_name, config=self.dataset_config, split="train", columns=None, weight=None),
            DatasetConfig(id=self.dataset_name, config=self.dataset_config, split="test", columns=None, weight=None),
        ]
        dataset_mixture = DatasetMixtureConfig(
            datasets=dataset_configs,
        )
        args = ScriptArguments(dataset_mixture=asdict(dataset_mixture))
        dataset = get_dataset(args)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertIn("train", dataset)
        self.assertEqual(len(dataset["train"]), len(self.ref_dataset["train"]) + len(self.ref_dataset["test"]))

    def test_weighted_mixture(self):
        """Test loading a dataset mixture with weights."""
        dataset_configs = [
            DatasetConfig(id=self.dataset_name, config=self.dataset_config, split="train", columns=None, weight=0.25),
            DatasetConfig(id=self.dataset_name, config=self.dataset_config, split="test", columns=None, weight=0.5),
        ]
        dataset_mixture = DatasetMixtureConfig(
            datasets=dataset_configs,
        )
        args = ScriptArguments(dataset_mixture=asdict(dataset_mixture))
        dataset = get_dataset(args)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertIn("train", dataset)
        self.assertEqual(
            len(dataset["train"]), len(self.ref_dataset["train"]) // 4 + len(self.ref_dataset["test"]) // 2
        )

    def test_mixture_and_test_split(self):
        """Test loading a dataset mixture with test split."""
        dataset_configs = [
            DatasetConfig(
                id=self.dataset_name, config=self.dataset_config, split="train[:10]", columns=None, weight=None
            ),
        ]
        dataset_mixture = DatasetMixtureConfig(datasets=dataset_configs, test_split_size=0.2)
        args = ScriptArguments(dataset_name=None, dataset_mixture=asdict(dataset_mixture))
        dataset = get_dataset(args)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertIn("train", dataset)
        self.assertIn("test", dataset)
        self.assertEqual(len(dataset["train"]), 8)
        self.assertEqual(len(dataset["test"]), 2)

    def test_mixture_column_selection(self):
        """Test loading a dataset mixture with column selection."""
        dataset_configs = [
            DatasetConfig(
                id=self.dataset_name,
                config=self.dataset_config,
                split="train",
                columns=["prompt", "chosen"],
                weight=None,
            ),
        ]
        dataset_mixture = DatasetMixtureConfig(
            datasets=dataset_configs,
        )
        args = ScriptArguments(dataset_mixture=asdict(dataset_mixture))
        dataset = get_dataset(args)
        self.assertIsInstance(dataset, DatasetDict)
        self.assertIn("train", dataset)
        self.assertIn("prompt", dataset["train"].column_names)
        self.assertIn("chosen", dataset["train"].column_names)

    def test_mixture_with_mismatched_columns(self):
        dataset_configs = [
            DatasetConfig(
                id=self.dataset_name, config=self.dataset_config, split="train", columns=["prompt"], weight=None
            ),
            DatasetConfig(
                id=self.dataset_name, config=self.dataset_config, split="train", columns=["chosen"], weight=None
            ),
        ]
        dataset_mixture = DatasetMixtureConfig(
            datasets=dataset_configs,
        )
        with self.assertRaises(ValueError) as context:
            _ = ScriptArguments(dataset_mixture=asdict(dataset_mixture))
        self.assertIn("Column names must be consistent", str(context.exception))

    def test_no_dataset_name_or_mixture(self):
        with self.assertRaises(ValueError) as context:
            _ = ScriptArguments(dataset_name=None, dataset_mixture=None)
        self.assertIn("Either `dataset_name` or `dataset_mixture` must be provided", str(context.exception))


if __name__ == "__main__":
    unittest.main()

import unittest

import torch


@unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
class TestGRPOScriptArguments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from open_r1.grpo import GRPOScriptArguments

        cls.GRPOScriptArguments = GRPOScriptArguments

    def test_default_weights(self):
        """Test that default weights are correctly set when not provided."""
        args = self.GRPOScriptArguments(dataset_name="ABC")
        self.assertEqual(len(args.reward_funcs), len(args.reward_weights))
        self.assertEqual(args.reward_weights, [1.0] * len(args.reward_funcs))

    def test_custom_weights_valid(self):
        """Test that custom weights are accepted when matching reward_funcs length."""
        args = self.GRPOScriptArguments(
            dataset_name="ABC", reward_funcs=["accuracy", "format", "reasoning_steps"], reward_weights=[0.5, 1.0, 2.0]
        )
        self.assertEqual(args.reward_weights, [0.5, 1.0, 2.0])

    def test_custom_weights_invalid(self):
        """Test that mismatched weights raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.GRPOScriptArguments(
                dataset_name="ABC", reward_funcs=["accuracy", "format"], reward_weights=[1.0, 2.0, 3.0]
            )
        self.assertIn("Number of reward weights", str(context.exception))
        self.assertIn("must match number of reward functions", str(context.exception))

    def test_empty_weights_with_custom_funcs(self):
        """Test that empty weights are filled with 1.0 for custom reward functions."""
        args = self.GRPOScriptArguments(
            dataset_name="ABC",
            reward_funcs=["accuracy", "format", "reasoning_steps"],
        )
        self.assertEqual(len(args.reward_weights), 3)
        self.assertEqual(args.reward_weights, [1.0, 1.0, 1.0])


if __name__ == "__main__":
    unittest.main()

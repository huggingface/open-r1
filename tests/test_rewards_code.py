import unittest

from open_r1.rewards_internal.code.htgen import (
    fix_triple_reward,
    mk_dataset_fix_triple,
    mk_dataset_totality_check,
    totality_check_reward,
)


class TestRewardsCode(unittest.TestCase):
    def test_mk_dataset_totality_check_format_correct(self):
        """test output format of dataset generator mk_dataset_iter_totality_check"""
        ds = mk_dataset_totality_check(n_examples=1)
        examples = list(ds)
        prompt = examples[0]["prompt"]
        label = examples[0]["ground_truth"]
        triple = examples[0]["triple"]
        self.assertIsInstance(prompt, str)
        self.assertIsInstance(label, bool)
        self.assertIsInstance(triple, dict)

    def test_mk_dataset_fix_triple_format_correct(self):
        """test output format of dataset generator mk_dataset_fix_triple"""
        ds = mk_dataset_fix_triple(n_examples=1, seed=5556)
        examples = list(ds)
        ex = examples[0]
        prompt = ex["prompt"]
        label = ex["ground_truth"]
        triple = ex["triple"]
        self.assertIsInstance(prompt, str)
        self.assertIsInstance(label, str)
        self.assertIn(label, ["ok_total", "bad_pre", "bad_post"])
        self.assertIsInstance(triple, dict)

    def test_totality_check_reward_correct(self):
        """Test totality_check_reward"""
        completion = ["True"]
        solution = [True]
        rewards = totality_check_reward(completion, solution)
        self.assertEqual(rewards[0], 1.0)

    def test_totality_check_reward_wrong_format(self):
        """Test totality_check_reward, wrong format"""
        completion = ["The triple is total"]
        solution = [True]
        rewards = totality_check_reward(completion, solution)
        self.assertEqual(rewards[0], 0.0)

    def test_fix_triple_reward_correct(self):
        """fix_triple task: assert a correct completion gives 1.0 reward"""
        triple = {
            "pre": "v3 > 0 && v4 > 2",
            "program": "v5 = 2\nv3 = v5\nv4 = ((5 + (3 + v3)) + (v4 + v5))\nv4 = 9\nv4 = (v3 - 7)",
            "post": "v5 > 6",
        }
        completion = "v5 = 2\nv3 = v5\nv4 = ((5 + (3 + v3)) + (v4 + v5))\nv5 = v4"
        rewards = fix_triple_reward([completion], [triple])
        self.assertEqual(rewards[0], 1.0)

    def test_fix_triple_reward_wrong_0(self):
        """fix_triple task: asserts an incorrect completion gives 0.0 reward"""
        triple = {
            "pre": "v3 > 0 && v4 > 2",
            "program": "v5 = 2\nv3 = v5\nv4 = ((5 + (3 + v3)) + (v4 + v5))\nv4 = 9\nv4 = (v3 - 7)",
            "post": "v5 > 6",
        }
        completion = "v5 = 2\nv3 = v5\nv4 = ((5 + (3 + v3)) + (v4 + v5))\nv5 = v3 + v3"
        rewards = fix_triple_reward([completion], [triple])
        self.assertEqual(rewards[0], 0.0)


if __name__ == "__main__":
    unittest.main()

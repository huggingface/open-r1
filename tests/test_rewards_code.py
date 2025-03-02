import unittest
from open_r1.rewards_internal.code.htgen import totality_check_reward, mk_dataset_totality_check

class TestRewardsCode(unittest.TestCase):
    def test_totality_check_reward_correct(self):
        """Test totality_check_reward"""
        completion = ["True"]
        solution = [True]
        rewards = totality_check_reward(completion, solution)
        self.assertEqual(rewards[0], 1.0)
    def test_mk_dataset_totality_check_format_correct(self):
        """test output format of dataset generator mk_dataset_iter_totality_check"""
        ds = mk_dataset_totality_check(n_examples= 1)
        examples = list(ds)
        prompt = examples[0]['prompt']
        label = examples[0]['ground_truth']
        triple = examples[0]['triple']
        self.assertIsInstance(prompt, str)
        self.assertIsInstance(label, bool)
        self.assertIsInstance(triple, dict)
    def test_totality_check_reward_wrong_format(self):
        """Test totality_check_reward, wrong format"""
        completion = ["The triple is total"]
        solution = [True]
        rewards = totality_check_reward(completion, solution)
        self.assertEqual(rewards[0], 0.0)

if __name__ == "__main__":
    unittest.main()
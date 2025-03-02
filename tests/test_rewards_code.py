import unittest
from open_r1.rewards_internal.code.htgen import totality_check_reward

class TestRewardsCode(unittest.TestCase):
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

if __name__ == "__main__":
    unittest.main()
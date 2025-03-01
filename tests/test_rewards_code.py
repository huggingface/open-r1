from open_r1.rewards.code.htgen import totality_check_reward

class TestRewardsCode(unittest.TestCase):
    def test_totality_check_reward_correct(self):
        """Test totality_check_reward"""
        completion = ["True"]
        solution = [True]
        reward = totality_check_reward(completion, solution)
        self.assertEqual(reward, 1.0)
    def test_totality_check_reward_wrong_format(self):
        """Test totality_check_reward, wrong format"""
        completion = ["The triple is total"]
        solution = [True]
        reward = totality_check_reward(completion, solution)
        self.assertEqual(reward, 0.0)

if __name__ == "__main__":
    unittest.main()
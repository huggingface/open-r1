import unittest

from open_r1.rewards import code_reward


class TestLocalProvider(unittest.TestCase):
    def test_local_python_code_reward(self):
        # Two samples: one correct, one incorrect
        completions = [
            [{"content": "```python\nprint('hello')\n```"}],
            [{"content": "```python\nprint('bye')\n```"}],
        ]

        verification_info = [
            {
                "language": "python",
                "test_cases": [
                    {
                        "input": "",
                        "output": "hello",
                        "type": "stdin_stdout",
                    }
                ],
            },
            {
                "language": "python",
                "test_cases": [
                    {
                        "input": "",
                        "output": "hello",
                        "type": "stdin_stdout",
                    }
                ],
            },
        ]

        rewards = code_reward(
            completions,
            provider_type="local",
            verification_info=verification_info,
            num_parallel=2,
        )

        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)


if __name__ == "__main__":
    unittest.main()


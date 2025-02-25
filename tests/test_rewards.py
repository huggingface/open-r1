import unittest

from open_r1.rewards import (
    accuracy_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    len_reward,
    reasoning_steps_reward,
    tag_count_reward,
)


class TestRewards(unittest.TestCase):
    def test_accuracy_reward_correct_answer(self):
        """Test accuracy_reward with a correct answer."""
        completion = [[{"content": r"\boxed{\frac{63}{400}}"}]]
        solution = [r"\frac{63}{400}"]

        rewards = accuracy_reward(completion, solution)
        self.assertEqual(rewards[0], 1.0)

    def test_accuracy_reward_wrong_answer(self):
        """Test accuracy_reward with an incorrect answer."""
        completion = [[{"content": r"\boxed{\frac{64}{400}}"}]]
        solution = [r"\frac{63}{400}"]

        rewards = accuracy_reward(completion, solution)
        self.assertEqual(rewards[0], 0.0)

    def test_format_reward_correct(self):
        """Test format_reward with correct format."""
        completion = [[{"content": "<think>\nSome reasoning\n</think>\n<answer>\nThe answer\n</answer>"}]]
        rewards = format_reward(completion)
        self.assertEqual(rewards[0], 1.0)

    def test_format_reward_incorrect(self):
        """Test format_reward with incorrect format."""
        incorrect_formats = [
            "<think>Only thinking</think>",
            "<answer>Only answer</answer>",
            "No tags at all",
            "<think>Missing closing</think><answer>Missing closing",
            "<think>Wrong order</answer><answer>Wrong order</think>",
        ]

        for fmt in incorrect_formats:
            completion = [[{"content": fmt}]]
            rewards = format_reward(completion)
            self.assertEqual(rewards[0], 0.0)

    def test_reasoning_steps_reward(self):
        """Test reasoning_steps_reward with various formats."""
        test_cases = [
            # Full credit cases (3 or more steps)
            ("Step 1: First step\nStep 2: Second step\nStep 3: Third step", 1.0),
            ("First, we do this.\nSecond, we do that.\nFinally, we conclude.", 1.0),
            # Partial credit cases (less than 3 steps)
            ("Step 1: Only step", 1 / 3),
            ("First, we do this.\nFinally, we conclude.", 2 / 3),
            # No credit case
            ("Just plain text without any clear steps", 0.0),
        ]

        for content, expected_reward in test_cases:
            completion = [[{"content": content}]]
            rewards = reasoning_steps_reward(completion)
            self.assertAlmostEqual(rewards[0], expected_reward)

    def test_multiple_completions(self):
        """Test handling multiple completions at once."""
        completions = [[{"content": r"\boxed{\frac{63}{400}}"}], [{"content": r"\boxed{\frac{64}{400}}"}]]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]

        rewards = accuracy_reward(completions, solutions)
        self.assertEqual(len(rewards), 2)
        self.assertEqual(rewards[0], 1.0)
        self.assertEqual(rewards[1], 0.0)

    def test_cosine_scaled_reward(self):
        """Test cosine_scaled_reward with various cases."""
        # Test parameters
        test_params = {
            "min_value_wrong": -1.0,
            "max_value_wrong": -0.5,
            "min_value_correct": 0.5,
            "max_value_correct": 1.0,
            "max_len": 100,
        }

        test_cases = [
            # Correct answers with different lengths
            (r"\boxed{\frac{63}{400}}", r"\frac{63}{400}", 20, 0.943),  # Short correct answer
            (r"\boxed{\frac{63}{400}}", r"\frac{63}{400}", 80, 0.547),  # Long correct answer
            # Wrong answers with different lengths
            (r"\boxed{\frac{64}{400}}", r"\frac{63}{400}", 20, -0.942),  # Short wrong answer
            (r"\boxed{\frac{64}{400}}", r"\frac{63}{400}", 80, -0.547),  # Long wrong answer
        ]

        for content, solution, content_len, expected_reward in test_cases:
            # Pad content to desired length
            padded_content = content + " " * (content_len - len(content))
            completion = [[{"content": padded_content}]]

            rewards = get_cosine_scaled_reward(**test_params)(completion, [solution])
            self.assertAlmostEqual(rewards[0], expected_reward, places=2)

    def test_format_reward_specific_multiline(self):
        """Test format_reward with a specific multiline input."""
        inputs = "<think>\nI will count each distinct object in the image:\n1. Purple scooter\n2. Red bicycle\n3. Green motorcycle\n4. Gray sedan\n5. Yellow school bus\n6. Small green double-decker bus\n7. Small red car\n8. Small purple car\n9. Small gray dirt bike\n\nThere are 9 distinct objects in total.\n</think>\n<answer>\n9\n</answer>"
        completion = [[{"content": inputs}]]
        rewards = format_reward(completion)
        self.assertEqual(rewards[0], 1.0)

    def test_same_length_responses(self):
        """Test len_reward when all responses have the same length."""
        completions = [[{"content": r"\boxed{\frac{63}{400}}"}], [{"content": r"\boxed{\frac{64}{400}}"}]]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]

        rewards = len_reward(completions, solutions)
        self.assertEqual(rewards, [0.0, 0.0])

    def test_different_lengths_correct_answers(self):
        """Test len_reward with different length correct answers."""
        completions = [
            [{"content": r"\boxed{\frac{63}{400}}"}],  # shorter
            [{"content": r"\boxed{\frac{63}{400}}  " + "x" * 10}],  # longer
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]

        rewards = len_reward(completions, solutions)
        self.assertGreater(rewards[0], rewards[1])  # shorter answer should get higher reward
        self.assertAlmostEqual(rewards[0], 0.5)  # shortest correct answer gets maximum reward

    def test_different_lengths_incorrect_answers(self):
        """Test len_reward with different length incorrect answers."""
        completions = [
            [{"content": r"\boxed{\frac{64}{400}}"}],  # shorter
            [{"content": r"\boxed{\frac{64}{400}}  " + "x" * 10}],  # longer
        ]
        solutions = [r"\frac{63}{400}", r"\frac{63}{400}"]

        rewards = len_reward(completions, solutions)
        self.assertLessEqual(rewards[0], 0.0)  # incorrect answers should get non-positive rewards
        self.assertLessEqual(rewards[1], 0.0)
        self.assertGreater(rewards[0], rewards[1])  # shorter answer should still be penalized less

    def test_mixed_correctness(self):
        """Test len_reward with mix of correct and incorrect answers of different lengths."""
        completions = [
            [{"content": r"\boxed{\frac{63}{400}}"}],  # correct, shorter
            [{"content": r"\boxed{\frac{63}{400}}  " + "x" * 10}],  # correct, longer
            [{"content": r"\boxed{\frac{64}{400}}"}],  # incorrect, shorter
            [{"content": r"\boxed{\frac{64}{400}}  " + "x" * 10}],  # incorrect, longer
        ]
        solutions = [r"\frac{63}{400}"] * 4

        rewards = len_reward(completions, solutions)

        # Shortest correct answer should get positive reward
        self.assertGreater(rewards[0], 0.0)

        # Longer correct answer might get negative reward:
        self.assertGreater(rewards[2], rewards[1])
        self.assertGreaterEqual(rewards[1], rewards[3])

        # Incorrect answers should get non-positive rewards
        self.assertLessEqual(rewards[2], 0.0)
        self.assertLessEqual(rewards[3], 0.0)

        # Shorter answers should get better rewards within their correctness category
        self.assertGreater(rewards[0], rewards[1])  # correct answers
        self.assertGreater(rewards[2], rewards[3])  # incorrect answers

    def test_unparseable_solution(self):
        """Test len_reward with unparseable solution."""
        completions = [[{"content": r"\boxed{answer}"}], [{"content": r"\boxed{answer} " + "x" * 10}]]
        solutions = ["unparseable_latex", "unparseable_latex"]

        rewards = len_reward(completions, solutions)
        self.assertGreater(rewards[0], rewards[1])  # shorter answer should still get better reward
        self.assertAlmostEqual(rewards[0], 0.5)  # treated as correct, shortest gets maximum reward


class TestRepetitionPenaltyReward(unittest.TestCase):
    def test_positive_max_penalty_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_repetition_penalty_reward(ngram_size=2, max_penalty=1.0)
        with self.assertRaisesRegex(ValueError, "max_penalty 1.5 should not be positive"):
            get_repetition_penalty_reward(ngram_size=2, max_penalty=1.5)

    def test_no_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completions = [[{"content": "this is a test sentence"}]]
        rewards = reward_fn(completions)
        self.assertEqual(rewards, [0.0])

    def test_full_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completions = [[{"content": "this this this this this"}]]

        rewards = reward_fn(completions)
        # (1 - 1/4) * -1 = -0.75
        self.assertEqual(rewards, [-0.75])

    def test_partial_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completions = [[{"content": "this is a this is a test"}]]

        rewards = reward_fn(completions)
        # Unique 2-grams: (this, is), (is, a), (a, this), (a, test).  4 unique out of 6 total
        # (1 - 4/6) * -1 = -1/3 = -0.3333...
        self.assertAlmostEqual(rewards[0], -1 / 3)

    def test_multiple_completions(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
        completions = [
            [{"content": "this is a test"}],
            [{"content": "test test test test"}],
        ]

        rewards = reward_fn(completions)
        # Completion 1:  (this, is, a), (is, a, test) -> 2 unique / 2 total -> (1 - 2/2) * -0.5 = 0
        # Completion 2: (test, test, test) -> 1 unique / 2 total -> (1 - 1/2) * -0.5 = -0.25
        self.assertAlmostEqual(rewards[0], 0.0)
        self.assertAlmostEqual(rewards[1], -0.25)

    def test_empty_completion(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completions = [[{"content": ""}]]
        rewards = reward_fn(completions)
        self.assertEqual(rewards, [0.0])

    def test_different_ngram_size(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-2.0)
        completions = [[{"content": "this is a this is a test"}]]

        rewards = reward_fn(completions)
        self.assertAlmostEqual(rewards[0], -0.4)

    def test_mixed_case(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=2, max_penalty=-1.0)
        completions = [
            [{"content": "This is A Test"}],
            [{"content": "this IS a test"}],
        ]

        rewards = reward_fn(completions)
        # both completions should produce the same reward, because the text gets lowercased
        self.assertAlmostEqual(rewards[0], rewards[1])

    def test_one_word_completion(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "word"}]]

        rewards = reward_fn(completions)
        self.assertEqual(rewards, [0.0])

    def test_two_word_completion(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "two words"}]]

        rewards = reward_fn(completions)
        self.assertEqual(rewards, [0.0])

    def test_three_word_completion(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "three different words"}]]

        rewards = reward_fn(completions)
        self.assertEqual(rewards, [0.0])

    def test_three_word_repetition_completion(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "word word word word"}]]

        rewards = reward_fn(completions)
        self.assertEqual(rewards, [-0.5])

    def test_four_word_completion_with_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "one two one two"}]]

        rewards = reward_fn(completions)
        # ngrams are (one two one) (two one two). unique is 2 and count is 2, therefore (1-1) * -1.
        self.assertEqual(rewards, [0.0])

    def test_five_word_completion_with_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-0.5)
        completions = [[{"content": "A B C A B"}]]

        rewards = reward_fn(completions)
        # (A B C) (B C A) (C A B). unique is 3. count is 3 (1-1) * -.5 = 0
        self.assertEqual(rewards, [0.0])

    def test_six_word_completion_with_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "A B C A B C"}]]

        rewards = reward_fn(completions)
        self.assertEqual(rewards, [-0.25])

    def test_long_completion_with_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "A B C A B C E F G A B C A B C"}]]
        rewards = reward_fn(completions)
        self.assertAlmostEqual(rewards[0], -0.3846, places=4)

    def test_long_completion_without_repetition(self):
        reward_fn = get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)
        completions = [[{"content": "A B C D E F G H I J K L"}]]

        rewards = reward_fn(completions)
        self.assertEqual(rewards, [0.0])

    def test_tag_count_rewards_all_correct(self):
        """Test tag_count_reward with correct tags."""
        completion = [[{"content": "<think>\nSome reasoning\n</think>\n<answer>\nThe answer\n</answer>"}]]
        rewards = tag_count_reward(completion)
        self.assertEqual(rewards[0], 1.0)

    def test_tag_count_rewards_missing_think_begin(self):
        """Test tag_count_reward with missing <think> tag."""
        completion = [[{"content": "Some reasoning\n</think>\n<answer>\nThe answer\n</answer>"}]]
        rewards = tag_count_reward(completion)
        self.assertEqual(rewards[0], 0.75)

    def test_tag_count_rewards_missing_think_end(self):
        """Test tag_count_reward with missing </think> tag."""
        completion = [[{"content": "<think>\nSome reasoning\n<answer>\nThe answer\n</answer>"}]]
        rewards = tag_count_reward(completion)
        self.assertEqual(rewards[0], 0.75)

    def test_tag_count_rewards_missing_answer_begin(self):
        """Test tag_count_reward with missing <answer> tag."""
        completion = [[{"content": "<think>\nSome reasoning\n</think>\nThe answer\n</answer>"}]]
        rewards = tag_count_reward(completion)
        self.assertEqual(rewards[0], 0.75)

    def test_tag_count_rewards_missing_answer_end(self):
        """Test tag_count_reward with missing </answer> tag."""
        completion = [[{"content": "<think>\nSome reasoning\n</think>\n<answer>\nThe answer"}]]
        rewards = tag_count_reward(completion)
        self.assertEqual(rewards[0], 0.75)

    def test_tag_count_rewards_missing_all_tags(self):
        """Test tag_count_reward with missing all tags."""
        completion = [[{"content": "Some reasoning\nThe answer"}]]
        rewards = tag_count_reward(completion)
        self.assertEqual(rewards[0], 0.0)


class TestCodeFormat(unittest.TestCase):
    def test_correct_python_format(self):
        """Test code format reward with correct Python format."""
        completion = [
            [
                {
                    "content": "<think>\nLet's solve this\nStep 1: First step\n</think>\n<answer>\n```python\ndef hello():\n    print('world')\n```\n</answer>"
                }
            ]
        ]
        reward_fn = get_code_format_reward(language="python")
        rewards = reward_fn(completion)
        self.assertEqual(rewards[0], 1.0)

    def test_incorrect_formats(self):
        """Test code format reward with various incorrect formats."""
        incorrect_formats = [
            # Missing think/answer tags
            "```python\ndef hello():\n    print('world')\n```",
            # Missing code block
            "<think>Some thinking</think><answer>Just plain text</answer>",
            # Wrong language
            "<think>Analysis</think><answer>```javascript\nconsole.log('hello');\n```</answer>",
            # Missing language identifier
            "<think>Analysis</think><answer>```\ndef hello(): pass\n```</answer>",
            # Wrong order of tags
            "<answer>```python\ndef hello(): pass\n```</answer><think>Analysis</think>",
        ]

        reward_fn = get_code_format_reward(language="python")
        for fmt in incorrect_formats:
            completion = [[{"content": fmt}]]
            rewards = reward_fn(completion)
            self.assertEqual(rewards[0], 0.0)

    def test_multiple_code_blocks(self):
        """Test format reward with multiple code blocks in think and answer sections."""
        completion = [
            [
                {
                    "content": "<think>\nHere's an example:\n```python\nx = 1\n```\nNow the solution:\n</think>\n<answer>\n```python\ndef solution():\n    return 42\n```\n</answer>"
                }
            ]
        ]
        reward_fn = get_code_format_reward(language="python")
        rewards = reward_fn(completion)
        self.assertEqual(rewards[0], 1.0)

    def test_different_languages(self):
        """Test code format reward with different programming languages."""
        completion = [
            [
                {
                    "content": "<think>\nAnalysis\n</think>\n<answer>\n```javascript\nconsole.log('hello');\n```\n</answer>"
                }
            ]
        ]

        # Test with JavaScript
        js_reward_fn = get_code_format_reward(language="javascript")
        rewards = js_reward_fn(completion)
        self.assertEqual(rewards[0], 1.0)

        # Same completion should fail for Python
        py_reward_fn = get_code_format_reward(language="python")
        rewards = py_reward_fn(completion)
        self.assertEqual(rewards[0], 0.0)

    def test_multiline_code(self):
        """Test format reward with complex multiline code blocks."""
        completion = [
            [
                {
                    "content": "<think>\nHere's the analysis\n</think>\n<answer>\n```python\nclass Solution:\n    def __init__(self):\n        self.value = 42\n        \n    def get_value(self):\n        return self.value\n```\n</answer>"
                }
            ]
        ]
        reward_fn = get_code_format_reward(language="python")
        rewards = reward_fn(completion)
        self.assertEqual(rewards[0], 1.0)


if __name__ == "__main__":
    unittest.main()

"""Reward functions for GRPO training."""

import json
import math
import re
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils import is_e2b_available


if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import Sandbox

    load_dotenv()


class BaseRewardFunction(ABC):
    """Reward function base class"""

    def __init__(self, max_workers: int = 1, *args, **kwargs) -> None:
        self.max_workers = max_workers
        super().__init__()

    @abstractmethod
    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        """Process a single completion and return its score

        Args:
            completion: Single completion content to evaluate

        Returns:
            float:
        """
        raise NotImplementedError("reward_on_single_completion should be impl by subclass")

    def _single_thread_call(self, completions: List[Dict[str, str]], **kwargs) -> List[float]:
        results = []
        for idx, completion in enumerate(completions):
            # prepare per-completion kwargs
            per_completion_kwargs = {}
            for key, value in kwargs.items():
                if isinstance(value, list):
                    per_completion_kwargs[key] = value[idx]
                else:
                    per_completion_kwargs[key] = value
            results.append(self.reward_on_single_completion(completion, **per_completion_kwargs))
        return results

    def __call__(self, completions: List[Dict[str, str]], **kwargs) -> List[float]:
        """Process and score multiple model completions in parallel.

        Args:
            completions: List of model completions, where each completion is a dictionary with 'content' key storing the completion text
            **kwargs: Additional keyword arguments that can include:
                - max_workers: Optional int, maximum number of parallel workers
                - Any other arguments needed by reward_on_single_completion()

        Returns:
            List[float]: A list of reward scores, one for each completion,
                        computed by reward_on_single_completion()

        Raises:
            RuntimeError: If processing any completion fails
        """
        completions = [completion[0]["content"] for completion in completions]

        if "max_workers" in kwargs:
            max_workers = kwargs["max_workers"]
        else:
            max_workers = self.max_workers

        if max_workers == 1:
            return self._single_thread_call(completions, **kwargs)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {}
            for idx, completion in enumerate(completions):
                # prepare per-completion kwargs
                per_completion_kwargs = {}
                for key, value in kwargs.items():
                    if isinstance(value, list):
                        per_completion_kwargs[key] = value[idx]
                    else:
                        per_completion_kwargs[key] = value

                future = executor.submit(self.reward_on_single_completion, completion, **per_completion_kwargs)
                future_to_idx[future] = idx

            results = [None] * len(completions)
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    raise RuntimeError(f"Error processing completion {idx}: {e}") from e

        return results


class AccuracyReward(BaseRewardFunction):
    """Reward function that checks if the completion is the same as the ground truth."""

    def reward_on_single_completion(self, completion: str, solution: str, **kwargs) -> float:
        """Process a single completion and return its score

        Args:
            completion: Single completion content to evaluate
            **kwargs: Must contain 'solution' key with the ground truth solution

        Returns:
            float: 1.0 if the completion matches the ground truth, 0.0 otherwise
        """
        gold_parsed = parse(
            solution,
            extraction_mode="first_match",
            extraction_config=[LatexExtractionConfig()],
        )

        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                completion,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed="all",
                            units=True,
                        ),
                        # Ensures that boxed is tried first
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            # Reward 1 if the content is the same as the ground truth, 0 otherwise
            reward = float(verify(answer_parsed, gold_parsed))
        else:
            # If the gold solution is not parseable, we reward 1 to skip this example
            reward = 1.0
            print("Failed to parse gold solution: ", solution)

        return reward


class FormatReward(BaseRewardFunction):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""

    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        """Process a single completion and return its score

        Args:
            completion: Single completion content to evaluate
            **kwargs: Additional arguments (unused)

        Returns:
            float: 1.0 if the completion has correct format, 0.0 otherwise
        """
        pattern = r"^<think>.*?</think>\s*<answer>.*?</answer>$"
        match = re.match(pattern, completion, re.DOTALL | re.MULTILINE)
        return 1.0 if match else 0.0


class ReasoningStepsReward(BaseRewardFunction):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """

    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        """Process a single completion and return its score based on number of reasoning steps.

        Args:
            completion: Single completion content to evaluate
            **kwargs: Additional arguments (unused)

        Returns:
            float: Score between 0.0 and 1.0 based on number of reasoning steps found
        """
        pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
        matches = len(re.findall(pattern, completion))

        # Magic nubmer 3 to encourage 3 steps and more, otherwise partial reward
        return min(1.0, matches / 3)


class LengthReward(BaseRewardFunction):
    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        raise NotImplementedError("LengthReward don't need to impl reward_on_single_completion")

    def __call__(self, completions: List[Dict[str, str]], solution: list[str], **kwargs) -> List[float]:
        """Compute length-based rewards to discourage overthinking and promote token efficiency.

        Taken from from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        Returns:
            List of rewards where:
            - For correct answers: reward = 0.5 - (len - min_len)/(max_len - min_len)
            - For incorrect answers: reward = min(0, 0.5 - (len - min_len)/(max_len - min_len))
        """
        contents = [completion[0]["content"] for completion in completions]

        # First check correctness of answers
        correctness = []
        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                # Skip unparseable examples
                correctness.append(True)  # Treat as correct to avoid penalizing
                print("Failed to parse gold solution: ", sol)
                continue

            answer_parsed = parse(
                content,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(
                            nits=False,
                            malformed_operators=False,
                            basic_latex=True,
                            equations=True,
                            boxed=True,
                            units=True,
                        ),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            correctness.append(verify(answer_parsed, gold_parsed))

        # Calculate lengths
        lengths = [len(content) for content in contents]
        min_len = min(lengths)
        max_len = max(lengths)

        # If all responses have the same length, return zero rewards
        if max_len == min_len:
            return [0.0] * len(completions)

        rewards = []
        for length, is_correct in zip(lengths, correctness):
            lambda_val = 0.5 - (length - min_len) / (max_len - min_len)

            if is_correct:
                reward = lambda_val
            else:
                reward = min(0, lambda_val)

            rewards.append(float(reward))

        return rewards


class CosineScaledReward(BaseRewardFunction):
    def __init__(
        self,
        min_value_wrong: float = -1.0,
        max_value_wrong: float = -0.5,
        min_value_correct: float = 0.5,
        max_value_correct: float = 1.0,
        max_len: int = 1000,
        max_workers: int = 1,
    ):
        """Initialize CosineScaledReward.

        Args:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        super().__init__(max_workers=max_workers)
        self.min_value_wrong = min_value_wrong
        self.max_value_wrong = max_value_wrong
        self.min_value_correct = min_value_correct
        self.max_value_correct = max_value_correct
        self.max_len = max_len

    def reward_on_single_completion(self, completion: str, solution: str, **kwargs) -> float:
        gold_parsed = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
        if len(gold_parsed) == 0:
            print("Failed to parse gold solution: ", solution)
            return 1.0  # Skip unparseable examples

        answer_parsed = parse(
            completion,
            extraction_config=[
                LatexExtractionConfig(
                    normalization_config=NormalizationConfig(
                        nits=False,
                        malformed_operators=False,
                        basic_latex=True,
                        equations=True,
                        boxed=True,
                        units=True,
                    ),
                    boxed_match_priority=0,
                    try_extract_without_anchor=False,
                )
            ],
            extraction_mode="first_match",
        )

        is_correct = verify(answer_parsed, gold_parsed)
        gen_len = len(completion)

        # Apply cosine scaling based on length
        progress = gen_len / self.max_len
        cosine = math.cos(progress * math.pi)

        if is_correct:
            min_value = self.min_value_correct
            max_value = self.max_value_correct
        else:
            # Swap min/max for incorrect answers
            min_value = self.max_value_wrong
            max_value = self.min_value_wrong

        reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
        return float(reward)


class RepetitionPenaltyReward(BaseRewardFunction):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py
    """

    def __init__(self, ngram_size: int, max_penalty: float, max_workers: int = 1):
        """
        Args:
            ngram_size: size of the n-grams
            max_penalty: Maximum (negative) penalty for wrong answers
        """
        super().__init__(max_workers=max_workers)
        if max_penalty > 0:
            raise ValueError(f"max_penalty {max_penalty} should not be positive")
        self.ngram_size = ngram_size
        self.max_penalty = max_penalty

    def _zipngram(self, text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        """
        Reward function that penalizes repetitions

        Args:
            completion: Model completion text
        """
        if completion == "":
            return 0.0

        if len(completion.split()) < self.ngram_size:
            return 0.0

        ngrams = set()
        total = 0
        for ng in self._zipngram(completion, self.ngram_size):
            ngrams.add(ng)
            total += 1

        scaling = 1 - len(ngrams) / total
        reward = scaling * self.max_penalty
        return float(reward)


class CodeReward(BaseRewardFunction):
    """Reward function that evaluates code snippets using the E2B code interpreter."""

    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        passed = 0
        total = len(test_cases)
        exec_timeout = 5

        for case in test_cases:
            process = subprocess.run(
                ["python3", "-c", code],
                input=case["input"],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )

            if process.returncode != 0:  # Error in execution
                continue

            output = process.stdout.strip()
            if output.strip() == case["output"].strip():
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    def __init__(self, max_workers: int = 1):
        super().__init__(max_workers=max_workers)
        if not is_e2b_available():
            raise ImportError(
                "E2B is not available and required for this reward function. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
            )

    def _extract_code(self, completion: str) -> str:
        pattern = re.compile(r"```python\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ""
        return extracted_answer

    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        """
        Evaluate code snippets using test cases.

        Args:
            completions: List of model completions
            **kwargs: Must contain 'verification_info' with test cases

        Returns:
            List of reward scores between 0 and 1
        """

        code = self._extract_code(completion)
        verification_info = kwargs["verification_info"]
        script = self.evaluation_script_template.format(
            code=json.dumps(code), test_cases=json.dumps(json.dumps(verification_info["test_cases"]))
        )

        try:
            with Sandbox(timeout=30, request_timeout=3) as sbx:
                execution = sbx.run_code(script, language=verification_info["language"])
                try:
                    score = float(execution.text)
                except (TypeError, ValueError):
                    score = 0.0
                return score
        except Exception as e:
            print(f"Error from E2B executor: {e}")
            return 0, 0


class CodeFormatReward(BaseRewardFunction):
    """Format reward function specifically for code responses."""

    def __init__(self, language: str = "python", max_workers: int = 1):
        """
        Initialize the code format reward function.

        Args:
            language: Programming language supported by E2B
        """
        super().__init__(max_workers=max_workers)
        self.pattern = rf"^<think>.*?</think>\s*<answer>.*?```{language}\n.*?```.*?</answer>$"

    def reward_on_single_completion(self, completion: str, **kwargs) -> float:
        """
        Check if completions match the expected code format.

        Args:
            completions: List of model completions

        Returns:
            List of 1.0 for matching format, 0.0 otherwise
        """
        match = re.match(self.pattern, completion, re.DOTALL | re.MULTILINE)
        return 1.0 if match else 0.0

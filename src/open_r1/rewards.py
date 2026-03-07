# coding=utf-8
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

"""Reward functions for GRPO training."""

import ast
import asyncio
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import unittest
from functools import partial, update_wrapper
from typing import Callable, Dict, Literal, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.code_providers import get_provider
from .utils.competitive_programming import (
    SubtaskResult,
    add_includes,
    get_morph_client_from_env,
    get_piston_client_from_env,
)
from .utils.competitive_programming import patch_code as cf_patch_code
from .utils.competitive_programming import score_submission as cf_score_submission
from .utils.competitive_programming import score_subtask


def accuracy_reward(completions: list[list[dict[str, str]]], solution: list[str], **kwargs) -> list[Optional[float]]:
    """Reward function that checks if the completion is the same as the ground truth."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol in zip(contents, solution):
        gold_parsed = parse(
            sol,
            extraction_mode="first_match",
        )
        if len(gold_parsed) != 0:
            # We require the answer to be provided in correct latex (no malformed operators)
            answer_parsed = parse(
                content,
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
            # Compute binary rewards if verifiable, `None` otherwise to skip this example
            try:
                reward = float(verify(gold_parsed, answer_parsed))
            except Exception as e:
                print(f"verify failed: {e}, answer: {answer_parsed}, gold: {gold_parsed}")
                reward = None
        else:
            # If the gold solution is not parseable, we assign `None` to skip this example
            reward = None
            print("Failed to parse gold solution: ", sol)
        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the reasoning process is enclosed within <think> and </think> tags, while the final answer is enclosed within <answer> and </answer> tags."""
    pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?\n</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of think and answer tags associated with `format_reward()`.

    Adapted from: https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb#file-grpo_demo-py-L90
    """

    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<think>\n") == 1:
            count += 0.25
        if text.count("\n</think>\n") == 1:
            count += 0.25
        if text.count("\n<answer>\n") == 1:
            count += 0.25
        if text.count("\n</answer>") == 1:
            count += 0.25
        return count

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]


def reasoning_steps_reward(completions, **kwargs):
    r"""Reward function that checks for clear step-by-step reasoning.
    Regex pattern:
        Step \d+: - matches "Step 1:", "Step 2:", etc.
        ^\d+\. - matches numbered lists like "1.", "2.", etc. at start of line
        \n- - matches bullet points with hyphens
        \n\* - matches bullet points with asterisks
        First,|Second,|Next,|Finally, - matches transition words
    """
    pattern = r"(Step \d+:|^\d+\.|\n-|\n\*|First,|Second,|Next,|Finally,)"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [len(re.findall(pattern, content)) for content in completion_contents]

    # Magic number 3 to encourage 3 steps and more, otherwise partial reward
    return [min(1.0, count / 3) for count in matches]


def len_reward(completions: list[Dict[str, str]], solution: list[str], **kwargs) -> float:
    """Compute length-based rewards to discourage overthinking and promote token efficiency.

    Taken from the Kimi 1.5 tech report: https://huggingface.co/papers/2501.12599

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


def get_cosine_scaled_reward(
    min_value_wrong: float = -1.0,
    max_value_wrong: float = -0.5,
    min_value_correct: float = 0.5,
    max_value_correct: float = 1.0,
    max_len: int = 1000,
):
    def cosine_scaled_reward(completions, solution, **kwargs):
        """Reward function that scales based on completion length using a cosine schedule.

        Shorter correct solutions are rewarded more than longer ones.
        Longer incorrect solutions are penalized less than shorter ones.

        Args:
            completions: List of model completions
            solution: List of ground truth solutions

        This function is parameterized by the following arguments:
            min_value_wrong: Minimum reward for wrong answers
            max_value_wrong: Maximum reward for wrong answers
            min_value_correct: Minimum reward for correct answers
            max_value_correct: Maximum reward for correct answers
            max_len: Maximum length for scaling
        """
        contents = [completion[0]["content"] for completion in completions]
        rewards = []

        for content, sol in zip(contents, solution):
            gold_parsed = parse(
                sol,
                extraction_mode="first_match",
                extraction_config=[LatexExtractionConfig()],
            )
            if len(gold_parsed) == 0:
                rewards.append(1.0)  # Skip unparseable examples
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

            is_correct = verify(answer_parsed, gold_parsed)
            gen_len = len(content)

            # Apply cosine scaling based on length
            progress = gen_len / max_len
            cosine = math.cos(progress * math.pi)

            if is_correct:
                min_value = min_value_correct
                max_value = max_value_correct
            else:
                # Swap min/max for incorrect answers
                min_value = max_value_wrong
                max_value = min_value_wrong

            reward = min_value + 0.5 * (max_value - min_value) * (1.0 + cosine)
            rewards.append(float(reward))

        return rewards

    return cosine_scaled_reward


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float, language: str = "en"):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://huggingface.co/papers/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    language: Language of the text, defaults to `en`. Used to choose the way to split the text into n-grams.
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    if language == "en":

        def zipngram(text: str, ngram_size: int):
            words = text.lower().split()
            return zip(*[words[i:] for i in range(ngram_size)]), words

    elif language == "zh":
        from transformers.utils.import_utils import _is_package_available

        if not _is_package_available("jieba"):
            raise ValueError("Please install jieba to use Chinese language")

        def zipngram(text: str, ngram_size: int):
            import jieba

            seg_list = list(jieba.cut(text))
            return zip(*[seg_list[i:] for i in range(ngram_size)]), seg_list

    else:
        raise ValueError(
            f"Word splitting for language `{language}` is not yet implemented. Please implement your own zip-ngram function."
        )

    def repetition_penalty_reward(completions, **kwargs) -> float:
        """
        reward function the penalizes repetitions
        ref implementation: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

        Args:
            completions: List of model completions
        """

        contents = [completion[0]["content"] for completion in completions]
        rewards = []
        for completion in contents:
            if completion == "":
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            ngram_array, words = zipngram(completion, ngram_size)

            if len(words) < ngram_size:
                rewards.append(0.0)
                continue

            for ng in ngram_array:
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * max_penalty
            rewards.append(reward)
        return rewards

    return repetition_penalty_reward


def _init_event_loop():
    """Initialize or get the current event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", **kwargs) -> list[float]:
    """Reward function that evaluates IOI problems using a specified execution client.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/ioi

    Args:
        completions: List of model completions to evaluate
        test_batch_size: Evaluate these many test cases in parallel, then check if any of them failed (0 score):
                       if so stop evaluating; otherwise continue with the next batch of test cases.
        provider_type: The execution provider to use (default: "piston"). Supported values: "piston", "morph"
        **kwargs: Additional arguments passed from the dataset
    """
    # Get the appropriate client based on provider_type
    if provider_type == "morph":
        execution_client = get_morph_client_from_env()
    else:
        # for info on setting up piston workers, see slurm/piston/README.md
        execution_client = get_piston_client_from_env()

    code_snippets = [
        # note: grading is automatically skipped if no code is extracted
        add_includes(extract_code(completion[-1]["content"], "cpp"), problem_id)
        for completion, problem_id in zip(completions, kwargs["id"])
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from {provider_type} worker: {e}")
            return SubtaskResult()

    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                score_subtask(
                    execution_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def cf_code_reward(
    completions,
    test_batch_size: int = 1,
    patch_code: bool = False,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    **kwargs,
) -> list[float]:
    """Reward function that evaluates Codeforces problems using Piston+our CF package.

    Assumes the dataset has the same format as hf.co/datasets/open-r1/codeforces (verifiable-prompts subset)

    test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    """
    # for info on setting up piston workers, see slurm/piston/README.md
    piston_client = get_piston_client_from_env()

    languages = kwargs["language"] if "language" in kwargs else [None] * len(completions)
    code_snippets = [
        # note: grading is automatically skipped if a problem has no tests
        cf_patch_code(extract_code(completion[-1]["content"], language), language)
        if patch_code
        else extract_code(completion[-1]["content"], language)
        for completion, language in zip(completions, languages)
    ]

    async def run_catch_exceptions(task):
        try:
            return await task
        except Exception as e:
            print(f"Error from Piston worker: {e}")
            return None

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(
                cf_score_submission(
                    piston_client,
                    problem_data,
                    code,
                    test_batch_size=test_batch_size,
                    scoring_mode=scoring_mode,
                    submission_language=problem_data.get("language", None),
                )
            )
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return results


def extract_code(completion: str, language: str | None = "python") -> str:
    if language is None:
        return ""
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    rewards = code_reward(
        completions,
        num_parallel=num_parallel,
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs,
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward(
    completions,
    num_parallel: int = 2,
    provider_type: str = "e2b",
    enforce_same_language: bool = False,
    **kwargs,
) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.

    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
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

            # TODO: implement a proper validator to compare against ground truth. For now we just check for exact string match on each line of stdout.
            all_correct = True
            for line1, line2 in zip(output.split('\\n'), case['output'].split('\\n')):
                all_correct = all_correct and line1.strip() == line2.strip()

            if all_correct:
                passed += 1

        success_rate = (passed / total)
        return success_rate

    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """

    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]

    template = evaluation_script_template

    scripts = [
        template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]

    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        **kwargs,
    )

    return execution_provider.execute_scripts(scripts, ["python"] * len(scripts))

def unittest_reward(completions, solution, is_correct, **kwargs) -> list[float]:
    """
    Reward function that executes the generated unit tests (completion) against the
    provided solution code. Returns the number of passed tests.
    """
    contents = [completion[0]["content"] for completion in completions]
    rewards = []

    for content, sol in zip(contents, solution):
        # Extract code from completion (the tests)
        test_code = extract_code(content)
        # Fallback: if extract_code returns empty but content looks like code, use it
        if not test_code and ("def " in content or "class " in content):
             test_code = content

        # Extract code from solution (the solve function)
        sol_code = extract_code(sol)
        if not sol_code:
            # Assume raw code if no markdown found in solution
            sol_code = sol

        if not test_code.strip():
            rewards.append(0.0)
            continue

        script_content = f"""
import unittest
import sys
from typing import *
# Solution Code
{sol_code}
# Test Code
{test_code}
if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    passed = result.testsRun - len(result.errors) - len(result.failures)
    print(f"METRICS:{{passed}}:{{result.testsRun}}")
"""
        script_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(script_content)
                script_path = f.name

            # Execute the script
            result = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=5
            )

            output = result.stdout
            match = re.search(r"METRICS:(\d+):(\d+)", output)
            if match:
                passed = int(match.group(1))
                total = int(match.group(2))
                reward = float(passed) / float(total) if total > 0 else 0.0
                if not is_correct:
                    reward = 1 - reward
                rewards.append(reward)
            else:
                rewards.append(0.0)

        except Exception:
            rewards.append(0.0)
        finally:
            if script_path and os.path.exists(script_path):
                os.remove(script_path)

    return rewards


def _run_unittest_with_per_test_metrics(
    test_code: str,
    sol_code: str,
    timeout: int = 5,
) -> dict[str, int]:
    """
    Execute the full unittest suite once and return a per-test pass/fail map.

    The returned dict maps each test id (e.g. '__main__.Test.test1') to:
      1 if the test passed,
      0 if the test failed or errored.
    """
    if not test_code.strip() or not sol_code.strip():
        return {}

    script_content = f"""
import unittest
import sys
from typing import *

# Solution Code
{sol_code}

# Test Code
{test_code}

def _flatten_suite(suite: unittest.TestSuite):
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from _flatten_suite(item)
        else:
            yield item

if __name__ == '__main__':
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromModule(sys.modules[__name__])
    all_tests = list(_flatten_suite(suite))
    runner = unittest.TextTestRunner(verbosity=0, stream=sys.stdout)
    result = runner.run(suite)
    fail_ids = {{case.id() for case, _ in result.failures}}
    error_ids = {{case.id() for case, _ in result.errors}}

    print("METRICS_START")
    for t in all_tests:
        tid = t.id()
        status = 0 if (tid in fail_ids or tid in error_ids) else 1
        print(f"{{tid}}:{{status}}")
    print("METRICS_END")
"""
    script_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        metrics: dict[str, int] = {}
        in_block = False
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped == "METRICS_START":
                in_block = True
                continue
            if stripped == "METRICS_END":
                break
            if not in_block:
                continue
            # Line format: "<test_id>:<0-or-1>"
            try:
                test_id, status_str = stripped.rsplit(":", 1)
                metrics[test_id.strip()] = int(status_str)
            except Exception:
                continue

        return metrics
    except Exception:
        return {}
    finally:
        if script_path and os.path.exists(script_path):
            os.remove(script_path)


def _compute_bik_matrices(
    test_code: str,
    solutions: list[dict],
    timeout: int = 5,
) -> tuple[list[str], list[list[float]], list[list[float]]]:
    """
    Compute B_ik values grouped by semantic test f_k.

    Returns:
        canonical_test_ids: Ordered semantic test ids (f_k).
        B_correct_matrix: K x M+ matrix; row k contains B_ik over correct solutions.
        B_wrong_matrix: K x M- matrix; row k contains B_ik over wrong solutions.
    """
    if not test_code.strip() or not solutions:
        return [], [], []

    correct_indices = [i for i, s in enumerate(solutions) if s["is_correct"]]
    wrong_indices = [i for i, s in enumerate(solutions) if not s["is_correct"]]

    per_solution_metrics: list[dict[str, int]] = []
    canonical_test_ids: list[str] = []

    for sol in solutions:
        sol_code = sol["solve_func"]
        metrics = _run_unittest_with_per_test_metrics(test_code, sol_code, timeout=timeout)
        per_solution_metrics.append(metrics)
        # Use the first non-empty metrics dict to fix semantic-test order.
        if not canonical_test_ids and metrics:
            canonical_test_ids = list(metrics.keys())

    if not canonical_test_ids:
        return [], [], []

    B_correct_matrix: list[list[float]] = []
    B_wrong_matrix: list[list[float]] = []

    for test_id in canonical_test_ids:
        B_correct_row = [float(per_solution_metrics[idx].get(test_id, 0)) for idx in correct_indices]
        B_wrong_row = [float(per_solution_metrics[idx].get(test_id, 0)) for idx in wrong_indices]
        B_correct_matrix.append(B_correct_row)
        B_wrong_matrix.append(B_wrong_row)

    return canonical_test_ids, B_correct_matrix, B_wrong_matrix


def cross_solution_unittest_reward(
    completion_text: str,
    solutions: list[dict],
    lambda_1: float = 0.1,
    lambda_2: float = 0.1,
    lambda_t: float = 0.5,
    timeout: int = 5,
) -> float:
    """
    Compute reward for generated unit tests using Equation 10 formulation.

    Each semantic test function f_k (individual test_xxx method) is evaluated
    against all solutions. The reward combines:
      R^1_{f_k} (validity):       encourages f_k to pass all correct solutions
      R^-_{f_k} (discrimination): encourages f_k to fail at least one wrong solution

    Formulas (Equation 10):
      R^1_{f_k} = prod(B_ik, i in correct) + (lambda_1 / M+) * sum(B_ik, i in correct)
      R^-_{f_k} = prod(B_ik, i in correct) * (1 - prod(B_ik, i in wrong))
                  - (lambda_2 / M-) * sum(B_ik, i in wrong)
      R_{f_k}   = lambda_t * R^1 + (1 - lambda_t) * R^-
      R         = mean(R_{f_k}) over all k

    Args:
        completion_text: Model completion containing unittest code with test methods.
        solutions: List of dicts with keys "solve_func" (str) and "is_correct" (bool).
        lambda_1: Soft validity coefficient (default 0.1).
        lambda_2: Soft discrimination penalty coefficient (default 0.1).
        lambda_t: Weight between validity and discrimination (default 0.5).
        timeout: Timeout in seconds for each test execution.

    Returns:
        Float reward in roughly [-lambda_2, 1 + lambda_1].
    """
    test_code = extract_code(completion_text)
    if not test_code and ("def " in completion_text or "class " in completion_text):
        test_code = completion_text

    if not test_code.strip():
        return 0.0

    # Prepare index mapping for correct / wrong solutions
    correct_indices = [i for i, s in enumerate(solutions) if s["is_correct"]]
    wrong_indices = [i for i, s in enumerate(solutions) if not s["is_correct"]]
    M_plus = len(correct_indices)
    M_minus = len(wrong_indices)

    if M_plus + M_minus == 0:
        return 0.0

    canonical_test_ids, B_correct_matrix, B_wrong_matrix = _compute_bik_matrices(
        test_code=test_code,
        solutions=solutions,
        timeout=timeout,
    )

    if not canonical_test_ids:
        return 0.0

    K = len(canonical_test_ids)

    R_fk_scores: list[float] = []
    B_matrix: list[list[float]] = []  # B_matrix[k] = B_correct_k + B_wrong_k
    R_details: list[tuple[float, float, float]] = []  # (R1, R_minus, R_fk) per f_k

    for k, _ in enumerate(canonical_test_ids):
        B_correct = B_correct_matrix[k]
        B_wrong = B_wrong_matrix[k]

        B_matrix.append(B_correct + B_wrong)

        # R^1_{f_k} (validity)
        prod_correct = 1.0
        for b in B_correct:
            prod_correct *= b
        sum_correct = sum(B_correct)
        R1 = prod_correct + (lambda_1 / M_plus * sum_correct if M_plus > 0 else 0.0)

        # R^-_{f_k} (discrimination)
        prod_wrong = 1.0
        for b in B_wrong:
            prod_wrong *= b
        sum_wrong = sum(B_wrong)
        R_minus = prod_correct * (1.0 - prod_wrong) - (
            lambda_2 / M_minus * sum_wrong if M_minus > 0 else 0.0
        )

        # R_{f_k} = lambda_t * R^1 + (1 - lambda_t) * R^-
        R_fk = lambda_t * R1 + (1.0 - lambda_t) * R_minus
        R_fk_scores.append(R_fk)
        R_details.append((R1, R_minus, R_fk))

    # ── Print B_ik matrix and per-f_k reward breakdown ──
    header = "B_ik matrix (rows=f_k, cols=solutions [correct | wrong]):"
    col_labels = [f"s+{i}" for i in range(M_plus)] + [f"s-{i}" for i in range(M_minus)]
    col_header = "        " + "  ".join(f"{c:>4}" for c in col_labels)
    print(header)
    print(col_header)
    for k, row in enumerate(B_matrix):
        vals = "  ".join(f"{int(v):>4}" for v in row)
        print(f"  f_{k+1:>2}:  {vals}")
    print("Per-f_k rewards:")
    for k, (r1, rm, rfk) in enumerate(R_details):
        print(f"  f_{k+1:>2}: R^1={r1:.4f}, R^-={rm:.4f}, R_fk={rfk:.4f}")
    final_reward = sum(R_fk_scores) / K
    print(f"Final reward = {final_reward:.4f} (K={K})")

    return final_reward


def cross_solution_unittest_reward_v2(
    completion_text: str,
    solutions: list[dict],
    timeout: int = 5,
) -> float:
    """
    Compute reward for generated unit tests (v2).

    Changes vs v1:
      - Remove lambda_1 soft-validity term from R^1.
      - Remove lambda_2 soft-discrimination penalty from R^-.
      - Combine as direct sum: R_{f_k} = R^1 + R^- (no lambda_t weighting).

    Formulas:
      R^1_{f_k} = prod(B_ik, i in correct)
      R^-_{f_k} = prod(B_ik, i in correct) * (1 - prod(B_ik, i in wrong))
      R_{f_k}   = R^1 + R^-
      R         = mean(R_{f_k}) over all k
    """
    test_code = extract_code(completion_text)
    if not test_code and ("def " in completion_text or "class " in completion_text):
        test_code = completion_text

    if not test_code.strip():
        return 0.0

    correct_indices = [i for i, s in enumerate(solutions) if s["is_correct"]]
    wrong_indices = [i for i, s in enumerate(solutions) if not s["is_correct"]]
    M_plus = len(correct_indices)
    M_minus = len(wrong_indices)

    if M_plus + M_minus == 0:
        return 0.0

    canonical_test_ids, B_correct_matrix, B_wrong_matrix = _compute_bik_matrices(
        test_code=test_code,
        solutions=solutions,
        timeout=timeout,
    )
    if not canonical_test_ids:
        return 0.0

    K = len(canonical_test_ids)
    R_fk_scores: list[float] = []
    B_matrix: list[list[float]] = []  # B_matrix[k] = B_correct_k + B_wrong_k
    R_details: list[tuple[float, float, float]] = []  # (R1, R_minus, R_fk) per f_k

    for k, _ in enumerate(canonical_test_ids):
        B_correct = B_correct_matrix[k]
        B_wrong = B_wrong_matrix[k]
        B_matrix.append(B_correct + B_wrong)

        prod_correct = 1.0
        for b in B_correct:
            prod_correct *= b
        R1 = prod_correct

        prod_wrong = 1.0
        for b in B_wrong:
            prod_wrong *= b
        R_minus = prod_correct * (1.0 - prod_wrong)

        R_fk = R1 + R_minus
        R_fk_scores.append(R_fk)
        R_details.append((R1, R_minus, R_fk))

    header = "B_ik matrix (rows=f_k, cols=solutions [correct | wrong]):"
    col_labels = [f"s+{i}" for i in range(M_plus)] + [f"s-{i}" for i in range(M_minus)]
    col_header = "        " + "  ".join(f"{c:>4}" for c in col_labels)
    print(header)
    print(col_header)
    for k, row in enumerate(B_matrix):
        vals = "  ".join(f"{int(v):>4}" for v in row)
        print(f"  f_{k+1:>2}:  {vals}")
    print("Per-f_k rewards (v2):")
    for k, (r1, rm, rfk) in enumerate(R_details):
        print(f"  f_{k+1:>2}: R^1={r1:.4f}, R^-={rm:.4f}, R_fk={rfk:.4f}")
    final_reward = sum(R_fk_scores) / K
    print(f"Final reward v2 = {final_reward:.4f} (K={K})")

    return final_reward


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """

    def code_format_reward(completions, **kwargs):
        # if there is a language field, use it instead of the default language. This way we can have mixed language training.
        languages = kwargs["language"] if "language" in kwargs else [language] * len(completions)

        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [
            re.match(
                rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{sample_language}.*?```.*?\n</answer>$",
                content,
                re.DOTALL | re.MULTILINE,
            )
            for content, sample_language in zip(completion_contents, languages)
        ]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


def get_soft_overlong_punishment(max_completion_len, soft_punish_cache):
    """
    Reward function that penalizes overlong completions. It is used to penalize overlong completions,
    but not to reward shorter completions. Reference: Eq. (13) from the DAPO paper (https://huggingface.co/papers/2503.14476)

    Args:
        max_completion_len: Maximum length of the completion
        soft_punish_cache: Minimum length of the completion. If set to 0, no minimum length is applied.
    """

    def soft_overlong_punishment_reward(completion_ids: list[list[int]], **kwargs) -> list[float]:
        """Reward function that penalizes overlong completions."""
        rewards = []
        for ids in completion_ids:
            completion_length = len(ids)
            if completion_length <= max_completion_len - soft_punish_cache:
                rewards.append(0.0)
            elif max_completion_len - soft_punish_cache < completion_length <= max_completion_len:
                rewards.append((max_completion_len - soft_punish_cache - completion_length) / soft_punish_cache)
            else:
                rewards.append(-1.0)
        return rewards

    return soft_overlong_punishment_reward


def get_reward_funcs(script_args) -> list[Callable]:
    REWARD_FUNCS_REGISTRY = {
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "cf_code": update_wrapper(
            partial(
                cf_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                scoring_mode=script_args.code_eval_scoring_mode,
            ),
            cf_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
        "soft_overlong_punishment": get_soft_overlong_punishment(
            max_completion_len=script_args.max_completion_len,
            soft_punish_cache=script_args.soft_punish_cache,
        ),
        "unittest": unittest_reward,
        # "cross_solution_unittest": cross_solution_unittest_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs

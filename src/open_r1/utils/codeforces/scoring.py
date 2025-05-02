import asyncio
from itertools import islice
from typing import Literal

from .piston_client import PistonClient

def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        return iterable
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch


async def score_single_test_case(
    client: PistonClient, problem_data: dict, test_input: str, test_output: str, submission: str, submission_language: str = 'cpp'
) -> tuple[str, str]:
    try:
        result = await client._send_execute({
            "files": [
                {
                    "name": f"main.{submission_language}",
                    "content": submission
                },
                *([{
                    "name": "checker.py",
                    "content": problem_data['generated_checker']
                }] if problem_data['generated_checker'] else []),
                {
                    "name": "input.txt",
                    "content": test_input
                },
                {
                    "name": "correct_output.txt",
                    "content": test_output
                },
                {
                    "name": "grader_config",
                    "content": "\n".join(
                        f"{key}={value}" for key, value in {
                            "TIME_LIMIT": problem_data['time_limit'],
                            "MEMORY_LIMIT": problem_data['memory_limit'],
                            "INPUT_MODE": problem_data['input_mode']
                        }.items()
                    )
                }
            ],
            "run_timeout": (problem_data['time_limit'] + 3) * 1000
            # +3 seconds hard limit. time limits are handled by the codeforces script
        }, language="cf_python3" if submission_language == "python" else "c++17")
    except Exception as e:
        print(f"Error scoring submission: {e}")
        return False

    return result

async def score_submission(
    client: PistonClient,
    problem_data: dict,
    submission: str,
    test_batch_size: int = 1,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    no_compile_reward: float = -0.1,
    no_submission_reward: float = -1.0,
    submission_language: str = "cpp"
) -> float:
    test_cases = problem_data["official_tests"]
    # invalid/not a coding problem
    if test_cases is None or len(test_cases) == 0:
        return None
    # no code extracted
    if not submission:
        return no_submission_reward

    passed_test_cases = 0
    # run one batch, check if any of them failed (0 score): if so stop evaluating (assuming non partial score); otherwise continue with the next batch of test cases.
    for test_batch_to_run in (batched(test_cases, test_batch_size) if test_batch_size >= 1 else [test_cases]):
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client, problem_data, test_case["input"], test_case["output"], submission, submission_language
                    )
                )
                for test_case in test_batch_to_run
            ]
        )
        if any(result and result['compile']['code'] != 0 for result in results):
            return no_compile_reward

        tests_passed_results = [result and result['run']['code'] == 0 and result['run']['stdout'].strip() == "1" for result in results]
        if scoring_mode == "pass_fail" and any(not test_passed for test_passed in tests_passed_results):
            break
        passed_test_cases += sum(1 for test_passed in tests_passed_results if test_passed)

    pass_fail_score = 1.0 if passed_test_cases == len(test_cases) else 0.0

    if scoring_mode == "pass_fail":
        return pass_fail_score
    elif scoring_mode == "partial":
        return passed_test_cases / len(test_cases)
    elif scoring_mode == "weighted_sum":
        return pass_fail_score + 0.1 * (passed_test_cases / len(test_cases))
    else:
        raise ValueError(f"Invalid scoring mode: {scoring_mode}")

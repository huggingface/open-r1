import asyncio
from itertools import islice

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

    return result and result['compile']['code'] == 0 and result['run']['code'] == 0 and result['run']['stdout'].strip() == "1"

async def score_submission(
    client: PistonClient,
    problem_data: dict,
    submission: str,
    test_batch_size: int = 1,
    partial_scoring: bool = False,
) -> float:
    test_cases = problem_data["official_tests"]
    # we skip submissions where no code was extracted
    if not submission or len(test_cases) == 0:
        return 0.0

    passed_test_cases = 0
    # run one batch, check if any of them failed (0 score): if so stop evaluating (assuming non partial score); otherwise continue with the next batch of test cases.
    for test_batch_to_run in (batched(test_cases, test_batch_size) if test_batch_size >= 1 else [test_cases]):
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client, problem_data, test_case["input"], test_case["output"], submission
                    )
                )
                for test_case in test_batch_to_run
            ]
        )
        if not partial_scoring and any(not test_passed for test_passed in results):
            break
        passed_test_cases += sum(1 for test_passed in results if test_passed)

    return (1.0 if passed_test_cases == len(test_cases) else 0.0) if not partial_scoring \
        else passed_test_cases / len(test_cases)

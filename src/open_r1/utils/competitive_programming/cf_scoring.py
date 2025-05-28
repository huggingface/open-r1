import asyncio
import os
from io import BytesIO
from typing import Literal

from async_lru import alru_cache

from .piston_client import PistonClient
from .utils import batched


async def score_single_test_case(
    client: PistonClient,
    problem_data: dict,
    test_input: str,
    test_output: str,
    submission: str,
    submission_language: str = "cpp",
) -> tuple[str, str]:
    if submission_language not in ["python", "cpp"]:
        raise ValueError(f"Invalid submission language: {submission_language}")
    try:
        result = await client.send_execute(
            {
                "files": [
                    {"name": f"main.{submission_language}", "content": submission},
                    *(
                        [{"name": "checker.py", "content": problem_data["generated_checker"]}]
                        if problem_data["generated_checker"]
                        else []
                    ),
                    {"name": "input.txt", "content": test_input},
                    {"name": "correct_output.txt", "content": test_output},
                    {
                        "name": "grader_config",
                        "content": "\n".join(
                            f"{key}={value}"
                            for key, value in {
                                "TIME_LIMIT": problem_data["time_limit"],
                                "MEMORY_LIMIT": problem_data["memory_limit"],
                                "INPUT_MODE": problem_data["input_mode"],
                            }.items()
                        ),
                    },
                ],
                "run_timeout": (problem_data["time_limit"] + 10) * 1000,
                # +10 seconds hard limit. time limits are handled by the codeforces script
            },
            language="cf_python3" if submission_language == "python" else "c++17",
        )
    except Exception as e:
        print(f"Error scoring submission: {e}")
        return False

    return result


@alru_cache(maxsize=32)  # TODO make this configurable
async def get_generated_contest_tests(contest_id: str) -> list[dict]:
    import pandas as pd

    import aiofiles
    import aiofiles.os

    tests_folder = os.environ.get("CF_TESTS_FOLDER", None)
    if not tests_folder:
        raise ValueError(
            "CF_TESTS_FOLDER environment variable not set! Please download the codeforces generated tests and set CF_TESTS_FOLDER to the folder path. See https://huggingface.co/datasets/open-r1/codeforces for more information."
        )
    if not await aiofiles.os.path.exists(tests_folder):
        raise ValueError(
            f"CF_TESTS_FOLDER path '{tests_folder}' does not exist! Please download the codeforces generated tests and set CF_TESTS_FOLDER to the folder path. See https://huggingface.co/datasets/open-r1/codeforces for more information."
        )
    parquet_path = os.path.join(tests_folder, f"test_cases_{int(contest_id):04d}.parquet")
    if not await aiofiles.os.path.exists(parquet_path):
        return {}

    # Read parquet file asynchronously
    async with aiofiles.open(parquet_path, "rb") as f:
        content = await f.read()
        df = pd.read_parquet(BytesIO(content))

    # Group by problem_id and convert to dictionary of lists
    grouped_tests = df.groupby("problem_id").apply(lambda x: x[["input", "output"]].to_dict("records")).to_dict()

    return grouped_tests


async def get_generated_tests(problem_id: str) -> list[dict]:
    contest_id = problem_id.split("/")[0]
    return (await get_generated_contest_tests(contest_id)).get(problem_id, [])


async def score_submission(
    client: PistonClient,
    problem_data: dict,
    submission: str,
    test_batch_size: int = 1,
    scoring_mode: Literal["pass_fail", "partial", "weighted_sum"] = "weighted_sum",
    no_compile_reward: float = -0.1,
    no_submission_reward: float = -1.0,
    submission_language: str = "cpp",
) -> float:
    if submission_language not in ["python", "cpp"]:
        raise ValueError(f"Invalid submission language: {submission_language}")
    test_cases = problem_data["official_tests"] + (await get_generated_tests(problem_data["id"]))
    # invalid/not a coding problem
    if test_cases is None or len(test_cases) == 0:
        return None
    # no code extracted
    if not submission:
        return no_submission_reward

    passed_test_cases = 0
    # run one batch, check if any of them failed (0 score): if so stop evaluating (assuming non partial score); otherwise continue with the next batch of test cases.
    for test_batch_to_run in batched(test_cases, test_batch_size) if test_batch_size >= 1 else [test_cases]:
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
        if any(result and result["compile"]["code"] != 0 for result in results):
            return no_compile_reward

        tests_passed_results = [
            result and result["run"]["code"] == 0 and result["run"]["stdout"].strip() == "1" for result in results
        ]
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

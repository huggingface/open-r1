import asyncio
from dataclasses import asdict, dataclass, field
from typing import Union

from .piston_client import PistonClient
from .utils import batched, load_ioi_tests


@dataclass
class TestResult:
    """
    Represents the result of a single test case execution.

    Attributes:
        test_name: Name of the test case
        score: Score achieved for this test (0.0 to 1.0)
        status: Status code of the test result (e.g., 'AC', 'WA', 'TLE')
        feedback: Detailed feedback message from the judge or an error message
    """

    test_name: str
    score: float = 0.0
    status: str = "SKIPPED"
    feedback: str = None


@dataclass
class SubtaskResult:
    """
    Represents the result of a subtask containing multiple test cases.

    Attributes:
        problem: Problem identifier
        subtask: Subtask identifier
        points: Maximum points available for this subtask
        score_precision: Number of decimal places for score rounding
        test_results: List of individual test case results
    """

    problem: str = None
    subtask: str = None

    points: float = 0.0
    score_precision: int = 2

    test_results: list[TestResult] = field(default_factory=list)

    @property
    def status(self):
        """
        Determines the overall status of the subtask based on the worst status among test results.
        Status priorities are ordered from worst to best.

        Returns:
            str: The status with the highest priority (lowest value)
        """
        status_prios = {"CE": -1, "RE": 0, "WA": 1, "MLE": 2, "TLE": 3, "PA": 4, "AC": 5, "SKIPPED": 999}
        return min([x.status for x in self.test_results], key=lambda x: status_prios[x])

    @property
    def score(self):
        """
        Calculates the raw score for the subtask as the minimum score across all test results.

        Returns:
            float: The rounded minimum score
        """
        return (
            0
            if not self.test_results
            else round(min([test_result.score for test_result in self.test_results]), self.score_precision)
        )

    @property
    def weighted_score(self):
        """
        Calculates the weighted score by multiplying the raw score by the available points.

        Returns:
            float: The rounded weighted score
        """
        return (
            0
            if not self.test_results
            else round(
                min([test_result.score for test_result in self.test_results]) * self.points, self.score_precision
            )
        )

    def to_dict(self):
        """
        Converts the SubtaskResult to a dictionary representation.

        Returns:
            dict: Dictionary containing all subtask result data
        """
        return {
            "problem": self.problem,
            "subtask": self.subtask,
            "score": self.score,
            "weighted_score": self.weighted_score,
            "points": self.points,
            "score_precision": self.score_precision,
            "status": self.status,
            "test_results": [asdict(test_result) for test_result in self.test_results],
        }


def _extract_single_status(score: float, feedback: str) -> str:
    """
    Determines the status code based on the score and feedback message.

    Args:
        score: The numeric score (0.0 to 1.0)
        feedback: The feedback message from the execution

    Returns:
        str: Status code ('CE', 'MLE', 'TLE', 'WA', 'RE', 'AC', or 'PA')
    """
    if score == 0.0:
        if "Compilation error" in feedback:
            return "CE"
        elif "Memory limit exceeded" in feedback:
            return "MLE"
        elif "Time limit exceeded" in feedback:
            return "TLE"
        elif "Output isn't correct" in feedback:
            return "WA"
        else:
            return "RE"
    elif score == 1.0:
        return "AC"
    else:
        return "PA"


async def score_single_test_case(
    client: PistonClient, subtask: dict, test_name: str, test_input: str, test_output: str, submission: str
) -> TestResult:
    """
    Scores a single test case by running the submission against the provided input and output.

    Args:
        client: PistonClient instance for executing code
        subtask: Dictionary containing subtask configuration
        test_name: Name of the test case
        test_input: Input data for the test case
        test_output: Expected output for the test case
        submission: Source code of the submission

    Returns:
        TestResult: Result of the test case execution
    """
    # Run submission for this test case
    score, feedback = await run_submission(client, subtask, test_input, submission, test_output)
    score = float(score)

    return TestResult(
        test_name=test_name, score=score, status=_extract_single_status(score, feedback), feedback=feedback
    )


async def score_subtask(
    client: PistonClient,
    subtask: dict,
    submission: str,
    test_case_run_cache: Union[dict, None] = None,
    test_batch_size: int = 1,
) -> SubtaskResult:
    """
    Scores all test cases in a subtask.

    Args:
        client: PistonClient instance for executing code
        subtask: Dictionary containing subtask configuration
        test_cases: Dictionary mapping test names to (input, output) tuples
        submission: Source code of the submission
        test_case_run_cache: Optional cache of previously run test cases
        test_batch_size: evaluate these many test cases in parallel, then check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
        -1 to evaluate all test cases in parallel
    Returns:
        SubtaskResult: Result of the subtask evaluation
    """
    subtask_result = SubtaskResult(
        problem=subtask["id"],
        subtask=subtask["subtask"],
        points=subtask["score"],
        score_precision=subtask["score_precision"],
        test_results=[],
    )

    # tests that are not cached
    tests_to_run = [
        (ti, test_name)
        for ti, test_name in enumerate(subtask["test_names"])
        if test_case_run_cache is None or test_name not in test_case_run_cache
    ]

    # initialize test results with cached results or empty (SKIPPED) TestResult objects
    subtask_result.test_results = [
        test_case_run_cache[test_name]
        if test_case_run_cache is not None and test_name in test_case_run_cache
        else TestResult(test_name=test_name)
        for test_name in subtask["test_names"]
    ]

    # we skip submissions where no code was extracted
    # no need to do anything, as we have a failed cached result
    if not submission or any(
        test_result.status != "SKIPPED" and test_result.score == 0.0 for test_result in subtask_result.test_results
    ):
        return subtask_result

    if "test_cases" in subtask:
        test_cases = subtask["test_cases"]
        if isinstance(subtask["test_cases"], list):
            test_cases = {test_name: test for test_name, test in zip(subtask["test_names"], subtask["test_cases"])}
    else:
        test_cases = load_ioi_tests(subtask["year"], subtask["id"])

    # run one batch, check if any of them failed (0 score): if so stop evaluating; otherwise continue with the next batch of test cases.
    for test_batch_to_run in batched(tests_to_run, test_batch_size):
        results = await asyncio.gather(
            *[
                asyncio.create_task(
                    score_single_test_case(
                        client, subtask, test_name, test_cases[test_name][0], test_cases[test_name][1], submission
                    )
                )
                for _, test_name in test_batch_to_run
            ]
        )
        for (ti, test_name), test_result in zip(test_batch_to_run, results):
            if test_case_run_cache is not None:
                test_case_run_cache[test_name] = test_result
            subtask_result.test_results[ti] = test_result

        # Stop early if it failed
        if any(test_result.score == 0.0 for test_result in results):
            break

    return subtask_result


async def score_subtasks(
    client: PistonClient, subtasks: list[dict], submission: str, skip_mode: bool = True
) -> list[SubtaskResult]:
    """
    Scores multiple subtasks for a submission.

    Args:
        client: PistonClient instance for executing code
        subtasks: List of dictionaries containing subtask configurations
        submission: Source code of the submission
        skip_mode: If True, evaluates test by test and stops after the first failure. Otherwise, runs all tests in parallel. Should be True when evaluating a large number of submissions.

    Returns:
        list[SubtaskResult]: Results for all subtasks
    """
    # avoid rerunning tests present in multiple subtasks
    test_case_run_cache = {}

    return [await score_subtask(client, subtask, submission, test_case_run_cache, skip_mode) for subtask in subtasks]


async def run_submission(
    client: PistonClient, problem: dict, test_input: str, submission: str, test_output: str | None = None
) -> tuple[str, str]:
    """
    Executes a submission against a test case using the Piston execution environment.

    Args:
        client: PistonClient instance for executing code
        problem: Dictionary containing problem configuration
        test_input: Input data for the test case
        submission: Source code of the submission
        test_output: Optional expected output for the test case

    Returns:
        tuple[str, str]: A tuple containing (score, feedback)
    """
    data = {
        "files": [
            # the actual submission
            {"name": f"graders/{problem['id'].lower()}.cpp", "content": submission},
            # pass the input
            {"name": "input.txt", "content": test_input},
            # pass the expected output
            *([{"name": "correct_output.txt", "content": test_output}] if test_output else []),
            # grader files
            *({"name": name, "content": content} for name, content in problem["grader_files"] if content),
        ],
        "run_timeout": round(
            (problem["time_limit"] + 3) * 1000
        ),  # +3 seconds hard limit. time limits are handled by the ioi script
        "run_memory_limit": problem["memory_limit"],
    }
    return await client.execute(data)

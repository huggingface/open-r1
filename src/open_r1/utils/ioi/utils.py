from collections import defaultdict
from functools import lru_cache
from itertools import islice

from datasets import load_dataset


def add_includes(code: str, problem_id: str) -> str:
    """
    Fix common compilation errors for IOI problems.
    """
    if not code:
        return code
    # has most of the useful functions
    code_header = "#include <bits/stdc++.h>\n"
    # include the problem header
    problem_header_include = f'#include "{problem_id}.h"'
    if problem_header_include not in code:
        code_header += problem_header_include + "\n"
    # use namespace std since models forget std:: often
    if "using namespace std;" not in code and "std::" not in code:
        code_header += "\nusing namespace std;\n\n"
    return code_header + code


@lru_cache
def load_ioi_tests_for_year(year: int) -> dict[str, dict[str, tuple[str, str]]]:
    """
    Load IOI tests for a given year.
    """
    tests_dataset = load_dataset("open-r1/ioi-test-cases", name=f"{year}", split="train")
    test_cases = defaultdict(dict)
    for test_case in tests_dataset:
        test_cases[test_case["problem_id"]][test_case["test_name"]] = test_case["test_input"], test_case["test_output"]
    return test_cases


def load_ioi_tests(year: int, problem_id: str) -> dict[str, tuple[str, str]]:
    """
    Load IOI tests for a given year and problem id.
    """
    return load_ioi_tests_for_year(year)[problem_id]


def batched(iterable, n):
    "Batch data into lists of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        return iterable
    it = iter(iterable)
    while batch := list(islice(it, n)):
        yield batch

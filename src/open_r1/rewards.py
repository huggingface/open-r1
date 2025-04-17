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

import asyncio
import json
import math
import re
import textwrap
import time
from functools import partial, update_wrapper
from typing import Callable, Dict, Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from .utils.ioi import SubtaskResult, add_includes, get_piston_client_from_env, get_morph_client_from_env, score_subtask
from .utils.code_providers import get_provider


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

    Taken from the Kimi 1.5 tech report: https://arxiv.org/abs/2501.12599

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
            gold_parsed = parse(sol, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
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


def get_repetition_penalty_reward(ngram_size: int, max_penalty: float):
    """
    Computes N-gram repetition penalty as described in Appendix C.2 of https://arxiv.org/abs/2502.03373.
    Reference implementation from: https://github.com/eddycmu/demystify-long-cot/blob/release/openrlhf/openrlhf/reward/repetition.py

    Args:
    ngram_size: size of the n-grams
    max_penalty: Maximum (negative) penalty for wrong answers
    """
    if max_penalty > 0:
        raise ValueError(f"max_penalty {max_penalty} should not be positive")

    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

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
            if len(completion.split()) < ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in zipngram(completion, ngram_size):
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
        # for info on setting up MorphCloud, see MorphCloud_IOI_Implementation_Guide.md
        print('provider: morph')
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
            return SubtaskResult()  # score 0.0

    # load problem data. undo separating kwargs by column
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]

    loop = _init_event_loop()
    evals = [
        loop.create_task(
            run_catch_exceptions(score_subtask(execution_client, problem_data, code, test_batch_size=test_batch_size))
        )
        for problem_data, code in zip(problems_data, code_snippets)
    ]
    results = loop.run_until_complete(asyncio.gather(*evals))

    return [result.score for result in results]


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


def binary_code_reward(completions, num_parallel: int = 2, e2b_router_url=None, provider_type: str = None, enforce_same_language: bool = False, **kwargs) -> list[float]:
    rewards = code_reward(
        completions, 
        num_parallel=num_parallel, 
        e2b_router_url=e2b_router_url, 
        provider_type=provider_type,
        enforce_same_language=enforce_same_language,
        **kwargs
    )
    BINARY_THRESHOLD = 0.99

    output = []
    for reward in rewards:
        if reward is None:
            output.append(None)
        else:
            output.append(1.0 if reward > BINARY_THRESHOLD else 0.0)

    return output


def code_reward_morph(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    """Reward function that evaluates code using MorphCloud's direct language execution.
    
    This function executes code directly in the appropriate language kernel:
    - The test case loop is handled locally in Python
    - Each submission is run in the appropriate language on the MorphCloud sandbox
    - Output is checked locally against expected values
    
    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions
        **kwargs: Additional arguments including verification_info with test cases
        
    Returns:
        List of float rewards (one per completion)
    """
        
    # Extract code from completions
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    
    # Get the MorphProvider
    execution_provider = get_provider(
        provider_type="morph",
        num_parallel=num_parallel,
    )
    
    async def process_snippets(code_snippets, verification_info):
        """Process all code snippets in parallel."""
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(num_parallel)
        
        # Create tasks for parallel processing
        tasks = [
            process_single_snippet(code, info, semaphore, execution_provider) 
            for code, info in zip(code_snippets, verification_info)
        ]
        
        # Execute all tasks and gather results
        rewards = await asyncio.gather(*tasks)
        return rewards
    
    async def process_single_snippet(code, info, semaphore, provider):
        """Process a single code snippet against all test cases."""
        # Extract test cases and language
        test_cases = info["test_cases"]
        language = info.get("language", "python")
        
        async with semaphore:
            # Create a sandbox
            try:
                sandbox = await asyncio.to_thread(
                    provider.Sandbox.new,
                    client=provider.client,
                    ttl_seconds=30
                )
                
                sandbox_id = getattr(sandbox, 'id', None) or getattr(sandbox._instance, 'id', 'unknown')
                print(f"MorphProvider: Processing {language} code in sandbox {sandbox_id[:8]}...")
                
                # Process each test case
                passed = 0
                total = len(test_cases)
                
                for i, case in enumerate(test_cases):
                    try:
                        # Create code with input handling for this test case
                        test_code = prepare_code_with_input(code, case["input"], language)
                        
                        # Run the code in the appropriate language
                        result = await asyncio.to_thread(
                            sandbox.run_code,
                            test_code,
                            language=language,
                            timeout=10
                        )
                        
                        # Check if execution was successful and output matches
                        if result.success:
                            output = result.stdout.strip() if result.stdout else ""
                            expected = case["output"].strip()
                            
                            # Compare outputs (same logic as in the template)
                            all_correct = True
                            output_lines = output.split('\n')
                            expected_lines = expected.split('\n')
                            
                            for line1, line2 in zip(output_lines, expected_lines):
                                all_correct = all_correct and line1.strip() == line2.strip()
                                
                            if all_correct:
                                passed += 1
                                print(f"MorphProvider: Test case {i+1}/{total} passed")
                            else:
                                print(f"MorphProvider: Test case {i+1}/{total} failed - output mismatch")
                        else:
                            print(f"MorphProvider: Test case {i+1}/{total} failed - execution error: {result.error}")
                    except Exception as e:
                        print(f"MorphProvider: Error in test case {i+1}/{total}: {e}")
                
                # Calculate success rate
                reward = passed / total if total > 0 else 0.0
                print(f"MorphProvider: Final reward: {reward} ({passed}/{total} tests passed)")
                return reward
                
            except Exception as e:
                print(f"MorphProvider: Error in code execution: {e}")
                return 0.0
            finally:
                # Clean up the sandbox
                if 'sandbox' in locals():
                    try:
                        await asyncio.to_thread(sandbox.close)
                        await asyncio.to_thread(sandbox.shutdown)
                    except Exception as e:
                        print(f"MorphProvider: Error cleaning up sandbox: {e}")
    
    def prepare_code_with_input(code: str, input_str: str, language: str) -> str:
        """Prepare code with input handling based on language.
        
        Based on proven patterns from sandbox_lang_test_mock.py for more reliable input mocking.
        """
        # Ensure input_str is a string - convert if needed
        if not isinstance(input_str, str):
            print(f"Warning: Input is not a string type, converting from {type(input_str)}")
            input_str = str(input_str)
        
        if language == "python":
            # For Python, use array-based input mocking with a counter
            input_lines = input_str.strip().split('\n')
            input_lines_json = json.dumps(input_lines)
            indented_code = textwrap.indent(code.strip(), ' ' * 4)
            
            return f"""
import sys
import traceback

# Mock input function for Python
_input_values = {input_lines_json}
_input_index = 0
def input(*args):
    global _input_index
    if _input_index >= len(_input_values):
        raise EOFError("Attempted to read past end of input buffer.")
    value = _input_values[_input_index]
    _input_index += 1
    return value

# Execute user code
try:
    # --- User code starts ---
{indented_code}
    # --- User code ends ---
except EOFError as e:
     print(f"Execution stopped: {{e}}", file=sys.stderr)
except Exception as e:
    print(f"Runtime Error: {{e}}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
"""
        elif language == "javascript" or language == "js":
            # For JavaScript, use array-based input mocking with readLine function
            input_lines = input_str.strip().split('\n')
            input_lines_json = json.dumps(input_lines)
            clean_user_code = code.strip()
            indented_code = textwrap.indent(textwrap.dedent(f"""
                const _input_lines = {input_lines_json};
                let _input_index = 0;
                function readLine() {{
                    if (_input_index < _input_lines.length) {{
                        return _input_lines[_input_index++];
                    }} else {{
                        return null;
                    }}
                }}
                try {{
{textwrap.indent(clean_user_code, ' ' * 12)} // Ensure user code is indented within try
                }} catch (e) {{
                    console.error("Runtime Error:", e.message);
                    // Optionally re-throw or log stack trace if needed
                    // console.error(e.stack);
                }}
            """), ' ' * 4)
            return f"""
(function() {{
{indented_code}
}})();
"""
        elif language == "cpp" or language == "c++":
            # For C++, use stringstream-based input mocking
            input_lines = input_str.strip().split('\n')
            escaped_lines = [line.replace('\\', '\\\\').replace('"', '\\"') for line in input_lines]
            input_vector_init = ", ".join([f'"{line}"' for line in escaped_lines])
            clean_user_code = code.strip()
            
            return textwrap.dedent(f"""
                #include <iostream>
                #include <sstream>
                #include <string>
                #include <vector>
                #include <stdexcept>

                int run_user_code() {{
                    std::stringstream _mock_input_stream;
                    std::vector<std::string> _input_lines = {{{input_vector_init}}};
                    for(const auto& line : _input_lines) {{
                        _mock_input_stream << line << std::endl;
                    }}
                    std::streambuf* _orig_cin_buf = std::cin.rdbuf();
                    std::cin.rdbuf(_mock_input_stream.rdbuf());

                    int exit_code = 0;
                    try {{
                        // --- User C++ Code Execution ---
                        {clean_user_code}
                        // --- End User Code ---
                    }} catch (const std::exception& e) {{
                        std::cerr << "Runtime Error: " << e.what() << std::endl;
                        exit_code = 1;
                    }} catch (...) {{
                        std::cerr << "Unknown runtime error." << std::endl;
                        exit_code = 1;
                    }}

                    std::cin.rdbuf(_orig_cin_buf);
                    return exit_code;
                }}
                run_user_code();
            """)
        elif language == "rust":
            # For Rust, use BufReader around a Cursor for input mocking
            clean_user_code = code.strip()
            # Use raw string literal r#""# for input to handle most chars easily
            escaped_input = input_str.replace('#', "\\#") # Only need to escape # if using r#""#
            
            # Wrap the setup and user code in a main function for better scoping
            # Indent the user code appropriately
            indented_user_code = textwrap.indent(clean_user_code, ' ' * 8) # Indent inside the inner block
            
            return textwrap.dedent(f"""
                // --- Rust Input Mocking Setup ---
                use std::io::{{self, BufRead, BufReader, Read, Cursor}};

                fn main() {{
                    eprintln!("DEBUG: Rust main started."); // Debug output

                    // Input data embedded as a string literal
                    let input_data_str = r#"{escaped_input}"#;
                    let input_data_bytes = input_data_str.as_bytes();

                    // Create the in-memory reader the user code will use
                    let mut input_reader = BufReader::new(Cursor::new(input_data_bytes));
                    eprintln!("DEBUG: Mock input reader created with {{}} bytes.", input_data_bytes.len());

                    // --- User Rust Code Execution Block ---
                    // User code reads from `input_reader` defined above.
                    {{
                        eprintln!("DEBUG: Entering user code block."); // Debug output
{indented_user_code}
                        eprintln!("DEBUG: Exiting user code block."); // Debug output
                    }}
                    // --- End User Code ---

                    eprintln!("DEBUG: Rust main finished."); // Debug output
                }}

                main()
            """)
        else:
            # Default handling for unsupported languages
            print(f"Warning: Language '{language}' not explicitly supported. Using raw code.")
            return code
    
    # Run async processing and return rewards
    print(f"MorphProvider: Starting code evaluation with parallelism={num_parallel}")
    start_time = time.time()
    rewards = asyncio.run(process_snippets(code_snippets, verification_info))
    elapsed = time.time() - start_time
    print(f"MorphProvider: Evaluation completed in {elapsed:.2f}s ({len(code_snippets)/elapsed:.2f} snippets/sec)")
    
    return rewards


def code_reward(completions, num_parallel: int = 2, e2b_router_url=None, provider_type: str = "e2b", enforce_same_language: bool = False, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using a code execution provider.

    Assumes the dataset contains a `verification_info` column with test cases.
    
    Args:
        completions: List of model completions to evaluate
        num_parallel: Number of parallel code executions (default: 2)
        e2b_router_url: URL for E2B router (if using E2B provider with router mode)
        provider_type: Which code execution provider to use (default: "e2b")
        enforce_same_language: If True, verify all problems use the same language (default: False)
        **kwargs: Additional arguments passed to the verification
    """
    # Use MorphCloud-specific implementation for more accurate language support
    if provider_type == "morph":
        return code_reward_morph(completions, num_parallel=num_parallel, **kwargs)
    
    # Script template used for evaluation - all providers use this same template
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
    
    # Extract code from completions
    code_snippets = [extract_code(completion[-1]["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    
    # Create scripts by populating the template with code and test cases
    scripts = [
        evaluation_script_template.format(
            code=json.dumps(code), 
            test_cases=json.dumps(json.dumps(info["test_cases"]))
        )
        for code, info in zip(code_snippets, verification_info)
    ]

    # Get language from first problem
    language = verification_info[0]["language"]
    
    # Verify all problems use the same language if enforce_same_language is True
    if enforce_same_language:
        all_same_language = all(v["language"] == language for v in verification_info)
        if not all_same_language:
            raise ValueError("All verification_info must have the same language", verification_info)

    # Get the appropriate provider and execute the scripts
    execution_provider = get_provider(
        provider_type=provider_type,
        num_parallel=num_parallel,
        e2b_router_url=e2b_router_url,
    )
    
    return execution_provider.execute_scripts(scripts, language)


def get_code_format_reward(language: str = "python"):
    """Format reward function specifically for code responses.

    Args:
        language: Programming language supported by E2B https://e2b.dev/docs/code-interpreting/supported-languages
    """
    pattern = rf"^<think>\n.*?\n</think>\n<answer>\n.*?```{language}.*?```.*?\n</answer>$"

    def code_format_reward(completions, **kwargs):
        completion_contents = [completion[0]["content"] for completion in completions]
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completion_contents]
        return [1.0 if match else 0.0 for match in matches]

    return code_format_reward


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
                e2b_router_url=script_args.e2b_router_url,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                e2b_router_url=script_args.e2b_router_url,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(script_args, "enforce_same_language", False),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward, 
                test_batch_size=script_args.code_eval_test_batch_size, 
                provider_type=getattr(script_args, "ioi_provider", "piston")
            ), 
            ioi_code_reward
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs

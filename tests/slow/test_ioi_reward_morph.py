from datasets import load_dataset
from collections import defaultdict
import asyncio
import re
import json

# Import necessary functions from your module
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../src")))

from open_r1.rewards import ioi_code_reward as original_ioi_code_reward
from open_r1.rewards import add_includes
from open_r1.utils.ioi import SubtaskResult, score_subtask, score_subtasks

# Define a patched version of ioi_code_reward with additional debug output
def ioi_code_reward(completions, test_batch_size: int = 1, provider_type: str = "piston", 
                    evaluate_all_subtasks: bool = True, **kwargs) -> list[float]:
    """A patched version of ioi_code_reward with additional debug output.
    
    Args:
        completions: List of completions containing code to evaluate
        test_batch_size: Number of test cases to run in parallel before checking for failures
        provider_type: Type of execution provider ("piston" or "morph")
        evaluate_all_subtasks: If True, evaluates all subtasks for each problem; if False,
                              evaluates only the specified subtask
        **kwargs: Additional arguments including problem data
    
    Returns:
        list[float]: If evaluate_all_subtasks is False, returns raw scores.
                     If evaluate_all_subtasks is True, returns list of dictionaries with all subtask results.
    """
    print("\n==== DEBUG: STARTING IOI CODE REWARD (PATCHED VERSION) ====")
    print(f"Provider type: {provider_type}")
    print(f"Test batch size: {test_batch_size}")
    print(f"Evaluate all subtasks: {evaluate_all_subtasks}")
    print(f"Number of completions: {len(completions)}")
    
    # Get the appropriate client
    if provider_type == "morph":
        print("DEBUG: Using MorphCloud client")
        from open_r1.utils.ioi import get_morph_client_from_env
        execution_client = get_morph_client_from_env()
    else:
        print("DEBUG: Using Piston client")
        from open_r1.utils.ioi import get_piston_client_from_env
        execution_client = get_piston_client_from_env()
    
    # Extract code from completions
    def extract_code(completion: str, language: str = "python") -> str:
        pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
        matches = pattern.findall(completion)
        extracted_answer = matches[-1] if len(matches) >= 1 else ""
        return extracted_answer
    
    print("\n==== DEBUG: EXTRACTING CODE FROM COMPLETIONS ====")
    code_snippets = []
    for i, (completion, problem_id) in enumerate(zip(completions, kwargs["id"])):
        raw_code = extract_code(completion[-1]["content"], "cpp")
        processed_code = add_includes(raw_code, problem_id)
        code_snippets.append(processed_code)
        print(f"DEBUG: Code snippet {i+1} length: {len(processed_code)} characters")
        print(f"DEBUG: Code snippet {i+1} first 100 chars: {processed_code[:100]}...")
    
    # Load problem data
    print("\n==== DEBUG: PROCESSING PROBLEM DATA ====")
    problems_data = [dict(zip(kwargs.keys(), values)) for values in zip(*kwargs.values())]
    for i, problem in enumerate(problems_data):
        print(f"DEBUG: Problem {i+1}: id={problem['id']}, year={problem['year']}, subtask={problem['subtask']}")
        print(f"DEBUG: Test names: {problem['test_names'][:3]}{'...' if len(problem['test_names']) > 3 else ''}")
        print(f"DEBUG: Grader files count: {len(problem['grader_files'])}")
    
    # Set up async functions
    async def run_catch_exceptions(task, description=None):
        try:
            print(f"DEBUG: Starting task {description or task}")
            result = await task
            if isinstance(result, list):
                result_scores = [r.score for r in result]
                print(f"DEBUG: Task completed with subtask scores: {result_scores}")
            else:
                print(f"DEBUG: Task completed with score: {result.score}")
            return result
        except Exception as e:
            print(f"ERROR from {provider_type} worker: {e}")
            return SubtaskResult() if not evaluate_all_subtasks else []  # empty or score 0.0
    
    # Initialize event loop
    print("\n==== DEBUG: INITIALIZING EVENT LOOP ====")
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    # Create and run tasks
    print("\n==== DEBUG: CREATING AND RUNNING TASKS ====")
    evals = []
    
    if evaluate_all_subtasks:
        # First, we need to get all subtasks for each problem
        problems_lookup = {}
        if "all_subtasks" in kwargs:
            # If all_subtasks provided directly, use them
            problems_lookup = kwargs["all_subtasks"]
        else:
            # Otherwise, use the problem data to identify subtasks
            problem_ids = set([(problem['year'], problem['id']) for problem in problems_data])
            print(f"DEBUG: Looking up subtasks for {len(problem_ids)} problems")
            
            for i, (year, problem_id) in enumerate(problem_ids):
                print(f"DEBUG: Problem {i+1}: {problem_id} (year {year})")
                if (year, problem_id) not in problems_lookup:
                    problems_lookup[(year, problem_id)] = []
                
                # Add the current subtask from kwargs if not already included
                for problem in problems_data:
                    if problem['year'] == year and problem['id'] == problem_id:
                        if problem not in problems_lookup[(year, problem_id)]:
                            problems_lookup[(year, problem_id)].append(problem)
            
            print(f"DEBUG: Found subtasks for {len(problems_lookup)} problems")
        
        # Now evaluate submissions on all subtasks for their problem
        for i, (code, problem) in enumerate(zip(code_snippets, problems_data)):
            year, problem_id = problem['year'], problem['id']
            subtasks = problems_lookup.get((year, problem_id), [])
            
            if not subtasks:
                print(f"WARNING: No subtasks found for {problem_id} (year {year})")
                evals.append(SubtaskResult())
                continue
            
            print(f"DEBUG: Evaluating submission {i+1} on {len(subtasks)} subtasks for {problem_id}")
            
            # Use the score_subtasks function to evaluate all subtasks
            test_case_run_cache = {}  # Cache to avoid re-running the same tests
            task = loop.create_task(
                run_catch_exceptions(
                    score_subtasks(execution_client, subtasks, code, test_batch_size),
                    f"Submission {i+1} - {problem_id} (all subtasks)"
                )
            )
            evals.append(task)
    else:
        # Original single-subtask behavior
        for i, (problem_data, code) in enumerate(zip(problems_data, code_snippets)):
            print(f"DEBUG: Creating task {i+1} for problem {problem_data['id']} (single subtask)")
            task = loop.create_task(
                run_catch_exceptions(
                    score_subtask(execution_client, problem_data, code, test_batch_size=test_batch_size),
                    f"Submission {i+1} - {problem_data['id']} (subtask {problem_data['subtask']})"
                )
            )
            evals.append(task)
    
    print("DEBUG: Running all tasks...")
    results = loop.run_until_complete(asyncio.gather(*evals))
    
    # Process results
    print("\n==== DEBUG: PROCESSING RESULTS ====")
    if evaluate_all_subtasks:
        # For multi-subtask evaluation, return detailed results
        detailed_results = []
        for i, subtask_results in enumerate(results):
            if not subtask_results:
                print(f"DEBUG: No results for submission {i+1}")
                detailed_results.append({
                    "status": "ERROR",
                    "all_subtasks_points": 0,
                    "all_subtasks_results": []
                })
                continue
                
            # Calculate weighted score sum across all subtasks
            weighted_score_sum = sum(result.weighted_score for result in subtask_results)
            
            # Convert to dictionary format
            result_dict = {
                "status": "OK" if any(result.score > 0 for result in subtask_results) else "FAILED",
                "all_subtasks_points": weighted_score_sum,
                "all_subtasks_results": [result.to_dict() for result in subtask_results]
            }
            detailed_results.append(result_dict)
            
            print(f"DEBUG: Result {i+1}: all_subtasks_points={weighted_score_sum}")
            print(f"DEBUG: Subtask scores: {[result.score for result in subtask_results]}")
            print(f"DEBUG: Subtask statuses: {[result.status for result in subtask_results]}")
        
        print("\n==== DEBUG: IOI CODE REWARD COMPLETE (MULTI-SUBTASK MODE) ====")
        return detailed_results
    else:
        # Original single-subtask behavior
        scores = []
        for i, result in enumerate(results):
            print(f"DEBUG: Result {i+1}: score={result.score}, status={result.status}")
            if result.test_results:
                print(f"DEBUG: Test results: {[(tr.test_name, tr.score, tr.status) for tr in result.test_results]}")
            scores.append(result.score)
        
        print("\n==== DEBUG: IOI CODE REWARD COMPLETE (SINGLE-SUBTASK MODE) ====")
        return scores

def main():
    # Configuration
    dataset_to_evaluate = "open-r1/ioi-sample-solutions"
    id_column = "label"
    test_batch_size = 1
    use_morph = True
    max_samples = 3  # Just test with a few samples
    
    print(f"Loading dataset {dataset_to_evaluate}...")
    
    try:
        # Load dataset with streaming to filter through all samples
        streaming_dataset = load_dataset(dataset_to_evaluate, split="train", streaming=True)
        
        # Keep track of unique problem IDs we've seen
        unique_problems = set()
        selected_samples = []
        
        # Iterate through the dataset to find unique problems
        print(f"Searching for {max_samples} unique problems...")
        for i, sample in enumerate(streaming_dataset):
            # Create a comprehensive unique ID using all available fields
            problem_id = sample.get('problem_id', '')
            year = sample.get('year', 2024)
            day = sample.get('day', '')
            problem_name = sample.get('problem_name', '')
            subtask = sample.get('subtask', '')
            label = sample.get(id_column, '')
            
            # Create a unique identifier tuple with all available information
            unique_id = (
                str(year), 
                str(day), 
                problem_name, 
                problem_id, 
                subtask, 
                label
            )
            
            # For logging, create a simpler display key
            display_key = f"{problem_id} (year {year})"
            
            # If this is a new problem we haven't seen before, add it
            if unique_id not in unique_problems and len(selected_samples) < max_samples:
                print(f"Found unique problem {i}: {display_key}")
                print(f"  Full ID: {unique_id}")
                unique_problems.add(unique_id)
                selected_samples.append(sample)
                
            # Break once we have enough unique problems
            if len(selected_samples) >= max_samples:
                break
                
            # Show progress every 100 samples
            if i % 100 == 0 and i > 0:
                print(f"Processed {i} samples, found {len(unique_problems)} unique problems so far...")
        
        # Convert to a Dataset object
        from datasets import Dataset
        samples = Dataset.from_list(selected_samples)
        print(f"Selected {len(samples)} unique problems for testing:")
        
        # Print details of the selected samples with more readable formatting
        print("\n=== SELECTED PROBLEMS FOR TESTING ===")
        for i, sample in enumerate(samples):
            problem_id = sample.get('problem_id', '')
            year = sample.get('year', 2024)
            day = sample.get('day', '')
            problem_name = sample.get('problem_name', '')
            subtask = sample.get('subtask', '')
            label = sample.get(id_column, '')
            
            print(f"\nProblem {i+1}: {problem_id} (year {year})")
            print(f"  Day: {day}")
            print(f"  Name: {problem_name}")
            print(f"  Subtask: {subtask}")
            print(f"  Label: {label}")
            
            # Print a sample of the generation or code if available
            if 'generation' in sample and sample['generation']:
                gen_snippet = sample['generation'][:100] + "..." if len(sample['generation']) > 100 else sample['generation']
                print(f"  Generation snippet: {gen_snippet}")
            elif 'code' in sample and sample['code']:
                code_snippet = sample['code'][:100] + "..." if len(sample['code']) > 100 else sample['code']
                print(f"  Code snippet: {code_snippet}")
        
        # Process samples for evaluation
        submissions_by_problem = defaultdict(list)
        
        for sample in samples:
            # Extract basic info using same fields as for unique_id
            problem_id = sample.get('problem_id', '')
            year = sample.get('year', 2024)
            day = sample.get('day', '')
            problem_name = sample.get('problem_name', '')
            subtask = sample.get('subtask', '')
            label = sample.get(id_column, '')
            
            # Create a consistent problem key that will match our unique_id
            # For submissions, we'll use a simpler key to group properly by problem
            problem_key = (str(year), problem_id)
            display_key = f"{problem_id} (year {year})"
            
            # Process code
            if 'code' not in sample or not sample['code']:
                if 'generation' in sample and "```cpp\n" in sample['generation']:
                    code = sample['generation'].split("```cpp\n")[-1].split("```")[0]
                else:
                    code = None
            else:
                code = sample['code']
            
            if code:
                code = add_includes(code, problem_id)

            print(f"Extracted code for submission - length: {len(code) if code else 0} characters")
            
            # Store processed submission
            sample_with_code = {**sample, 'code': code}
            submissions_by_problem[problem_key].append(sample_with_code)
            
            print(f"Processed submission for problem {display_key}")
        
        # Load problem data
        print("Loading problem data...")
        problems = load_dataset("open-r1/ioi", split="train+test")
        
        problem_subtasks = defaultdict(list)
        for problem in problems:
            key = (str(problem.get('year')), problem.get('id'))
            if key in submissions_by_problem:
                problem_subtasks[key].append(problem)
        
        # Now evaluate each problem's submissions with ioi_code_reward
        for (year, problem_id), submissions in submissions_by_problem.items():
            if not problem_subtasks[(year, problem_id)]:
                print(f"No problem data found for {problem_id} (year {year})")
                continue
            
            # Get all subtasks for this problem
            subtasks = problem_subtasks[(year, problem_id)]  # Get all subtasks
            print(f"Found {len(subtasks)} subtasks for {problem_id} (year {year})")
            
            print(f"Evaluating {len(submissions)} submissions for {problem_id} (year {year})...")
            
            # Format as expected by ioi_code_reward
            print("\n==== FORMATTING SUBMISSIONS FOR EVALUATION ====")
            print(f"Total submissions to format: {len(submissions)}")
            completions = []
            for i, submission in enumerate(submissions):
                print(f"\n--- Processing submission {i+1}/{len(submissions)} ---")
                code = submission.get('code', '')
                if not code:
                    print("WARNING: No code found in submission")
                    code = ""
                else:
                    print(f"Code found with length: {len(code)} characters")
                
                # Wrap the code in markdown code blocks so extract_code can find it
                markdown_content = f"```cpp\n{code}\n```"
                
                # Create a completion with the markdown formatted code
                completion = [
                    {"role": "assistant", "content": markdown_content}
                ]
                completions.append(completion)
                print(f"Submission formatted with code blocks: ```cpp\n{code[:50]}...\n```")
                
                # Print some additional detail about the submission
                print(f"Submission keys: {list(submission.keys())}")
                
                # Print sample of "generation" field if it exists
                if 'generation' in submission:
                    gen_sample = submission['generation'][:100] + "..." if len(submission['generation']) > 100 else submission['generation']
                    print(f"Generation sample: {gen_sample}")

            print("\n==== PREPARING KWARGS FOR IOI_CODE_REWARD ====")
            
            # Use the first subtask for template values to maintain backward compatibility
            # but will evaluate all subtasks
            reference_subtask = subtasks[0]
            
            # Prepare kwargs for ioi_code_reward
            kwargs = {
                "id": [problem_id] * len(submissions),
                "year": [int(year)] * len(submissions),
                "subtask": [reference_subtask.get("subtask", "examples")] * len(submissions),
                "score": [reference_subtask.get("score", 100)] * len(submissions),
                "score_precision": [reference_subtask.get("score_precision", 2)] * len(submissions),
                "test_names": [reference_subtask.get("test_names", [])] * len(submissions),
                "grader_files": [reference_subtask.get("grader_files", [])] * len(submissions),
                "time_limit": [reference_subtask.get("time_limit", 1.0)] * len(submissions),
                "memory_limit": [reference_subtask.get("memory_limit", 512)] * len(submissions),
                
                # Add all subtasks for evaluation
                "all_subtasks": {(int(year), problem_id): subtasks}
            }
            
            # Print detailed information about the problem configuration
            print(f"Problem ID: {problem_id}")
            print(f"Year: {year}")
            print(f"Number of subtasks: {len(subtasks)}")
            
            # Print details for each subtask
            for i, st in enumerate(subtasks):
                print(f"\nSubtask {i+1}: {st.get('subtask', 'examples')}")
                print(f"  Score: {st.get('score', 100)}")
                print(f"  Score precision: {st.get('score_precision', 2)}")
                print(f"  Test names count: {len(st.get('test_names', []))}")
                print(f"  Grader files count: {len(st.get('grader_files', []))}")
                print(f"  Time limit: {st.get('time_limit', 1.0)}s")
                print(f"  Memory limit: {st.get('memory_limit', 512)}MB")
            
            # Call ioi_code_reward with multi-subtask evaluation
            print('\n==== CALLING IOI CODE REWARD WITH MULTI-SUBTASK EVALUATION ====')
            results = ioi_code_reward(
                completions=completions,
                test_batch_size=test_batch_size,
                provider_type="morph" if use_morph else "piston",
                evaluate_all_subtasks=True,
                **kwargs
            )
            
            # Print detailed results
            print(f"\n==== RESULTS FOR {problem_id} (YEAR {year}) ====")
            for i, result in enumerate(results):
                print(f"\nSubmission {i+1}:")
                print(f"  All subtasks points: {result.get('all_subtasks_points', 0)}")
                print(f"  Status: {result.get('status', 'ERROR')}")
                
                all_subtasks_results = result.get('all_subtasks_results', [])
                if all_subtasks_results:
                    print(f"  Subtask results ({len(all_subtasks_results)}):")
                    for j, subtask_result in enumerate(all_subtasks_results):
                        print(f"    Subtask {j+1} ({subtask_result.get('subtask', 'unknown')}): "
                              f"Score {subtask_result.get('score', 0)}, "
                              f"Weighted {subtask_result.get('weighted_score', 0)}, "
                              f"Status {subtask_result.get('status', 'unknown')}")
                        
                        # Print test results for this subtask
                        test_results = subtask_result.get('test_results', [])
                        if test_results:
                            print(f"      Test results ({len(test_results)}):")
                            for k, test_result in enumerate(test_results[:3]):  # Show only first 3 for brevity
                                print(f"        Test {k+1} ({test_result.get('test_name', 'unknown')}): "
                                      f"Score {test_result.get('score', 0)}, "
                                      f"Status {test_result.get('status', 'unknown')}")
                            if len(test_results) > 3:
                                print(f"        ... and {len(test_results) - 3} more tests")
    except Exception as e:
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  # No asyncio.run() here since ioi_code_reward manages its own event loop

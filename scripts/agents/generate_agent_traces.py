import argparse
import hashlib
import inspect
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Set, Any, List
from pathlib import Path
import traceback

from datasets import load_dataset
from tqdm import tqdm
import requests
import requests.adapters

from transformers import AutoTokenizer

from smolagents import CodeAgent, Tool, HfApiModel
from open_r1.rewards import run_tests

from dotenv import load_dotenv

load_dotenv(override=True)

assert os.getenv("HF_TOKEN") is not None

# from huggingface_hub import login

# print("LOGIN:\n", login(token=os.getenv("HF_TOKEN")))

file_lock = Lock()

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")

print("Launching generation")

def test_cases_on_function(function: Any, test_cases: List[dict]) -> str:
    source_code = f"```python\n{function.__source__}\n{function.__name__}()```"
    test_completions = [[{"content": source_code}]]
    test_kwargs = {"verification_info": [{"language": "python", "test_cases": test_cases}]}
    run_tests(test_completions, e2b_router_url="0.0.0.0:8000", test_mode=True, **test_kwargs)


class ModifiedFinalAnswerTool(Tool):
    name = "final_answer"
    description = "Tests a function: if correct, returns it as the final answer; Else returns the test case that errored for solving."
    inputs = {'answer_function': {'type': 'any', 'description': 'The final function that solves the problem'}}
    output_type = "string"

    def __init__(self, test_cases):
        self.is_initialized = False
        self.test_cases = test_cases
        super().__init__()

    def forward(self, answer_function: Any) -> str:
        test_cases_on_function(answer_function, self.test_cases)
        return answer_function.__source__


class ChatMessage:
    def __init__(self, content):
        self.content = content

def generate_completion_from_messages(session, messages, args, stop_sequences) -> str:
    retry_budget = 10
    while retry_budget > 0:
        try:
            formatted_chat = tokenizer.apply_chat_template(messages, tokenize=False)
            print("Input token count:", len(tokenizer.encode(formatted_chat)))
            # Add a small random delay to prevent overwhelming the API
            time.sleep(random.uniform(0.0, 0.1))
            response = session.post(
                f"http://{args.api_addr}/v1/chat/completions",
                json={
                    "model": "default",
                    "messages": messages,
                    "max_tokens": args.max_tokens,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "stop": stop_sequences,
                },
                headers={"Authorization": "Bearer EMPTY"},
                timeout=2*60*60
            )

            # Check status code and log error content if needed
            if response.status_code >= 400:
                print(f"HTTP Error {response.status_code}: {response.reason}")
                print(f"Response content: {response.text}")
                traceback.print_exc()
                retry_budget -= 1
                time.sleep(20)
                continue

            # Parse JSON response
            try:
                output = response.json()["choices"][0]["message"]["content"]
                return output
            except ValueError as e:
                print(f"JSON parsing error: {e}")
                print(f"Response content: {response.text}")
                traceback.print_exc()
                retry_budget -= 1
                time.sleep(20)
                continue

        except requests.exceptions.RequestException as e:
            print(f"API request error (will retry): {e}")
            traceback.print_exc()
            retry_budget -= 1
            time.sleep(20)

    raise Exception("Failed to get a valid response after multiple retries")

def get_agent_run(session, task, test_cases, args):
    # def model(messages, stop_sequences = None):
    #     cleaned_messages = get_clean_message_list(messages, {"system": "user", "tool-call": "assistant", "tool-response": "user"}, flatten_messages_as_text=True)
    #     result = generate_completion_from_messages(session, cleaned_messages, args, stop_sequences)
    #     return ChatMessage(content=result)

    model = HfApiModel("Qwen/Qwen2.5-Coder-32B-Instruct", provider="fireworks-ai", token=os.getenv("HF_TOKEN"))

    agent = CodeAgent(
        model=model,
        tools=[ModifiedFinalAnswerTool(test_cases)],
        additional_authorized_imports=["numpy", "math"],
        max_steps=10,
        verbosity_level=2
    )

    try:
        output = agent.run(task, additional_args={"test_cases": test_cases})
        return output, agent.write_memory_to_messages()
    except Exception as e:
        print(f"Error when generating agentic trace: {e}")
        return None

def process_example(example, session, args, output_file, pbar=None):
    prompt = f"""Here is a task to solve using a function:
    {example[args.prompt_column]}

    Now write a function that solves the problem, then you can at once test and return it by using the tool final_answer(your_function).
    - The function should take the inputs described in the task above: use the input() function to get them.
    - As such your function will not take any arguments. IT SHOULD BE ONE SINGLE MONOLITHIC FUNCTION, no helper functions, all imports and variable defs should be inside the function.
    - And your function should give its output to stdout via print(). Returning anything is useless.
    - ALSO, DO NOT TRY TO CRAFT CUSTOM TEST FUNCTIONS or do not run your function: just test it using final_answer.
    - If you get this error: 'Forbidden function evaluation: 'input' is not among the explicitly allowed tools', it just means that you've tried to run your function: don't do that, just return it using final_answer
    """
    try:
        agent_outputs, agent_memories = [], []
        for _ in range(args.num_generations):
            agent_output, agent_memory = get_agent_run(session, prompt, example["test_cases"], args)
            agent_outputs.append(agent_output)
            agent_memories.append(agent_memory)

        if any(agent_output is None for agent_output in agent_outputs):
            print("Error processing example")
            if pbar:
                pbar.update(1)
            return None

        finish_reasons = []
        api_metadata = []

        for agent_run in agent_output:
            finish_reasons.append(None)
            api_metadata.append(None)

        # Convert agent_run to a serializable format
        serializable_generations = []
        for generation in agent_memories:
            if generation is not None:
                # Convert to a simple list of dictionaries if it's not already
                if isinstance(generation, list):
                    serializable_generations.append([
                        {k: v for k, v in msg.items() if isinstance(v, (str, int, float, bool, type(None), list, dict))}
                        for msg in generation if isinstance(msg, dict)
                    ])
                else:
                    # Handle other formats or provide a placeholder
                    serializable_generations.append(str(generation))
            else:
                serializable_generations.append(None)

        # Combine original dataset fields with generations
        result = {
            **example,  # Preserve all original dataset fields
            "generations": serializable_generations,
            "final_outputs": agent_outputs,
            # "finish_reasons": finish_reasons,
            # "api_metadata": api_metadata,
        }

        # Write to file with lock
        with file_lock:
            with open(output_file, mode="a") as f:
                try:
                    f.write(json.dumps(result) + "\n")
                    f.flush()
                except TypeError as e:
                    print(f"JSON serialization error: {e}")
                    # Fallback: store with minimal information
                    fallback_result = {
                        **{k: v for k, v in example.items() if isinstance(v, (str, int, float, bool, type(None), list, dict))},
                        "error": f"Failed to serialize full result: {e}"
                    }
                    f.write(json.dumps(fallback_result) + "\n")
                    f.flush()

        if pbar:
            pbar.update(1)

        return result
    except Exception as e:
        print(f"Error processing example: {e}")
        if pbar:
            pbar.update(1)
        return None

def load_processed_uuids(output_file, uuid_column):
    processed_uuids = set()
    if os.path.exists(output_file):
        with open(output_file, mode="r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    processed_uuids.add(hashlib.md5(str(data[uuid_column]).encode()).hexdigest())
                except json.JSONDecodeError:
                    continue
    return processed_uuids

def process_example_wrapper(args_tuple):
    example, session, args, output_file, pbar = args_tuple
    return process_example(example, session, args, output_file, pbar)

def main():
    test_function = ModifiedFinalAnswerTool([{"input": "1 2 3", "output": 5}])

    def add_numbers():
        numbers = input()
        print(sum([int(number) for number in numbers.split()]))

    from textwrap import dedent

    add_numbers.__source__ = dedent(inspect.getsource(add_numbers))
    # print(test_function(
    #     add_numbers,
    # ))
    # quit()

    parser = argparse.ArgumentParser()
    # parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--prompt-column", type=str, required=True)
    parser.add_argument("--uuid-column", type=str, required=True)
    parser.add_argument("--api-addr", type=str, default="localhost:39876")
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-tokens", type=int, default=8096)
    parser.add_argument("--max-concurrent", type=int, default=1000)
    args = parser.parse_args()

    dataset = load_dataset(
        "open-r1/codeforces-test-cases",
        split="train",
        token=os.getenv("HF_TOKEN")
    ).shuffle()
    dataset = dataset.filter(lambda x: x["full_test_set"])
    
    processed_uuids = load_processed_uuids(args.output_file, args.uuid_column)
    if processed_uuids:
        print(f"Found {len(processed_uuids)} already processed examples, resuming from there...")

    # Ensure the output directory exists
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create the file if it doesn't exist
    if not output_path.exists():
        with open(args.output_file, mode="w") as f:
            f.write("")

    # Create a session that will be shared among threads
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=args.max_concurrent,
        pool_maxsize=args.max_concurrent,
        max_retries=3
    )
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    
    # Filter out already processed examples
    examples_to_process = []
    for example in dataset:
        uuid = hashlib.md5(str(example[args.uuid_column]).encode()).hexdigest()
        if uuid not in processed_uuids:
            examples_to_process.append(example)
    
    print(f"Processing {len(examples_to_process)} examples with {args.max_concurrent} workers")
    
    pbar = tqdm(
        total=len(examples_to_process),
        desc="Generating responses",
        unit="row",
        mininterval=2,
        smoothing=0.0001,
    )
    
    # Prepare arguments for each example
    example_args = [(example, session, args, args.output_file, pbar) for example in examples_to_process]
    
    # Use ThreadPoolExecutor to process examples concurrently
    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        # Submit all tasks
        futures = [executor.submit(process_example_wrapper, arg) for arg in example_args]
        
        # Wait for all futures to complete
        for future in futures:
            future.result()  # This ensures exceptions are raised
    
    pbar.close()
    print("All examples processed!")

if __name__ == "__main__":
    main()
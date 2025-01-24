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

import re
from dataclasses import dataclass, field

from datasets import load_dataset

from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )


def extract_boxed_content(text):
    start = text.find("boxed{")  # Find the starting index of "\boxed{"
    if start == -1:
        return ""  # No match found

    # Start reading from the first '{' after "boxed{"
    start += len("boxed{")
    brace_count = 1
    content = []

    for i in range(start, len(text)):
        char = text[i]
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1

        # Add the character to the content
        if brace_count > 0:
            content.append(char)
        else:
            # We've matched all opening braces
            break

    # If the braces didn't balance, it's malformed
    if brace_count != 0:
        return ""

    return "".join(content)


def accuracy_reward(completions, ground_truth, **kwargs):
    """Reward function that checks if the completion is the same as the ground truth."""
    # Regular expression to capture content inside \boxed{}
    contents = [completion[0]["content"] for completion in completions]
    answers = [extract_boxed_content(content) for content in contents]
    # Reward 1 if the content is the same as the ground truth, 0 otherwise
    return [1.0 if answer == gt else 0.0 for answer, gt in zip(answers, ground_truth)]


def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward_func,
}


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    # Format into conversation
    def make_conversation(example):
        ground_truth = extract_boxed_content(example["solution"])
        return {
            "prompt": [{"role": "user", "content": example["problem"]}],
            "ground_truth": ground_truth,
        }

    dataset = dataset.map(make_conversation)
    dataset = dataset.remove_columns("messages")

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

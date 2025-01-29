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

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import AutoTokenizer

from math_equivalence import evaluate_answer
from src.chat_templates import LLAMA_3_TEMPLATE
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
        default_factory=lambda: ["final", "info_gain"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )

    alpha: Optional[float] = field(
        default=None,
        metadata={"help": "Alpha parameter for info_gain_reward. Used only if 'info_gain_reward' is selected."},
    )

    final_reward_weight: Optional[str] = field(default="no", metadata={"help": "Weight for final reward"})

    dataset_start: Optional[int] = field(
        default=None,
    )

    dataset_end: Optional[int] = field(
        default=None,
    )


def final_weighted_reward(completions, current_reward, ground_truth, **kwargs):
    weight = kwargs.get("weight")
    if weight == "zero":
        rewards = [0 for _ in range(len(completions))]
        return rewards
    rewards = [float(evaluate_answer(gt, c)) for c, gt in zip(completions, ground_truth)]
    if weight == "no" or sum(rewards) == 0:
        print(f"completions: {completions}")
        print(f"ground_truth: {ground_truth}")
        print(f"rewards: {rewards}")
        return rewards
    tokenizer = kwargs.get("tokenizer")
    num_tokens = [len(tokenizer.encode(completion)) for completion in completions]
    if weight == "tts":
        total_tts = 0
        for i in range(len(rewards)):
            if rewards[i] == 1:
                total_tts += num_tokens[i]
        avg_tts = 1.0 * total_tts / sum(rewards)
        print(f"avg_tts: {avg_tts}")
        rewards = [avg_tts * rewards[i] / num_tokens[i] for i in range(len(num_tokens))]
    elif weight == "ttf":
        total_ttf = sum(num_tokens)
        avg_ttf = 1.0 * total_ttf / len(rewards)
        rewards = [avg_ttf * rewards[i] / num_tokens[i] for i in range(len(num_tokens))]
        print(f"avg_ttf: {avg_ttf}")
    print(f"num_tokens: {num_tokens}")
    print(f"ground_truth: {ground_truth}")
    print(f"rewards: {rewards}")
    return rewards


def make_final_weighted_reward(tokenizer, weight):
    """Factory function to create a reward function with a specific alpha."""

    def reward_wrapper(completions, current_reward, ground_truth, **kwargs):
        return final_weighted_reward(
            completions, current_reward, ground_truth, tokenizer=tokenizer, weight=weight, **kwargs
        )

    return reward_wrapper


def info_gain_reward(completions, current_reward, ground_truth, **kwargs):
    """Reward function that adjusts rewards based on information gain."""
    print(f"alpha = {kwargs.get('alpha')}")
    final_rewards = [float(evaluate_answer(gt, c)) for c, gt in zip(completions, ground_truth)]
    adjusted_rewards = [kwargs.get("alpha", 0.1) * (f - c) for c, f in zip(current_reward, final_rewards)]

    print(f"current_reward: {current_reward}")
    print(f"final_rewards: {final_rewards}")
    print(f"adjusted_reward: {adjusted_rewards}")
    print(f"completions: {completions}")
    print(f"ground_truth: {ground_truth}")

    return adjusted_rewards


def make_info_gain_reward(alpha):
    """Factory function to create a reward function with a specific alpha."""

    def reward_wrapper(completions, current_reward, ground_truth, **kwargs):
        return info_gain_reward(completions, current_reward, ground_truth, alpha=alpha, **kwargs)

    return reward_wrapper


reward_funcs_registry = {"final": final_weighted_reward, "info_gain": info_gain_reward}


def main(script_args, training_args, model_args):
    print(script_args.reward_funcs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA_3_TEMPLATE
        tokenizer.eos_token = "<|eot_id|>"

    tokenizer.pad_token = "<|end_of_text|>"

    reward_funcs = []
    for func in script_args.reward_funcs:
        if func == "info_gain" and script_args.alpha is not None:
            print(f"Register <info_gain> reward with alpha = {script_args.alpha}")
            reward_func = make_info_gain_reward(script_args.alpha)
        elif func == "final" and script_args.final_reward_weight is not None:
            print(f"Register <final> reward with weight = {script_args.final_reward_weight}")
            reward_func = make_final_weighted_reward(tokenizer, script_args.final_reward_weight)
        else:
            print("Reward specification is wrong")
            exit()
        reward_funcs.append(reward_func)

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)
    if script_args.dataset_start is not None and script_args.dataset_end is not None:
        dataset["train"] = dataset["train"].select(range(script_args.dataset_start, script_args.dataset_end))

    def process_dataset(example):
        messages = example["messages"]
        messages[-1]["content"] += "\nWait, this seems off. Let's try something else.\nStep"
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, return_tensors="pt", continue_final_message=True, add_generation_prompt=False
        )
        # print(text)
        return {"prompt": text, "ground_truth": example["ground_truth"], "current_reward": example["current_reward"]}

    processed_dataset = dataset.map(process_dataset, remove_columns=["messages"])

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=processed_dataset[script_args.dataset_train_split],
        eval_dataset=(
            processed_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None
        ),
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

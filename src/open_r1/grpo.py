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

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from math_equivalence import evaluate_answer
from src.chat_templates import LLAMA_3_TEMPLATE
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config

def reward_func(completions, ground_truth, **kwargs):
    """Reward function that gives higher scores to longer completions."""
    print(ground_truth)
    print(completions)
    rewards = [float(evaluate_answer(gt, c)) for c, gt in zip(completions, ground_truth)]
    print(rewards)
    return rewards


def extract_boxed_value(text):
    import re

    pattern = r"\\boxed{([^}]*)}(?![^{]*})"
    matches = re.findall(pattern, text)
    return matches[-1] if matches else None

def main(script_args, training_args, model_args):
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code,
        revision=model_args.model_revision,
    )

    if tokenizer.chat_template is None:
        tokenizer.chat_template = LLAMA_3_TEMPLATE
        tokenizer.eos_token = "<|eot_id|>"

    tokenizer.pad_token = "<|end_of_text|>"

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name)

    def process_dataset(example):
        messages = example["messages"]
        last_message = messages[-1]["content"]
        last_message = last_message.split("\nWait, this seems off. Let's try something else.\nStep")[0] + "\nWait, this seems off. Let's try something else.\nStep"
            messages[-1]["content"] = last_message
            
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, return_tensors="pt", continue_final_message=True, add_generation_prompt=False
        )
        # print(text)
        return {"prompt": text, "ground_truth": example["ground_truth"]}

    processed_dataset = dataset.map(process_dataset, remove_columns=["messages"])

    # Initialize the GRPO trainer
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_func,
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
    parser = TrlParser((ScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

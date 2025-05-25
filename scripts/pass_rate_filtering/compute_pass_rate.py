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

# example usage python scripts/filter_dataset.py --config recipes/dataset_filtering/config_demo.yaml

import logging
from dataclasses import dataclass
from git import Optional
import torch
import sys

import datasets
import transformers
from datasets import load_dataset
from transformers import set_seed

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_tokenizer
from trl import ModelConfig, TrlParser
from trl.data_utils import apply_chat_template
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

@dataclass
class PassRateScriptArguments(GRPOScriptArguments):
    # we can be lazy and just use the same script args as GRPO
    output_dataset_name: Optional[str] = None
    pass_rate_min: float = 0.1
    pass_rate_max: float = 0.9
    dataset_start_index: Optional[int] = None
    dataset_end_index: Optional[int] = None
    dataset_split: str = "train"


def main(script_args, training_args, model_args):
    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Load the dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_split)
    if script_args.dataset_start_index is not None and script_args.dataset_end_index is not None:
        dataset = dataset.select(range(script_args.dataset_start_index, script_args.dataset_end_index))

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Format into conversation
    def make_conversation(example, prompt_column: str = script_args.dataset_prompt_column):
        example["prompt_backup"] = example[prompt_column]
        
        prompt = []

        if training_args.system_prompt is not None:
            prompt.append({"role": "system", "content": training_args.system_prompt})

        if prompt_column not in example:
            raise ValueError(f"Dataset Question Field Error: {prompt_column} is not supported.")

        prompt.append({"role": "user", "content": example[prompt_column]})
        return {"prompt": prompt}

    dataset = dataset.map(make_conversation)
    tokenizer = get_tokenizer(model_args, training_args)
    
    if "messages" in dataset.column_names:
        dataset = dataset.remove_columns("messages")
    
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})
    llm = LLM(
        model=model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
    )

    sampling_params=SamplingParams(
        temperature=training_args.temperature,
        top_p=training_args.top_p,
        top_k=training_args.top_k,
        n=training_args.num_generations,
        max_tokens=training_args.max_completion_length,
    )
    
    def batch_score(examples):
        prompts = examples["prompt"]
        
        outputs = llm.generate(
            prompts,
            sampling_params=sampling_params,
            use_tqdm=False,
        )
        repeated_prompts = []
        reward_completions = []
        grouped_completions = []
        for output in outputs:
            prompt = output.prompt
            group = []
            for completion in output.outputs:
                text = completion.text
                group.append(text)
                message = [{"role": "assistant", "content": text}]
                repeated_prompts.append(prompt)
                reward_completions.append(message)
            grouped_completions.append(group)
        
        def repeat_each_element_k_times(list_to_repeat: list, k: int) -> list:
            return [element for item in list_to_repeat for element in [item] * k]
        
        rewards_per_func = torch.zeros(len(repeated_prompts), len(reward_funcs))
        for i, reward_func in enumerate(reward_funcs):
            keys = [key for key in examples.data.keys() if key not in ["prompt", "completion"]]
            reward_kwargs = {key: repeat_each_element_k_times(examples[key], training_args.num_generations) for key in keys}
            output_reward_func = reward_func(prompts=repeated_prompts, completions=reward_completions, **reward_kwargs)
            # Convert None values to NaN
            output_reward_func = [reward if reward is not None else torch.nan for reward in output_reward_func]

            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32)
            
        reshaped_rewards = rewards_per_func.view(-1, training_args.num_generations)
        
        examples["pass_rate_generations"] = grouped_completions
        examples["pass_rate_rewards"] = reshaped_rewards.tolist()

            
        return examples
    
    dataset = dataset.map(batch_score, batched=True, batch_size=64)
    
    # we need to restore the prompt for the final dataset
    def restore_prompt(example):
        example["prompt"] = example["prompt_backup"]
        return example
    
    dataset = dataset.map(restore_prompt)
    dataset = dataset.remove_columns("prompt_backup")
    
    if script_args.output_dataset_name is not None:
        output_dataset_name = script_args.output_dataset_name
    else:
        model_name = model_args.model_name_or_path
        if "/" in model_name:
            model_name = model_name.split("/")[-1]
        model_revision = model_args.model_revision
    
        output_dataset_name = f"{script_args.dataset_name}-{model_name}-{model_revision}-gen"
    
    config_name="default"
    filtered_config_name = f"filt-{script_args.pass_rate_min}-{script_args.pass_rate_max}"
    
    if script_args.dataset_start_index is not None and script_args.dataset_end_index is not None:
        config_name = f"gen-{script_args.dataset_start_index}-{script_args.dataset_end_index}"
        filtered_config_name = f"{filtered_config_name}-{script_args.dataset_start_index}-{script_args.dataset_end_index}"
        
    dataset.push_to_hub(output_dataset_name, config_name=config_name, revision="gen")
    
    def filter_func(example):
        rewards = example["pass_rate_rewards"]
        # get the mean of the rewards that are not None
        mean_reward = torch.nanmean(torch.tensor(rewards, dtype=torch.float32))
        
        return script_args.pass_rate_min < mean_reward < script_args.pass_rate_max
    
    logger.info(f"Filtering dataset with low reward threshold {script_args.pass_rate_min} and high reward threshold {script_args.pass_rate_max}")
    logger.info(f"Dataset size before filtering: {dataset}")
    dataset = dataset.filter(filter_func)
    logger.info(f"Dataset size after filtering: {dataset}")
    dataset.push_to_hub(output_dataset_name, config_name=filtered_config_name, revision="pass_rate")
    
    

if __name__ == "__main__":
    parser = TrlParser((PassRateScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

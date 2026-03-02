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

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed
from transformers.trainer_utils import get_last_checkpoint

from open_r1.configs import GRPOConfig, GRPOScriptArguments
from open_r1.grouped_sol_grpo_trainer import GroupedSolGRPOTrainer
from open_r1.rewards import get_reward_funcs
from open_r1.utils import get_dataset, get_model, get_tokenizer
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from trl import ModelConfig, TrlParser, get_peft_config


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """
Below is a question and it's corresponding code answer. Please write test cases to check the correctness of the code answer. You need to use the unittest library in Python and create a test class for testing.
"""
USER_PROMPT = """
### question
{question}
### code solution
{code_solution}
Please add detailed comments to the test cases you write. You do not need to test the function's ability to throw exceptions.
"""


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

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Script parameters {script_args}")
    logger.info(f"Training parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    # Load the dataset
    dataset = get_dataset(script_args)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, training_args)

    ##############
    # Load model #
    ##############
    logger.info("*** Loading model ***")
    model = get_model(model_args, training_args)

    # Get reward functions from the registry
    reward_funcs = get_reward_funcs(script_args)

    # Group all solutions per question into one row
    def make_conversation(example):
        questions = []
        all_solutions = []

        for question, correct_sols, wrong_sols in zip(
            example['question'], example['correct_solutions'], example['wrong_solutions']
        ):
            questions.append(question)
            sols = []
            for sol in correct_sols:
                sols.append({"solve_func": sol["solve_func"], "is_correct": True})
            for sol in wrong_sols:
                sols.append({"solve_func": sol["solve_func"], "is_correct": False})
            all_solutions.append(sols)

        return {"prompt_question": questions, "solutions": all_solutions}

    # Remove original columns to avoid length mismatch (ArrowInvalid)
    # as make_conversation doubles the number of rows (correct + wrong solutions)
    column_names = list(dataset.values())[0].column_names
    dataset = dataset.map(make_conversation, batched=True, remove_columns=column_names)

    # Remove individual solutions whose prompt exceeds max_prompt_length,
    # and filter out examples only when all solutions are removed.
    if training_args.max_prompt_length is not None:
        def _prompt_length(question, solve_func):
            prompt = []
            if training_args.system_prompt is not None:
                prompt.append({"role": "system", "content": SYSTEM_PROMPT})
            prompt.append({
                "role": "user",
                "content": USER_PROMPT.format(question=question, code_solution=solve_func),
            })
            tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            return len(tokens)

        def trim_solutions(example):
            question = example["prompt_question"]
            kept = [
                sol for sol in example["solutions"]
                if _prompt_length(question, sol["solve_func"]) <= training_args.max_prompt_length
            ]
            example["solutions"] = kept
            return example

        dataset = dataset.map(trim_solutions)
        dataset = dataset.filter(lambda example: len(example["solutions"]) > 0)

    for split in dataset:
        if "messages" in dataset[split].column_names:
            dataset[split] = dataset[split].remove_columns("messages")

    #############################
    # Initialize the GRPO trainer
    #############################
    m = script_args.num_sampled_solutions
    n = script_args.num_completions_per_solution
    if m * n != training_args.num_generations:
        raise ValueError(
            f"num_sampled_solutions ({m}) * num_completions_per_solution ({n}) = {m * n} "
            f"must equal num_generations ({training_args.num_generations})"
        )

    trainer = GroupedSolGRPOTrainer(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None),
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
        processing_class=tokenizer,
        num_sampled_solutions=m,
        num_completions_per_solution=n,
        system_prompt_template=SYSTEM_PROMPT if training_args.system_prompt is not None else None,
        user_prompt_template=USER_PROMPT,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(dataset[script_args.dataset_train_split])
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    # Align the model's generation config with the tokenizer's eos token
    # to avoid unbounded generation in the transformers `pipeline()` function
    trainer.model.generation_config.eos_token_id = tokenizer.eos_token_id
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": ["open-r1"],
    }
    if trainer.accelerator.is_main_process:
        trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

    ##########
    # Evaluate
    ##########
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset[script_args.dataset_test_split])
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    #############
    # push to hub
    #############
    if training_args.push_to_hub:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

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

"""
Supervised fine-tuning script for decoder language models and vision-language models.

Usage:

# One 1 node of 8 x H100s
accelerate launch --config_file=recipes/accelerate_configs/zero3.yaml src/open_r1/sft.py \
    --model_name_or_path Qwen/Qwen2.5-1.5B-Instruct \
    --dataset_name smolagents/gaia-traces \
    --num_train_epochs 1 \
    --dataset_config all \
    --eos_token '<|im_end|>' \
    --learning_rate 4.0e-5 \
    --max_seq_length 32768 \
    --per_device_train_batch_size 2 \
    --gradient_checkpointing \
    --bf16 \
    --use_liger_kernel \
    --output_dir data/OpenR1-Distill-7B
"""

import logging
import os
import sys

import datasets
import transformers
from transformers import set_seed, AutoModelForVision2Seq, AutoProcessor, LlavaForConditionalGeneration
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from open_r1.configs import ScriptArguments, SFTConfig
from open_r1.utils import get_dataset, get_model, get_tokenizer, get_processor
from open_r1.utils.callbacks import get_callbacks
from open_r1.utils.wandb_logging import init_wandb_training
from transformers import Qwen2VLProcessor

logger = logging.getLogger(__name__)

from dotenv import load_dotenv
load_dotenv()

def create_vlm_collate_fn(processor):
    """Create a data collator for VLM training that handles images and text."""
    from qwen_vl_utils import process_vision_info

    def collate_fn(examples):
        # Convert dataset format to Qwen2.5-VL message format
        batch_messages = []
        
        for example in examples:
            example_texts = example["texts"]
            example_images = example["images"]
            
            # Convert to Qwen2.5-VL structured message format
            messages = []
            for i, msg in enumerate(example_texts):
                if msg["role"] == "user" and i == 0 and example_images:
                    # First user message - add images
                    content = []
                    # Add images first
                    for img in example_images:
                        content.append({"type": "image", "image": img})
                    # Then add text
                    content.append({"type": "text", "text": msg["content"]})
                    messages.append({"role": "user", "content": content})
                else:
                    # Regular text message
                    messages.append({"role": msg["role"], "content": msg["content"]})
            
            batch_messages.append(messages)

        # Process each example
        texts = []
        all_image_inputs = []
        all_video_inputs = []
        
        for messages in batch_messages:
            # Apply chat template
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            texts.append(text)
            
            # Extract vision info
            image_inputs, _ = process_vision_info(messages)
            all_image_inputs.extend(image_inputs if image_inputs else [])

        # Process the batch
        batch = processor(
            text=texts,
            images=all_image_inputs if all_image_inputs else None,
            padding=True,
            return_tensors="pt"
        )

        # The labels are the input_ids, and we mask the padding tokens in the loss computation
        labels = batch["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100  #

        # Ignore the image token index in the loss computation (model specific)
        if isinstance(processor, Qwen2VLProcessor):
            logger.info("DETECTED PROCESSOR")
            image_tokens = [151652,151653,151655]
        else: 
            image_tokens = [processor.tokenizer.convert_tokens_to_ids(processor.image_token)]
        for image_token_id in image_tokens:
            labels[labels == image_token_id] = -100
        batch["labels"] = labels

        return batch

    return collate_fn

def main(script_args, training_args, model_args):
    # Force single GPU mode if requested
    # if hasattr(script_args, 'single_gpu') and script_args.single_gpu:
    #     logger.info("Single GPU mode requested - setting CUDA_VISIBLE_DEVICES=0")
    #     # Disable distributed training
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # training_args.local_rank = -1
    # training_args.ddp_backend = None

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

    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    if "wandb" in training_args.report_to:
        init_wandb_training(training_args)

    ######################################
    # Load dataset, processor/tokenizer, and model #
    ######################################
    dataset = get_dataset(script_args)

    if training_args.vision_model:
        logger.info("Setting up vision-language model training")
        
        # Set VLM-specific training arguments (following TRL reference)
        training_args.gradient_checkpointing_kwargs = dict(use_reentrant=False)
        training_args.remove_unused_columns = False
        training_args.dataset_kwargs = {"skip_prepare_dataset": True}
        training_args.ddp_find_unused_parameters = True

        # Load processor and model for VLM
        processor = get_processor(model_args, training_args)
        model = get_model(model_args, training_args)  # This should return AutoModelForVision2Seq
        data_collator = create_vlm_collate_fn(processor)
        processing_class = processor.tokenizer
        model_tags = ["open-r1", "vision-language", "vlm"]
        
    else:
        logger.info("Setting up text-only model training")
        
        # Load tokenizer and model for text-only
        tokenizer = get_tokenizer(model_args, training_args)
        model = get_model(model_args, training_args)
        
        if tokenizer.chat_template is None:
            logger.info("No chat template provided, defaulting to ChatML.")
            model, tokenizer = setup_chat_format(model, tokenizer, format="chatml")
            
        data_collator = None  # Use default
        processing_class = tokenizer
        model_tags = ["open-r1"]

    ############################
    # Initialize the SFT Trainer
    ############################
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=(
            dataset[script_args.dataset_test_split]
            if training_args.eval_strategy != "no"
            else None
        ),
        processing_class=processing_class,
        peft_config=get_peft_config(model_args),
        callbacks=get_callbacks(training_args, model_args),
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
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "dataset_name": script_args.dataset_name,
        "tags": model_tags,
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
        trainer.push_to_hub(**kwargs, token=os.getenv("HF_TOKEN"))
        # Also push processor for VLM models
        if training_args.vision_model and trainer.accelerator.is_main_process:
            processor.push_to_hub(training_args.hub_model_id)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)

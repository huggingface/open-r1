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

"""Reward modeling implementation for DeepSeek-R1."""

from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config

@dataclass
class RewardModelScriptArguments(ScriptArguments):
    """Script arguments for reward model training."""
    comparison_column: str = field(
        default="comparison",
        metadata={"help": "Column containing preference comparisons"}
    )
    better_response_column: str = field(
        default="better_response",
        metadata={"help": "Column containing preferred responses"}
    )
    worse_response_column: str = field(
        default="worse_response",
        metadata={"help": "Column containing non-preferred responses"}
    )

def prepare_comparison_dataset(dataset, tokenizer, args):
    """Prepare dataset for reward modeling."""
    def tokenize_pairs(examples):
        better_responses = examples[args.better_response_column]
        worse_responses = examples[args.worse_response_column]
        
        better_encodings = tokenizer(
            better_responses,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length"
        )
        worse_encodings = tokenizer(
            worse_responses,
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length"
        )
        
        return {
            "better_input_ids": better_encodings["input_ids"],
            "better_attention_mask": better_encodings["attention_mask"],
            "worse_input_ids": worse_encodings["input_ids"],
            "worse_attention_mask": worse_encodings["attention_mask"],
            "labels": [1] * len(better_responses)  # 1 indicates better response
        }
    
    return dataset.map(
        tokenize_pairs,
        batched=True,
        remove_columns=dataset.column_names
    )

def main(script_args, training_args, model_args):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=model_args.trust_remote_code
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1,
        trust_remote_code=model_args.trust_remote_code
    )
    
    # Load and prepare dataset
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)
    processed_dataset = prepare_comparison_dataset(dataset, tokenizer, script_args)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset[script_args.dataset_train_split],
        eval_dataset=processed_dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args)
    )
    
    # Train the model
    trainer.train()
    
    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)

if __name__ == "__main__":
    parser = TrlParser((RewardModelScriptArguments, TrainingArguments, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
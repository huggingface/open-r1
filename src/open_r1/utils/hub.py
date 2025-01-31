#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import re
from concurrent.futures import Future

from transformers import AutoConfig

from huggingface_hub import (
    create_branch,
    create_repo,
    get_safetensors_metadata,
    list_repo_commits,
    list_repo_files,
    list_repo_refs,
    repo_exists,
    upload_folder,
)
from trl import GRPOConfig, SFTConfig


logger = logging.getLogger(__name__)


def push_to_hub_revision(training_args: SFTConfig | GRPOConfig, extra_ignore_patterns=[]) -> Future:
    """Pushes the model to branch on a Hub repo."""

    # Create a repo if it doesn't exist yet
    repo_url = create_repo(repo_id=training_args.hub_model_id, private=True, exist_ok=True)
    # Get initial commit to branch from
    initial_commit = list_repo_commits(training_args.hub_model_id)[-1]
    # Now create the branch we'll be pushing to
    create_branch(
        repo_id=training_args.hub_model_id,
        branch=training_args.hub_model_revision,
        revision=initial_commit.commit_id,
        exist_ok=True,
    )
    logger.info(f"Created target repo at {repo_url}")
    logger.info(f"Pushing to the Hub revision {training_args.hub_model_revision}...")
    ignore_patterns = ["checkpoint-*", "*.pth"]
    ignore_patterns.extend(extra_ignore_patterns)
    future = upload_folder(
        repo_id=training_args.hub_model_id,
        folder_path=training_args.output_dir,
        revision=training_args.hub_model_revision,
        commit_message=f"Add {training_args.hub_model_revision} checkpoint",
        ignore_patterns=ignore_patterns,
        run_as_future=True,
    )
    logger.info(f"Pushed to {repo_url} revision {training_args.hub_model_revision} successfully!")

    return future


def check_hub_revision_exists(training_args: SFTConfig | GRPOConfig):
    """Checks if a given Hub revision exists."""
    if repo_exists(training_args.hub_model_id):
        if training_args.push_to_hub_revision is True:
            # First check if the revision exists
            revisions = [rev.name for rev in list_repo_refs(training_args.hub_model_id).branches]
            # If the revision exists, we next check it has a README file
            if training_args.hub_model_revision in revisions:
                repo_files = list_repo_files(
                    repo_id=training_args.hub_model_id, revision=training_args.hub_model_revision
                )
                if "README.md" in repo_files and training_args.overwrite_hub_revision is False:
                    raise ValueError(
                        f"Revision {training_args.hub_model_revision} already exists. "
                        "Use --overwrite_hub_revision to overwrite it."
                    )


def get_param_count_from_repo_id(repo_id: str) -> int:
    """Function to get model param counts from safetensors metadata or find patterns like 42m, 1.5b, 0.5m or products like 8x7b in a repo ID."""
    try:
        metadata = get_safetensors_metadata(repo_id)
        return list(metadata.parameter_count.values())[0]
    except Exception:
        # Pattern to match products (like 8x7b) and single values (like 42m)
        pattern = r"((\d+(\.\d+)?)(x(\d+(\.\d+)?))?)([bm])"
        matches = re.findall(pattern, repo_id.lower())

        param_counts = []
        for full_match, number1, _, _, number2, _, unit in matches:
            if number2:  # If there's a second number, it's a product
                number = float(number1) * float(number2)
            else:  # Otherwise, it's a single value
                number = float(number1)

            if unit == "b":
                number *= 1_000_000_000  # Convert to billion
            elif unit == "m":
                number *= 1_000_000  # Convert to million

            param_counts.append(number)

        if len(param_counts) > 0:
            # Return the largest number
            return int(max(param_counts))
        else:
            # Return -1 if no match found
            return -1


def get_gpu_count_for_vllm(model_name: str, revision: str = "main", num_gpus: int = 8) -> int:
    """vLLM enforces a constraint that the number of attention heads must be divisible by the number of GPUs and 64 must be divisible by the number of GPUs.
    This function calculates the number of GPUs to use for decoding based on the number of attention heads in the model.
    """
    config = AutoConfig.from_pretrained(model_name, revision=revision, trust_remote_code=True)
    # Get number of attention heads
    num_heads = config.num_attention_heads
    # Reduce num_gpus so that num_heads is divisible by num_gpus and 64 is divisible by num_gpus
    while num_heads % num_gpus != 0 or 64 % num_gpus != 0:
        logger.info(f"Reducing num_gpus from {num_gpus} to {num_gpus - 1} to make num_heads divisible by num_gpus")
        num_gpus -= 1
    return num_gpus

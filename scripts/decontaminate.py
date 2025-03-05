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
"""
This script is used to decontaminate a dataset by checking for n-gram overlap with other datasets.
It uses the same approach presented in https://arxiv.org/abs/2501.19393,
as found in: https://github.com/simplescaling/s1/blob/main/data/decontaminate_util.py

Usage:

python scripts/decontaminate.py \
    --dataset open-r1/verifiable-coding-problems-python \
    --split train \
    --ngram_size 8 \
    --problem_column problem \
    --cleanup
"""

import collections

from tqdm import tqdm


def normalize_string(text: str) -> str:
    """Basic string normalization."""
    # Convert to lowercase and normalize whitespace
    text = text.lower().strip()
    # Replace multiple spaces with single space
    text = " ".join(text.split())
    return text


def word_ngrams(text: str, n: int) -> list:
    """Generate word-level n-grams from text."""
    words = text.split()
    return [" ".join(words[i : i + n]) for i in range(len(words) - n + 1)]


def build_ngram_lookup(documents: list[str], ngram_size: int = 8) -> dict[str, set[int]]:
    """Build ngram lookup for documents."""
    lookup = collections.defaultdict(set)

    for doc_id, document in enumerate(tqdm(documents)):
        normalized_text = normalize_string(document)
        ngrams = word_ngrams(normalized_text, ngram_size)
        for ngram in ngrams:
            lookup[ngram].add(doc_id)

    return lookup


def build_ngram_single(document: str, ngram_size: int = 8) -> set[str]:
    normalized_text = normalize_string(document)
    ngrams = word_ngrams(normalized_text, ngram_size)

    return set(ngrams)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Name of the dataset to check for contamination.")
    parser.add_argument("--config", type=str, default=None, help="Name of the dataset config to load.")
    parser.add_argument("--split", type=str, default="train", help="Split to check for contamination, defaults to `train`.")
    parser.add_argument("--ngram_size", type=int, default=8, help="Size of n-grams to build, defaults to 8.")
    parser.add_argument(
        "--problem_column", type=str, default="problem", help="Name of the column containing the problem (prompt)."
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Whether to remove the contaminated rows before pushing the dataset.",
    )
    parser.add_argument(
        "--new_dataset_name",
        type=str,
        default=None,
        help="New name for the dataset. If not provided, will reuse the name and add a `_decontaminated` to the name."
    )
    args = parser.parse_args()

    from datasets import load_dataset, Dataset

    # Load the dataset to check for contamination
    ds = load_dataset(args.dataset, name=args.config, split=args.split)

    eval_datasets = {
        "aime_2024": (load_dataset("HuggingFaceH4/aime_2024", split="train"), "problem"),
        "aime_2025": (load_dataset("yentinglin/aime_2025", split="train"), "problem"),
        "math_500": (load_dataset("HuggingFaceH4/MATH-500", split="test"), "problem"),
        "gpqa": (load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", trust_remote_code=True), "Question"),
        "lcb": (
            load_dataset(
                "livecodebench/code_generation_lite", split="test", version_tag="v4_v5", trust_remote_code=True
            ),
            "question_content",
        ),
    }
    ngram_lookups = {}
    for ds_name, (eval_dataset, problem_col) in eval_datasets.items():
        ngram_lookups[ds_name] = build_ngram_lookup(eval_dataset[problem_col], ngram_size=args.ngram_size)

    for eval_name, ngram_lookup in ngram_lookups.items():
        # Update the ngram_lookup variable for each dataset
        def find_contaminated(row):
            # For each example we have to build the ngrams and check for all of them on each row
            ngrams = build_ngram_single(row[args.problem_column], ngram_size=args.ngram_size)
            row[f"contaminated_{eval_name}"] = any(set(ngram in ngram_lookup for ngram in ngrams))
            return row

        ds = ds.map(find_contaminated, num_proc=8)

    # Allow cleaning up via CLI args (removing the contaminated examples and dropping the columns)
    def cleanup(dataset: Dataset) -> Dataset:
        initial_size = len(dataset)
        contamination_cols = [col for col in dataset.column_names if col.startswith("contaminated_")]
        for col in contamination_cols:
            if col.startswith("contaminated_"):
                size_prior = len(dataset)
                dataset = dataset.filter(lambda x: not x[col], num_proc=8)
                if len(dataset) < size_prior:
                    print(f"Removed {size_prior - len(dataset)} samples from '{col.replace('contaminated_', '')}'")
        dataset = dataset.remove_columns(contamination_cols)
        print(f"Initial size: {initial_size}, Final size: {len(dataset)}")
        return dataset

    if args.cleanup:
        ds = cleanup(ds)

    new_ds_name = args.new_dataset_name or f"{args.dataset}_decontaminated"
    config_name = args.config if args.config is not None else "default"
    url = ds.push_to_hub(new_ds_name, config_name=config_name, split="train")
    print(f"Decontaminated dataset: {url}")

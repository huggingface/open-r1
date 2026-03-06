#!/usr/bin/env python
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

import argparse
from collections import Counter
from dataclasses import dataclass
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer


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


@dataclass
class SplitStats:
    split: str
    rows_before: int
    rows_after_make_conversation: int
    rows_after_length_trim: int
    rows_after_label_filter: int
    label_filter_applied: bool
    solutions_before: int
    solutions_after_make_conversation: int
    solutions_after_length_trim: int
    solutions_after_label_filter: int
    dropped_by_length_total: int
    dropped_by_length_correct: int
    dropped_by_length_wrong: int
    rows_with_both_labels_before_label_filter: int
    rows_with_only_correct_before_label_filter: int
    rows_with_only_wrong_before_label_filter: int
    rows_with_no_labels_before_label_filter: int


def make_conversation(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
    questions: list[str] = []
    all_solutions: list[list[dict[str, Any]]] = []

    for question, correct_sols, wrong_sols in zip(
        batch["question"], batch["correct_solutions"], batch["wrong_solutions"]
    ):
        questions.append(question)
        sols = []
        for sol in correct_sols:
            sols.append({"solve_func": sol["solve_func"], "is_correct": True})
        for sol in wrong_sols:
            sols.append({"solve_func": sol["solve_func"], "is_correct": False})
        all_solutions.append(sols)

    return {"prompt_question": questions, "solutions": all_solutions}


def label_bucket(solutions: list[dict[str, Any]]) -> str:
    labels = {sol.get("is_correct") for sol in solutions if sol.get("is_correct") in (True, False)}
    if labels == {True, False}:
        return "both"
    if labels == {True}:
        return "only_correct"
    if labels == {False}:
        return "only_wrong"
    return "none"


def has_both_labels(example: dict[str, Any]) -> bool:
    return label_bucket(example["solutions"]) == "both"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze how many rows/solutions are filtered by the GRPO preprocessing logic."
    )
    parser.add_argument("--dataset-name", type=str, default="zzzzit/taco3", help="HuggingFace dataset name.")
    parser.add_argument("--dataset-config", type=str, default=None, help="Dataset config/subset name.")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated split names. Default: all splits in the dataset.",
    )
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        required=True,
        help="Tokenizer name/path used to compute prompt length (should match training tokenizer).",
    )
    parser.add_argument(
        "--max-prompt-length",
        type=int,
        default=1024,
        help="Same threshold as training_args.max_prompt_length. Set to <=0 to skip length trimming.",
    )
    parser.add_argument(
        "--use-system-prompt",
        action="store_true",
        help="If set, prepend the fixed SYSTEM_PROMPT exactly like grpo.py does when system_prompt is configured.",
    )
    parser.add_argument(
        "--num-proc",
        type=int,
        default=None,
        help="Number of processes for dataset map/filter.",
    )
    return parser.parse_args()


def count_original_solutions(split_ds: Dataset) -> tuple[int, int, int]:
    total = 0
    total_correct = 0
    total_wrong = 0
    for example in split_ds:
        correct = len(example["correct_solutions"])
        wrong = len(example["wrong_solutions"])
        total_correct += correct
        total_wrong += wrong
        total += correct + wrong
    return total, total_correct, total_wrong


def count_grouped_solutions(split_ds: Dataset) -> tuple[int, int, int]:
    total = 0
    total_correct = 0
    total_wrong = 0
    for example in split_ds:
        for sol in example["solutions"]:
            total += 1
            if sol.get("is_correct") is True:
                total_correct += 1
            elif sol.get("is_correct") is False:
                total_wrong += 1
    return total, total_correct, total_wrong


def analyze_split(
    split_name: str,
    split_ds: Dataset,
    tokenizer: Any,
    max_prompt_length: int | None,
    use_system_prompt: bool,
    num_proc: int | None,
) -> SplitStats:
    rows_before = len(split_ds)
    solutions_before, _, _ = count_original_solutions(split_ds)

    grouped_ds = split_ds.map(
        make_conversation,
        batched=True,
        remove_columns=split_ds.column_names,
        num_proc=num_proc,
        desc=f"[{split_name}] make_conversation",
    )
    rows_after_make_conversation = len(grouped_ds)
    solutions_after_make_conversation, _, _ = count_grouped_solutions(grouped_ds)

    dropped_by_length_total = 0
    dropped_by_length_correct = 0
    dropped_by_length_wrong = 0
    rows_after_length_trim = len(grouped_ds)
    rows_after_label_filter = len(grouped_ds)
    label_filter_applied = False

    if max_prompt_length is not None:

        def _prompt_length(question: str, solve_func: str) -> int:
            prompt = []
            if use_system_prompt:
                prompt.append({"role": "system", "content": SYSTEM_PROMPT})
            prompt.append(
                {
                    "role": "user",
                    "content": USER_PROMPT.format(question=question, code_solution=solve_func),
                }
            )
            tokens = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True)
            return len(tokens)

        def trim_solutions(example: dict[str, Any]) -> dict[str, Any]:
            question = example["prompt_question"]
            kept = []
            dropped_correct = 0
            dropped_wrong = 0
            for sol in example["solutions"]:
                if _prompt_length(question, sol["solve_func"]) <= max_prompt_length:
                    kept.append(sol)
                else:
                    if sol.get("is_correct") is True:
                        dropped_correct += 1
                    elif sol.get("is_correct") is False:
                        dropped_wrong += 1
            example["solutions"] = kept
            example["_dropped_by_length_total"] = dropped_correct + dropped_wrong
            example["_dropped_by_length_correct"] = dropped_correct
            example["_dropped_by_length_wrong"] = dropped_wrong
            return example

        grouped_ds = grouped_ds.map(
            trim_solutions,
            num_proc=num_proc,
            desc=f"[{split_name}] trim_solutions",
        )
        rows_after_length_trim = len(grouped_ds)
        dropped_by_length_total = sum(grouped_ds["_dropped_by_length_total"])
        dropped_by_length_correct = sum(grouped_ds["_dropped_by_length_correct"])
        dropped_by_length_wrong = sum(grouped_ds["_dropped_by_length_wrong"])

    label_counts = Counter(label_bucket(example["solutions"]) for example in grouped_ds)
    rows_with_both_labels_before_label_filter = label_counts.get("both", 0)
    rows_with_only_correct_before_label_filter = label_counts.get("only_correct", 0)
    rows_with_only_wrong_before_label_filter = label_counts.get("only_wrong", 0)
    rows_with_no_labels_before_label_filter = label_counts.get("none", 0)

    filtered_ds = grouped_ds
    if max_prompt_length is not None:
        filtered_ds = grouped_ds.filter(
            has_both_labels,
            num_proc=num_proc,
            desc=f"[{split_name}] keep rows with both correct and wrong solutions",
        )
        rows_after_label_filter = len(filtered_ds)
        label_filter_applied = True

    solutions_after_length_trim, _, _ = count_grouped_solutions(grouped_ds)
    solutions_after_label_filter, _, _ = count_grouped_solutions(filtered_ds)

    return SplitStats(
        split=split_name,
        rows_before=rows_before,
        rows_after_make_conversation=rows_after_make_conversation,
        rows_after_length_trim=rows_after_length_trim,
        rows_after_label_filter=rows_after_label_filter,
        label_filter_applied=label_filter_applied,
        solutions_before=solutions_before,
        solutions_after_make_conversation=solutions_after_make_conversation,
        solutions_after_length_trim=solutions_after_length_trim,
        solutions_after_label_filter=solutions_after_label_filter,
        dropped_by_length_total=dropped_by_length_total,
        dropped_by_length_correct=dropped_by_length_correct,
        dropped_by_length_wrong=dropped_by_length_wrong,
        rows_with_both_labels_before_label_filter=rows_with_both_labels_before_label_filter,
        rows_with_only_correct_before_label_filter=rows_with_only_correct_before_label_filter,
        rows_with_only_wrong_before_label_filter=rows_with_only_wrong_before_label_filter,
        rows_with_no_labels_before_label_filter=rows_with_no_labels_before_label_filter,
    )


def print_split_stats(stats: SplitStats) -> None:
    rows_removed_total = stats.rows_before - stats.rows_after_label_filter
    row_drop_pct = (rows_removed_total / stats.rows_before * 100) if stats.rows_before else 0.0
    row_drop_after_trim = stats.rows_after_length_trim - stats.rows_after_label_filter

    print(f"\n=== Split: {stats.split} ===")
    print(f"Rows: {stats.rows_before} -> {stats.rows_after_label_filter} (removed {rows_removed_total}, {row_drop_pct:.2f}%)")
    if stats.label_filter_applied:
        print(f"Rows removed in final label filter: {row_drop_after_trim}")
    else:
        print("Rows removed in final label filter: skipped (max_prompt_length is None)")
    print(
        "Label composition before final label filter: "
        f"both={stats.rows_with_both_labels_before_label_filter}, "
        f"only_correct={stats.rows_with_only_correct_before_label_filter}, "
        f"only_wrong={stats.rows_with_only_wrong_before_label_filter}, "
        f"none={stats.rows_with_no_labels_before_label_filter}"
    )

    if stats.dropped_by_length_total > 0:
        print(
            "Solutions dropped by max_prompt_length: "
            f"{stats.dropped_by_length_total} "
            f"(correct={stats.dropped_by_length_correct}, wrong={stats.dropped_by_length_wrong})"
        )
    else:
        print("Solutions dropped by max_prompt_length: 0")
    print(f"Solutions after length trim: {stats.solutions_before} -> {stats.solutions_after_length_trim}")
    print(f"Solutions after label filter: {stats.solutions_before} -> {stats.solutions_after_label_filter}")


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    dataset = load_dataset(args.dataset_name, args.dataset_config)
    if isinstance(dataset, Dataset):
        dataset = DatasetDict({"train": dataset})

    if args.splits:
        split_names = [split.strip() for split in args.splits.split(",") if split.strip()]
        missing = [split for split in split_names if split not in dataset]
        if missing:
            raise ValueError(f"Unknown splits: {missing}. Available splits: {list(dataset.keys())}")
    else:
        split_names = list(dataset.keys())

    max_prompt_length = args.max_prompt_length if args.max_prompt_length > 0 else None

    print("Running GRPO-style filter analysis with:")
    print(f"- dataset: {args.dataset_name}")
    print(f"- dataset_config: {args.dataset_config}")
    print(f"- splits: {split_names}")
    print(f"- tokenizer: {args.tokenizer_name}")
    print(f"- max_prompt_length: {max_prompt_length}")
    print(f"- use_system_prompt: {args.use_system_prompt}")

    all_stats: list[SplitStats] = []
    for split_name in split_names:
        all_stats.append(
            analyze_split(
                split_name=split_name,
                split_ds=dataset[split_name],
                tokenizer=tokenizer,
                max_prompt_length=max_prompt_length,
                use_system_prompt=args.use_system_prompt,
                num_proc=args.num_proc,
            )
        )

    for stats in all_stats:
        print_split_stats(stats)

    total_rows_before = sum(s.rows_before for s in all_stats)
    total_rows_after = sum(s.rows_after_label_filter for s in all_stats)
    total_solutions_before = sum(s.solutions_before for s in all_stats)
    total_solutions_after = sum(s.solutions_after_label_filter for s in all_stats)
    total_rows_removed = total_rows_before - total_rows_after
    total_row_drop_pct = (total_rows_removed / total_rows_before * 100) if total_rows_before else 0.0

    print("\n=== Overall ===")
    print(f"Rows: {total_rows_before} -> {total_rows_after} (removed {total_rows_removed}, {total_row_drop_pct:.2f}%)")
    print(f"Solutions: {total_solutions_before} -> {total_solutions_after}")
    print(f"Dropped by max_prompt_length: {sum(s.dropped_by_length_total for s in all_stats)}")


if __name__ == "__main__":
    main()

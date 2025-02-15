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
Push the details from a LightEval run to the Hub.

Usage:

python src/open_r1/utils/upload_details.py \
    --data_files {path_to_parquet_file} \
    --hub_repo_id {hub_repo_id} \
    --config_name {config_name}
"""

from dataclasses import dataclass, field
from typing import List

from datasets import load_dataset
from transformers import HfArgumentParser


@dataclass
class ScriptArguments:
    data_files: List[str] = field(default_factory=list)
    hub_repo_id: str = None
    config_name: str = None


def main():
    parser = HfArgumentParser(ScriptArguments)
    args = parser.parse_args_into_dataclasses()[0]

    if all(file.endswith(".json") for file in args.data_files):
        ds = load_dataset("json", data_files=args.data_files)
    elif all(file.endswith(".jsonl") for file in args.data_files):
        ds = load_dataset("json", data_files=args.data_files)
    else:
        ds = load_dataset("parquet", data_files=args.data_files)
    url = ds.push_to_hub(args.hub_repo_id, config_name=args.config_name, private=True)
    print(f"Dataset available at: {url}")


if __name__ == "__main__":
    main()

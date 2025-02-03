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
from typing import List, Optional

from open_r1.utils.evaluation import SUPPORTED_BENCHMARKS, run_benchmark_jobs
from open_r1.configs import SFTConfig
from trl import ModelConfig, TrlParser


@dataclass
class ScriptArguments:
    model_id: str = field(
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        metadata={"help": "The Hub model id to push the model to."},
    )
    model_revision: str = field(default="main", metadata={"help": "The Hub model branch to push the model to."})
    trust_remote_code: bool = field(default=False, metadata={"help": "Trust the remote code."})
    benchmarks: List[str] = field(
        default_factory=lambda: [], metadata={"help": "The benchmarks to run after training."}
    )
    list_benchmarks: bool = field(default=False, metadata={"help": "List all supported benchmarks."})
    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "The system prompt to use for the benchmark."}
    )


def main():
    parser = TrlParser(ScriptArguments)
    args = parser.parse_args_and_config()[0]
    if args.list_benchmarks:
        print("Supported benchmarks:")
        for benchmark in SUPPORTED_BENCHMARKS:
            print(f"  - {benchmark}")
        return
    benchmark_args = SFTConfig(
        output_dir="",
        hub_model_id=args.model_id,
        hub_model_revision=args.model_revision,
        benchmarks=args.benchmarks,
        system_prompt=args.system_prompt,
    )
    run_benchmark_jobs(
        benchmark_args,
        ModelConfig(model_name_or_path="", model_revision="", trust_remote_code=args.trust_remote_code),
    )


if __name__ == "__main__":
    main()

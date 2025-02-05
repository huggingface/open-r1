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

from typing import Optional
from typing import Union
from distilabel.llms import OpenAILLM
from distilabel.pipeline import Pipeline
from distilabel.steps import StepResources
from distilabel.steps.tasks import TextGeneration
import sglang as sgl


class LLMBackend:
    def __init__(self, backend: str = "vllm"):
        self.backend = backend

    def get_llm(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
        timeout: int = 900,
        max_retries: int = 0,
        generation_kwargs: dict = None,
    ) -> Union[OpenAILLM, sgl.Engine]:
        """Get LLM instance based on backend."""
        if generation_kwargs is None:
            generation_kwargs = {}

        if self.backend == "sglang":
            # SGLang engine initialization with correct parameters
            sglang_kwargs = {
                "model_path": model,
                "log_level": "error",
                "device": "cuda",
                "dtype": "float16",
                "max_model_len": 4096,  # Add reasonable default max length
            }
            
            # Create engine with basic parameters
            engine = sgl.Engine(**sglang_kwargs)
            
            # Set generation config after initialization
            max_tokens = generation_kwargs.get("max_new_tokens", 512)
            engine.set_generation_params(
                max_tokens=max_tokens,
                temperature=generation_kwargs.get("temperature", 0.7),
                top_p=generation_kwargs.get("top_p", 0.95)
            )
            
            return engine
        
        elif self.backend == "vllm":
            try:
                return OpenAILLM(
                    base_url=base_url,
                    api_key="dummy-key",  # vLLM doesn't require real key
                    model=model,
                    timeout=timeout,
                    max_retries=max_retries,
                    generation_kwargs=generation_kwargs,
                )
            except Exception as e:
                raise ConnectionError(f"Failed to connect to vLLM server at {base_url}: {str(e)}")
        else:
            raise ValueError(f"Unknown backend: {self.backend}")


def build_distilabel_pipeline(
    model: str,
    backend: str = "vllm",
    base_url: str = "http://localhost:8000/v1",
    prompt_column: Optional[str] = None,
    prompt_template: str = "{{ instruction }}",
    generation_kwargs: Optional[dict] = None,
    num_generations: int = 1,
    input_batch_size: int = 64,
    client_replicas: int = 1,
    timeout: int = 900,
    retries: int = 0,
    unique_id: Optional[str] = None,
) -> Pipeline:
    """Build a distilabel pipeline for text generation.
    
    Args:
        model (str): Model identifier or path
        backend (str, optional): Backend to use ("vllm" or "sglang"). Defaults to "vllm".
        base_url (str, optional): Base URL for vLLM server. Defaults to "http://localhost:8000/v1".
        prompt_column (str, optional): Column name containing prompts. Defaults to None.
        prompt_template (str, optional): Template for formatting prompts. Defaults to "{{ instruction }}".
        generation_kwargs (dict, optional): Generation parameters. Defaults to None.
        num_generations (int, optional): Number of generations per prompt. Defaults to 1.
        input_batch_size (int, optional): Batch size for processing. Defaults to 64.
        client_replicas (int, optional): Number of client replicas. Defaults to 1.
        timeout (int, optional): Timeout in seconds. Defaults to 900.
        retries (int, optional): Number of retries. Defaults to 0.
        unique_id (str, optional): Unique identifier for the pipeline. Defaults to None.

    Returns:
        Pipeline: Configured distilabel pipeline
    """
    if generation_kwargs is None:
        generation_kwargs = {}

    llm_backend = LLMBackend(backend=backend)
    pipeline_name = f"pipeline_{unique_id}" if unique_id else "pipeline"
    
    with Pipeline(name=pipeline_name).ray() as pipeline:
        TextGeneration(
            llm=llm_backend.get_llm(
                model,
                base_url=base_url,
                timeout=timeout,
                max_retries=retries,
                generation_kwargs=generation_kwargs,
            ),
            template=prompt_template,
            input_mappings={"instruction": prompt_column} if prompt_column is not None else {},
            input_batch_size=input_batch_size,
            num_generations=num_generations,
            group_generations=True,
            resources=StepResources(replicas=client_replicas),
        )

    return pipeline


if __name__ == "__main__":
    import argparse
    from datasets import load_dataset

    parser = argparse.ArgumentParser(description="Run distilabel pipeline for generating responses")
    parser.add_argument(
        "--hf-dataset",
        type=str,
        required=True,
        help="HuggingFace dataset to load",
    )
    parser.add_argument(
        "--hf-dataset-config",
        type=str,
        required=False,
        help="Dataset config to use",
    )
    parser.add_argument(
        "--hf-dataset-split",
        type=str,
        default="train",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--prompt-column",
        type=str,
        default="prompt",
        help="Column containing prompts",
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        default="{{ instruction }}",
        help="Template string for formatting prompts.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name to use for generation",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="vllm",
        choices=["vllm", "sglang"],
        help="Backend to use for generation",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for the backend server",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for generation",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p value for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--input-batch-size",
        type=int,
        default=32,
        help="Batch size for processing inputs",
    )
    parser.add_argument(
        "--client-replicas",
        type=int,
        default=1,
        help="Number of client replicas",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=900,
        help="Timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=0,
        help="Number of retries",
    )
    parser.add_argument(
        "--hf-output-dataset",
        type=str,
        help="HuggingFace dataset name to push results to",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Whether to make the output dataset private",
    )

    args = parser.parse_args()

    # Load dataset
    dataset = load_dataset(
        args.hf_dataset,
        args.hf_dataset_config,
        split=args.hf_dataset_split,
    )

    # Prepare generation kwargs
    generation_kwargs = {
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_new_tokens": args.max_new_tokens,
    }

    # Build and run pipeline
    pipeline = build_distilabel_pipeline(
        model=args.model,
        backend=args.backend,
        base_url=args.base_url,
        prompt_column=args.prompt_column,
        prompt_template=args.prompt_template,
        generation_kwargs=generation_kwargs,
        input_batch_size=args.input_batch_size,
        client_replicas=args.client_replicas,
        timeout=args.timeout,
        retries=args.retries,
    )

    print("Running generation pipeline...")
    distiset = pipeline.run(
        dataset=dataset,
        dataset_batch_size=args.input_batch_size * 1000,
        use_cache=False,
    )
    print("Generation pipeline finished!")

    if args.hf_output_dataset:
        print(f"Pushing resulting dataset to '{args.hf_output_dataset}'...")
        distiset.push_to_hub(args.hf_output_dataset, private=args.private)
        print("Dataset pushed!")
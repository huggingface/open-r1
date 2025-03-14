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

# great reference: https://github.com/vllm-project/vllm/issues/11400


import time

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import requests


class RemoteModel:
    """
    launch with:
    export LD_LIBRARY_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/nvjitlink/lib')"):$LD_LIBRARY_PATH
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port=30010 --skip-tokenizer-init --mem-fraction-static 0.4
    python3 -m sglang.launch_server --model-path HuggingFaceTB/SmolLM2-135M-Instruct --port=30010 --skip-tokenizer-init --mem-fraction-static 0.4 --host=0.0.0.0

    # on a separate node
    python3 -m sglang.launch_server --model-path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B --port=30010 --skip-tokenizer-init --mem-fraction-static 0.7 --host=0.0.0.0 --dp-size=8

    python3 -m sglang.launch_server --model-path HuggingFaceTB/SmolLM2-1.7B-Instruct --port=30010 --skip-tokenizer-init --mem-fraction-static 0.6 --host=0.0.0.0 --dp-size=8


    python3 -m sglang.launch_server --model-path open-r1/Qwen2.5-Coder-7B-Instruct-SFT --revision v00.08-step-000001280  --port=30010 --skip-tokenizer-init --mem-fraction-static 0.7 --host=0.0.0.0 --dp-size=8
    """

    def __init__(self, remote_model_url, remote_model_port, stop_token_id=None):
        self.remote_model_url = remote_model_url
        self.remote_model_port = remote_model_port
        self.stop_token_id = stop_token_id

    def is_healthy(self, timeout=5):
        """Checks if the remote model server is up and running."""
        try:
            url = f"http://{self.remote_model_url}:{self.remote_model_port}/health"
            response = requests.get(url, timeout=timeout)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def wait_for_server(self, max_retries=120, delay=5):
        """Waits for the server to become available before proceeding."""
        for attempt in range(max_retries):
            if self.is_healthy():
                print("Remote model server is healthy!")
                return True
            print(f"Waiting for server to start... (Attempt {attempt + 1}/{max_retries})")
            time.sleep(delay)
        raise RuntimeError("Remote model server did not start in time.")

    def generate(
        self, input_ids: list[list[int]], max_new_tokens=256, temperature=0.8, num_generations=2
    ) -> tuple[list[list[int]], list[list[int]]]:
        # Prepare the request body
        request_body = {
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop_token_ids": [self.stop_token_id],
                "n": num_generations,
            },
            "stream": False,
            "return_logprob": True,
            "logprob_start_len": 0,
        }

        # Send the POST request to the server
        # add a few retries?
        response = requests.post(
            f"http://{self.remote_model_url}:{self.remote_model_port}/generate", json=request_body
        )
        response_json = response.json()

        examples = []

        for i, result in enumerate(response_json):
            prompt_index = i // num_generations
            prompt_ids = input_ids[prompt_index]
            completion_ids = result["token_ids"]
            prompt_log_probs = [prob[0] for prob in result["meta_info"]["input_token_logprobs"]]
            completion_log_probs = [prob[0] for prob in result["meta_info"]["output_token_logprobs"]]

            example = {
                "prompt_ids": prompt_ids,
                "completion_ids": completion_ids,
                "prompt_log_probs": prompt_log_probs,
                "completion_log_probs": completion_log_probs,
            }
            examples.append(example)

        return examples

    def load_weights_from_path(self, path: str):
        url = f"http://{self.remote_model_url}:{self.remote_model_port}/update_weights_from_disk"
        data = {"model_path": path}

        response = requests.post(url, json=data)
        print(response.text)
        assert response.json()["success"] is True
        # assert response.json()["message"] == "Succeeded to update model weights."
        assert response.json().keys() == {"success", "message"}


if __name__ == "__main__":
    from datasets import load_dataset

    url = "0.0.0.0"
    port = 30010
    MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    remote_model = RemoteModel(url, port, tokenizer.eos_token_id)
    dataset = load_dataset("AI-MO/NuminaMath-TIR", split="train")
    dataloader = DataLoader(dataset, batch_size=4)

    for i, batch in zip(range(2), dataloader):
        problems = batch["problem"]
        ids = tokenizer(problems)
        new_ids, logprobs = remote_model.generate(ids["input_ids"])
        print(new_ids)
        print(logprobs)

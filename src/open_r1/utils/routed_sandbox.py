# coding=utf-8
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

import requests
from e2b_code_interpreter.models import Execution, ExecutionError, Result


class RoutedSandbox:
    """
    A sandbox environment that routes code execution requests to the E2B Router.
    This class is designed for batched execution of scripts, primarily for Python code.
    It mimics the usage of 'Sandbox' from 'e2b_code_interpreter', but adds support for batch processing.

    Attributes:
        router_url (str): The URL of the E2B Router to which code execution requests are sent.
    """

    def __init__(self, router_url: str):
        """
        Initializes the RoutedSandbox with the specified router URL.

        Args:
            router_url (str): The URL of the E2B Router.
        """
        self.router_url = router_url

    def run_code(
        self,
        scripts: list[str],
        language: str = "python",
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> list[Execution]:
        """
        Executes a batch of scripts in the sandbox environment.

        Args:
            scripts (list[str]): A list of code scripts to execute.
            language (str, optional): The programming language of the scripts. Defaults to "python".
            timeout (Optional[int], optional): The maximum execution time for each script in seconds. Defaults to 300 seconds.
            request_timeout (Optional[int], optional): The timeout for the HTTP request in seconds. Defaults to 30 seconds.

        Returns:
            list[Execution]: A list of Execution objects containing the results, logs, and errors (if any) for each script.
        """
        # Set default values for timeouts if not provided
        if timeout is None:
            timeout = 300  # Default to 5 minutes
        if request_timeout is None:
            request_timeout = 30  # Default to 30 seconds

        # Prepare the payload for the HTTP POST request
        payload = {
            "scripts": scripts,
            "language": language,
            "timeout": timeout,
            "request_timeout": request_timeout,
        }

        # Send the request to the E2B Router
        response = requests.post(f"http://{self.router_url}/execute_batch", json=payload)
        if not response.ok:
            print(f"Request failed with status code: {response.status_code}")

        # Parse the response and construct Execution objects
        results = response.json()
        output = []
        for result in results:
            if result["execution"] is None:
                # If execution is None, create an empty Execution object
                # This can happen when a script times out or fails to execute
                execution = Execution()
            else:
                execution = Execution(
                    results=[Result(**r) for r in result["execution"]["results"]],
                    logs=result["execution"]["logs"],
                    error=ExecutionError(**result["execution"]["error"]) if result["execution"]["error"] else None,
                    execution_count=result["execution"]["execution_count"],
                )
            output.append(execution)

        return output


if __name__ == "__main__":
    # for local testing launch an E2B router with: python scripts/e2b_router.py
    sbx = RoutedSandbox(router_url="0.0.0.0:8000")
    codes = ["print('hello world')", "print('hello world)"]
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)

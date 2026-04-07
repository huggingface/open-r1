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

from typing import List, Optional

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
        languages: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> list[Execution]:
        """
        Executes a batch of scripts in the sandbox environment.

        Args:
            scripts (list[str]): A list of code scripts to execute.
            languages (list[str], optional): List of programming languages for each script. If None, defaults to Python for all scripts.
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

        # Default to Python for all scripts if languages is not provided
        if languages is None:
            languages = ["python"] * len(scripts)

        # Prepare the payload for the HTTP POST request
        payload = {
            "scripts": scripts,
            "languages": languages,
            "timeout": timeout,
            "request_timeout": request_timeout,
        }

        try:
            # Send the request to the E2B Router
            response = requests.post(f"http://{self.router_url}/execute_batch", json=payload, timeout=request_timeout)
            if not response.ok:
                print(f"Request failed with status code: {response.status_code}")
                # Return empty Executions for each script to avoid hard failures
                return [Execution() for _ in scripts]

            # Parse the response and construct Execution objects
            try:
                results = response.json()
            except Exception as e:
                print(f"Failed to parse router JSON response: {e}")
                return [Execution() for _ in scripts]

            output = []
            for result in results if isinstance(results, list) else []:
                try:
                    exec_payload = result.get("execution") if isinstance(result, dict) else None
                    if not exec_payload:
                        execution = Execution()
                    else:
                        execution = Execution(
                            results=[Result(**r) for r in exec_payload.get("results", [])],
                            logs=exec_payload.get("logs"),
                            error=(ExecutionError(**exec_payload["error"]) if exec_payload.get("error") else None),
                            execution_count=exec_payload.get("execution_count"),
                        )
                except Exception as e:
                    print(f"Failed to build Execution from router item: {e}")
                    execution = Execution()
                output.append(execution)

            # Fallback if router returned unexpected structure
            if not output:
                return [Execution() for _ in scripts]
            return output
        except Exception as e:
            print(f"Error communicating with E2B router: {e}")
            return [Execution() for _ in scripts]


if __name__ == "__main__":
    # for local testing launch an E2B router with: python scripts/e2b_router.py
    sbx = RoutedSandbox(router_url="0.0.0.0:8000")
    codes = ["print('hello world')", "print('hello world)"]
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)

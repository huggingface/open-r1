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

import argparse
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict
from typing import  Optional
from fastapi import FastAPI, Request
import argparse
import asyncio
from fastapi import FastAPI
import uvicorn
from e2b_code_interpreter.models import Execution
from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox

load_dotenv()

class BatchRequest(BaseModel):
    """
    BatchRequest is a data model representing a batch processing request.

    Attributes:
        scripts (list[str]): A list of script names or paths to be executed.
        language (str): The programming language in which the scripts are written.
        timeout (int): The maximum allowed execution time for each script in seconds.
        request_timeout (int): The maximum allowed time for the entire batch request in seconds.
    """
    scripts: list[str]
    language: str
    timeout: int
    request_timeout: int

class ScriptResult(BaseModel):
    """
    ScriptResult is a Pydantic model that represents the result of a script execution.
    Attributes:
        execution (Optional[Execution]): An optional instance of the `Execution` class 
            that contains details about the script's execution, such as status, output, 
            or any other relevant metadata.
        exception_str (Optional[str]): An optional string that captures the exception 
            message or details if an error occurred during the script's execution.
        model_config (ConfigDict): A configuration dictionary that allows arbitrary 
            types to be used within the Pydantic model. This is necessary to support 
            custom types like `Execution` within the model.
    """
    execution: Optional[Execution]
    exception_str: Optional[str]
    
    # required to allow arbitrary types in pydantic models such as Execution
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
def create_app(args):
    """
    Creates and configures a FastAPI application instance.
    Args:
        args: An object containing configuration parameters for the application.
              - num_sandboxes (int): The maximum number of concurrent sandboxes allowed.
    Returns:
        FastAPI: A configured FastAPI application instance.
    The application includes the following endpoints:
        1. GET /health:
            - Returns the health status of the application.
            - Response: {"status": "ok"}
        2. POST /execute_batch:
            - Executes a batch of scripts in an isolated sandbox environment.
            - Request Body: BatchRequest object containing:
                - language (str): The programming language of the scripts (python or javascript).
                - timeout (int): The maximum execution time for each script.
                - request_timeout (int): The timeout for the request itself.
                - scripts (List[str]): A list of scripts to execute.
            - Response: A list of ScriptResult objects for each script, containing:
                - execution: The result of the script execution.
                - exception_str: Any exception encountered during execution.
    Notes:
        - A semaphore is used to limit the number of concurrent sandboxes.
        - Each script execution is wrapped in a timeout to prevent hanging.
        - Sandboxes are cleaned up after execution, even in case of errors.
    """
    app = FastAPI()

    # Instantiate semaphore and attach it to app state
    app.state.sandbox_semaphore = asyncio.Semaphore(args.max_num_sandboxes)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/execute_batch")
    async def execute_batch(batch: BatchRequest, request: Request):
        semaphore = request.app.state.sandbox_semaphore
        language = batch.language
        timeout = batch.timeout
        request_timeout = batch.request_timeout
        asyncio_timeout = batch.timeout + 1
        
        async def run_script(script: str) -> ScriptResult:

            async with semaphore:
                try:
                    sandbox = await AsyncSandbox.create(
                        timeout=timeout,
                        request_timeout=request_timeout,
                    )
                    execution = await asyncio.wait_for(
                        sandbox.run_code(script, language=language),
                        timeout=asyncio_timeout,
                    )
                    return ScriptResult(execution=execution, exception_str=None)

                except Exception as e:
                    return ScriptResult(execution=None, exception_str=str(e))
                
                finally:
                    try:
                        await sandbox.kill()
                    except Exception:
                        pass

        tasks = [run_script(script) for script in batch.scripts]
        return await asyncio.gather(*tasks)

    return app


def parse_args():
    """
    Parse command-line arguments for the e2b_router script.

    Arguments:
        --host (str): The hostname or IP address to bind the server to. Defaults to "0.0.0.0" (binds to all interfaces).
        --port (int): The port number on which the server will listen. Defaults to 8000.
        --max_num_sandboxes (int): The maximum number of sandboxes that can be created or managed simultaneously. Defaults to 20.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max_num_sandboxes", type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)

    uvicorn.run(app, host=args.host, port=args.port)
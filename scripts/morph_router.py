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
from typing import Optional, List
from fastapi import FastAPI, Request
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

class BatchRequest(BaseModel):
    """
    BatchRequest is a data model representing a batch processing request.

    Attributes:
        scripts (list[str]): A list of script names or paths to be executed.
        languages (List[str]): The programming languages for each script in the list.
        timeout (int): The maximum allowed execution time for each script in seconds.
        request_timeout (int): The maximum allowed time for the entire batch request in seconds.
    """
    scripts: List[str]
    languages: List[str]
    timeout: int
    request_timeout: int

class ScriptResult(BaseModel):
    """
    ScriptResult is a Pydantic model that represents the result of a script execution.
    Attributes:
        text (Optional[str]): The output text from the script execution.
        exception_str (Optional[str]): An optional string that captures the exception 
            message or details if an error occurred during the script's execution.
        model_config (ConfigDict): A configuration dictionary that allows arbitrary 
            types to be used within the Pydantic model.
    """
    text: Optional[str]
    exception_str: Optional[str]
    
    
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
def create_app(args):
    """
    Creates and configures a FastAPI application instance for the MorphCloud router.
    
    Args:
        args: An object containing configuration parameters for the application.
              - max_num_sandboxes (int): The maximum number of concurrent sandboxes allowed.
              - api_key (str): The MorphCloud API key to use.
    
    Returns:
        FastAPI: A configured FastAPI application instance.
    """
    app = FastAPI()
    
    from morphcloud.api import MorphCloudClient
    from morphcloud.sandbox import Sandbox
    
    app.state.client = MorphCloudClient(api_key=args.api_key)
    app.state.Sandbox = Sandbox

    app.state.sandbox_semaphore = asyncio.Semaphore(args.max_num_sandboxes)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/execute_batch")
    async def execute_batch(batch: BatchRequest, request: Request):
        semaphore = request.app.state.sandbox_semaphore
        client = request.app.state.client
        Sandbox = request.app.state.Sandbox
        
        languages = batch.languages
        timeout = batch.timeout
        request_timeout = batch.request_timeout
        asyncio_timeout = batch.timeout + 1
        
        async def run_script(script: str, language: str) -> ScriptResult:
            sandbox = None
            sandbox_id = "unknown"

            async with semaphore:
                try:
                    sandbox = await asyncio.to_thread(
                        Sandbox.new,
                        client=client,
                        ttl_seconds=timeout
                    )
                    
                    sandbox_id = getattr(sandbox, 'id', None) or getattr(sandbox._instance, 'id', 'unknown')
                    
                    execution = await asyncio.wait_for(
                        asyncio.to_thread(
                            sandbox.run_code,
                            script,
                            language=language,
                            timeout=timeout * 1000  
                        ),
                        timeout=asyncio_timeout,
                    )
                    
                    if hasattr(execution, 'text') and execution.text:
                        return ScriptResult(text=execution.text, exception_str=None)
                    elif hasattr(execution, 'stdout') and execution.stdout:
                        return ScriptResult(text=execution.stdout, exception_str=None)
                    else:
                        return ScriptResult(text="", exception_str="No output from execution")

                except Exception as e:
                    return ScriptResult(text=None, exception_str=str(e))
                
                finally:
                    if sandbox:
                        try:
                            await asyncio.to_thread(sandbox.close)
                            await asyncio.to_thread(sandbox.shutdown)
                        except Exception:
                            pass

        tasks = [run_script(script, lang) for script, lang in zip(batch.scripts, batch.languages)]
        return await asyncio.gather(*tasks)

    return app

def parse_args():
    """
    Parse command-line arguments for the morph_router script.

    Arguments:
        --host (str): The hostname or IP address to bind the server to. Defaults to "0.0.0.0".
        --port (int): The port number on which the server will listen. Defaults to 8001.
        --max_num_sandboxes (int): The maximum number of sandboxes that can be created simultaneously. Defaults to 20.
        --api_key (str): The MorphCloud API key. If not provided, it will be read from the MORPH_API_KEY environment variable.

    Returns:
        argparse.Namespace: Parsed command-line arguments as an object.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--max_num_sandboxes", type=int, default=20)
    parser.add_argument("--api_key", default=os.getenv("MORPH_API_KEY"))
    args = parser.parse_args()
    
    if not args.api_key:
        raise ValueError("MorphCloud API key not provided. Please set MORPH_API_KEY environment variable or use --api_key.")
    
    return args

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)
    
    print(f"Starting MorphCloud Router on {args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)
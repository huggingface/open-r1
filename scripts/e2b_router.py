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
from typing import List, Optional
from fastapi import FastAPI, Request
import argparse
import asyncio
from fastapi import FastAPI
import uvicorn
from e2b_code_interpreter.models import Execution

from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox

load_dotenv()

# Request/response models
    
class BatchRequest(BaseModel):
    scripts: List[str]
    language: str
    timeout: int
    request_timeout: int

class ScriptResult(BaseModel):
    execution: Optional[Execution]
    exception_str: Optional[str]
    
    # required to allow arbitrary types in pydantic models such as Execution
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
def create_app(args):
    app = FastAPI()

    # Instantiate semaphore and attach it to app state
    app.state.sandbox_semaphore = asyncio.Semaphore(args.num_sandboxes)

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
            try:
                async with semaphore:
                    sandbox = await AsyncSandbox.create(
                        timeout=timeout,
                        request_timeout=request_timeout,
                    )
                    execution = await asyncio.wait_for(
                        sandbox.run_code(script, language=language),
                        timeout=asyncio_timeout,
                    )
                    # note that execution.to_json() exists but does not serialize Result.is_main_result
                    return ScriptResult(execution=execution, exception_str=None)
            except Exception as e:
                return ScriptResult(execution=None, exception_str=str(e))
        
            finally:
                try:
                    await sandbox.kill()
                except Exception as e:
                    # do nothing
                    pass

        tasks = [run_script(script) for script in batch.scripts]
        return await asyncio.gather(*tasks)

    return app


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--num_sandboxes", type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)

    uvicorn.run(app, host=args.host, port=args.port)
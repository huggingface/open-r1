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
    result: Optional[Execution]
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
                    return ScriptResult(result=execution, exception_str=None)
            except Exception as e:
                return ScriptResult(result=None, exception_str=str(e))
        
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
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--num_sandboxes", type=int, default=20)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    app = create_app(args)

    uvicorn.run(app, host=args.host, port=args.port)
import argparse
import asyncio
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi import FastAPI, Request
import argparse
import asyncio
from fastapi import FastAPI
import uvicorn
from open_r1.utils import is_e2b_available

if is_e2b_available():
    from dotenv import load_dotenv
    from e2b_code_interpreter import AsyncSandbox

    load_dotenv()
else:
    AsyncSandbox = None


# Request/response models
class ScriptInput(BaseModel):
    code: str
    
class BatchRequest(BaseModel):
    scripts: List[ScriptInput]
    language: str

class ScriptResult(BaseModel):
    result: Optional[float]
    error: Optional[str]
    
def create_app(args):
    app = FastAPI()

    # Instantiate semaphore and attach it to app state
    app.state.args = args
    app.state.sandbox_semaphore = asyncio.Semaphore(args.num_sandboxes)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/execute_batch")
    async def execute_batch(batch: BatchRequest, request: Request):
        semaphore = request.app.state.sandbox_semaphore
        args = request.app.state.args
        language = batch.language

        async def run_script(script: ScriptInput) -> ScriptResult:
            try:
                async with semaphore:
                    sandbox = await AsyncSandbox.create(
                        timeout=args.sandbox_timeout,
                        request_timeout=args.request_timeout,
                    )
                    execution = await asyncio.wait_for(
                        sandbox.run_code(script.code, language=language),
                        timeout=args.asyncio_timeout,
                    )
                    return ScriptResult(result=float(execution.text), error=None)
            except Exception as e:
                return ScriptResult(result=None, error=str(e))
        
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
    parser.add_argument("--sandbox_timeout", type=int, default=10)
    parser.add_argument("--request_timeout", type=int, default=10)
    parser.add_argument("--asyncio_timeout", type=int, default=15)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    config = {
        "num_sandboxes": args.num_sandboxes,
        "sandbox_timeout": args.sandbox_timeout,
        "request_timeout": args.request_timeout,
        "asyncio_timeout": args.asyncio_timeout
    }
    app = create_app(args)

    uvicorn.run(app, host=args.host, port=args.port)
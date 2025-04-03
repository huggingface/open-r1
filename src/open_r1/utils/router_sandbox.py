from typing import Optional
import json

from e2b_code_interpreter.models import Execution, ExecutionError, Result
from e2b_code_interpreter import Sandbox
import requests


from e2b_code_interpreter.models import parse_output
class BatchedRoutedSandbox:
    """
    A Sandbox that is routed to E2B via the E2B Router. see scripts/e2b_router.py
    Usage should match 'from e2b_code_interpreter import Sandbox' , but is batched by default
    """
    def __init__(self, router_url: str):
        self.router_url = router_url
        
    def run_code(
        self,
        scripts: list[str],
        language: str = "python",
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> list[Execution]:
        
        # Defaults are the same as E2B's codebase
        if timeout is None:
            timeout = 300  # 5 minutes
        if request_timeout is None:
            request_timeout = 30
            
        payload = {
            "scripts": scripts, 
            "language": language,
            "timeout": timeout,
            "request_timeout": request_timeout,      
            }
        response = requests.post(f"http://{self.router_url}/execute_batch", json=payload)
        if not response.ok:
            print(f"Request failed: {response.status_code}")
        
        results = response.json()
        output = []
        for result in results:
            execution = Execution(
                    results=[Result(**r) for r in result["execution"]["results"]],
                    logs=result["execution"]["logs"],
                    error=ExecutionError(**result["execution"]["error"]) if result["execution"]["error"] else None,
                    execution_count=result["execution"]["execution_count"],
            )
            output.append(execution)
            
        return output
            
        
if __name__ == "__main__":
    sbx = BatchedRoutedSandbox(router_url="0.0.0.0:8001")
    codes = ["print('hello world')", "print('hello world)"]
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)
from typing import Optional

from e2b_code_interpreter.models import Execution, ExecutionError
from e2b_code_interpreter import Sandbox
import requests


class BatchedRoutedSandbox:
    """
    A Sandbox that is routed to E2B via the E2B Router. see scripts/e2b_router.py
    Usage should match 'from e2b_code_interpreter import Sandbox' , but is batched by default
    """
    def __init__(self, router_url: str):
        
        if router_url is None:
            raise ValueError("sandbox_url must be provided")
        self.router_url = router_url
        
    def run_code(
        self,
        scripts: list[str],
        language: str = "python",
        timeout: Optional[int] = None,
        request_timeout: Optional[int] = None,
    ) -> list[Execution]:
        
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
        results = response.json()
        if not response.ok:
            print(f"Request failed: {response.status_code}")
        
        output = []
        for result in results:
            execution =  Execution(**result["result"])
            if result["result"]["error"] is not None:
                error = ExecutionError(**result["result"]["error"])
                execution.error = error
            
            output.append(execution)
            
        return output
            
        
if __name__ == "__main__":
    sbx = Sandbox()  # By default the sandbox is alive for 5 minutes
    sbx = BatchedRoutedSandbox(router_url="0.0.0.0:8001")
    codes = ["print('hello world')", "print('hello world)"]
    executions = sbx.run_code(codes)  # Execute Python inside the sandbox

    print(executions)
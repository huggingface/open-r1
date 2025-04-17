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

"""Code execution providers for executing and evaluating code snippets."""

import abc
import asyncio
from typing import List, Optional

from ..utils import is_e2b_available

if is_e2b_available():
    from e2b_code_interpreter import AsyncSandbox
    from e2b_code_interpreter.models import Execution
    from .routed_sandbox import RoutedSandbox
else:
    AsyncSandbox = None
    Execution = None
    RoutedSandbox = None


class CodeExecutionProvider(abc.ABC):
    """Abstract base class for code execution providers."""
    
    @abc.abstractmethod
    def execute_scripts(self, scripts: List[str], language: str = "python") -> List[float]:
        """Execute multiple scripts and return their reward values.
        
        Args:
            scripts: List of code scripts to execute
            language: The programming language of the scripts
            
        Returns:
            List of float rewards (one per script)
        """
        pass


class E2BProvider(CodeExecutionProvider):
    """Provider that executes code using E2B sandboxes."""
    
    def __init__(self, num_parallel: int = 2, e2b_router_url: Optional[str] = None):
        """Initialize the E2B provider.
        
        Args:
            num_parallel: Number of parallel sandboxes to use
            e2b_router_url: URL for the E2B router (if using router mode)
        """
        if not is_e2b_available():
            raise ImportError(
                "E2B is not available and required for this provider. Please install E2B with "
                "`pip install e2b-code-interpreter` and add an API key to a `.env` file."
            )
        
        self.num_parallel = num_parallel
        self.e2b_router_url = e2b_router_url
    
    def execute_scripts(self, scripts: List[str], language: str = "python") -> List[float]:
        """Execute scripts using E2B sandboxes.
        
        If e2b_router_url is provided, uses the RoutedSandbox for batch processing.
        Otherwise, uses direct AsyncSandbox with parallelization.
        """
        if self.e2b_router_url is not None:
            routed_sandbox = RoutedSandbox(router_url=self.e2b_router_url)
            
            executions = routed_sandbox.run_code(
                scripts=scripts,
                language=language,
                timeout=30,
                request_timeout=28,
            )
            
            rewards = []
            for execution in executions:
                try:
                    reward = float(execution.text)
                    rewards.append(reward)
                except Exception:
                    rewards.append(None)
            return rewards
        
        try:
            rewards = self._run_async_from_sync(scripts, language, self.num_parallel)
        except Exception as e:
            print(f"Error from E2B executor: {e}")
            rewards = [0.0] * len(scripts)
        
        return rewards
    
    def _run_async_from_sync(self, scripts: List[str], language: str, num_parallel: int) -> List[float]:
        """Function wrapping the `_run_async` function."""
        try:
            # Run the async function and get the result
            rewards = asyncio.run(self._run_async(scripts, language, num_parallel))
        except Exception as e:
            print(f"Error from E2B executor async: {e}")
            raise e
        
        return rewards
    
    async def _run_async(self, scripts: List[str], language: str, num_parallel: int) -> List[float]:
        # Limit the number of concurrent tasks
        semaphore = asyncio.Semaphore(num_parallel)
        
        # Create a list of tasks for running scripts concurrently
        tasks = [self._run_script(script, language, semaphore) for script in scripts]
        
        # Wait for all tasks to complete and gather their results as they finish
        results = await asyncio.gather(*tasks)
        rewards = list(results)  # collect results
        
        return rewards
    
    async def _run_script(self, script: str, language: str, semaphore: asyncio.Semaphore) -> float:
        # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
        # These values are based on running 256 examples with the gold solution
        # from open-r1/verifiable-coding-problems-python_decontaminated
        # see scripts/benchmark_e2b.py
        
        SANDBOX_TIMEOUT = 30
        MARGIN = 2
        REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN
        
        async with semaphore:
            try:
                sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
                execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
                return float(execution.text)
            except (TypeError, ValueError):
                return 0.0
            except asyncio.TimeoutError:
                print("Operation timed out")
                return 0.0
            except Exception as e:
                print(f"Error in `_run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
                return 0.0
            finally:
                try:
                    await sandbox.kill()
                except Exception as e:
                    print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")


class LocalProvider(CodeExecutionProvider):
    """Provider that executes code locally using Python's multiprocessing.
    
    This provider executes Python code in separate processes for isolation.
    """
    
    def __init__(self, num_parallel: int = 2):
        """Initialize the local provider.
        
        Args:
            num_parallel: Number of parallel processes to use
        """
        self.num_parallel = num_parallel
        
        # Import here to avoid circular imports
        import multiprocessing
        self.multiprocessing = multiprocessing
    
    def execute_scripts(self, scripts: List[str], language: str = "python") -> List[float]:
        """Execute scripts locally using Python's multiprocessing.
        
        Args:
            scripts: List of Python scripts to execute
            language: Must be "python" for this provider
            
        Returns:
            List of float rewards (one per script)
        """
        if language != "python":
            raise ValueError(f"LocalProvider only supports Python, got {language}")
        
        # Create a pool with the specified number of processes
        with self.multiprocessing.Pool(processes=self.num_parallel) as pool:
            # Execute each script in the pool and collect results
            results = pool.map(self._execute_script, scripts)
        
        return results
    
    def _execute_script(self, script: str) -> float:
        """Execute a single script in a separate process.
        
        Args:
            script: Python script to execute
            
        Returns:
            Float reward from the script execution
        """
        import subprocess
        import tempfile
        
        # Create a temporary file to hold the script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
            temp_file_name = temp_file.name
            temp_file.write(script)
        
        try:
            # Execute the script and capture the output
            process = subprocess.run(
                ["python3", temp_file_name],
                text=True,
                capture_output=True,
                timeout=30  # Same timeout as E2B
            )
            
            # Check if execution was successful
            if process.returncode == 0:
                # Try to parse the final line as a float
                output_lines = process.stdout.strip().split('\n')
                try:
                    # The evaluation script should print the reward as the last line
                    if output_lines:
                        reward = float(output_lines[-1])
                        return reward
                    else:
                        print("Script execution produced no output")
                        return 0.0
                except ValueError:
                    print(f"Failed to parse reward from output: {output_lines[-1] if output_lines else None}")
                    return 0.0
            else:
                print(f"Script execution failed with code {process.returncode}: {process.stderr}")
                return 0.0
                
        except subprocess.TimeoutExpired:
            print("Script execution timed out")
            return 0.0
        except Exception as e:
            print(f"Error executing script: {e}")
            return 0.0
        finally:
            # Clean up temporary file
            import os
            try:
                os.unlink(temp_file_name)
            except Exception:
                pass


class MorphProvider(CodeExecutionProvider):
    """Provider that executes code using MorphCloud's Sandbox API."""
    
    def __init__(self, num_parallel: int = 2, morph_router_url: Optional[str] = None):
        """Initialize the Morph provider.
        
        Args:
            num_parallel: Number of parallel executions to use
            morph_router_url: URL for the MorphCloud router (if using router mode)
        """
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            print("Warning: python-dotenv not installed. Environment variables must be set directly.")
            
        self.num_parallel = num_parallel
        self.morph_router_url = morph_router_url
        
        # If router URL is provided, use RoutedMorphSandbox instead of direct Sandbox
        if self.morph_router_url is not None:
            from .routed_morph import RoutedMorphSandbox
            self.routed_sandbox = RoutedMorphSandbox(router_url=self.morph_router_url)
            return
        
        # Get API key from environment variables
        import os
        self.api_key = os.getenv("MORPH_API_KEY")
        if not self.api_key:
            raise ValueError(
                "MorphCloud API key not found. Please set the MORPH_API_KEY environment variable."
            )
        
        # Import the MorphCloud client and Sandbox
        try:
            from morphcloud.api import MorphCloudClient
            from morphcloud.sandbox import Sandbox
            
            self.client = MorphCloudClient(api_key=self.api_key)
            self.Sandbox = Sandbox
        except ImportError as e:
            raise ImportError(f"Required MorphCloud dependencies not installed: {e}")
    
    def execute_scripts(self, scripts: List[str], language: str = "python") -> List[float]:
        """Execute scripts using MorphCloud Sandbox API.
        
        Args:
            scripts: List of Python scripts to execute
            language: Programming language
            
        Returns:
            List of float rewards (one per script)
        """
        # If we have a router URL, use the routed sandbox
        if hasattr(self, 'routed_sandbox'):
            print(f"MorphProvider: Using routed sandbox at {self.morph_router_url}")
            try:
                # Execute scripts via the router
                results = self.routed_sandbox.run_code(
                    scripts=scripts,
                    language=language,
                    timeout=30,
                    request_timeout=28,
                )
                
                # Parse rewards from the results
                rewards = []
                for result in results:
                    try:
                        reward = float(result.text)
                        rewards.append(reward)
                    except (ValueError, AttributeError):
                        rewards.append(0.0)
                return rewards
            except Exception as e:
                print(f"Error from MorphCloud router: {e}")
                return [0.0] * len(scripts)
        
        # Otherwise, use direct sandbox execution
        import asyncio
        import time
        
        print(f"MorphProvider: Starting execution of {len(scripts)} scripts with parallelism={self.num_parallel}")
        start_time = time.time()
        
        # Create a new event loop and run the async function
        try:
            # Mirror E2B's approach using asyncio
            rewards = asyncio.run(self._run_async(scripts, language, self.num_parallel))
            elapsed = time.time() - start_time
            print(f"MorphProvider: Completed {len(scripts)} scripts in {elapsed:.2f}s ({len(scripts)/elapsed:.2f} scripts/sec)")
        except Exception as e:
            print(f"Error from MorphCloud executor: {e}")
            rewards = [0.0] * len(scripts)
        
        return rewards
        
    async def _run_async(self, scripts: List[str], language: str, num_parallel: int) -> List[float]:
        """Run multiple scripts concurrently with limited parallelism.
        
        Args:
            scripts: List of scripts to execute
            language: Programming language
            num_parallel: Maximum number of concurrent executions
            
        Returns:
            List of rewards
        """
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(num_parallel)
        
        # Create tasks for each script
        tasks = [self._run_script(script, language, semaphore) for script in scripts]
        
        # Execute all tasks and gather results
        results = await asyncio.gather(*tasks)
        
        return list(results)
    
    async def _run_script(self, script: str, language: str, semaphore: asyncio.Semaphore) -> float:
        """Execute a single script in a MorphCloud Sandbox.
        
        Args:
            script: The script to execute
            language: Programming language
            semaphore: Semaphore to limit concurrency
            
        Returns:
            Float reward from script execution
        """
        # Set timeouts similar to E2B
        SANDBOX_TIMEOUT = 30
        MARGIN = 2
        ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN
        
        sandbox = None
        sandbox_id = None
        async with semaphore:
            try:
                # Create a new sandbox (run in a thread to not block)
                print(f"MorphProvider: Creating new sandbox for script of length {len(script)}")
                sandbox = await asyncio.to_thread(
                    self.Sandbox.new,
                    client=self.client,
                    ttl_seconds=SANDBOX_TIMEOUT
                )
                sandbox_id = getattr(sandbox, 'id', None) or getattr(sandbox._instance, 'id', 'unknown')
                print(f"MorphProvider: Created sandbox {sandbox_id[:8]}...")
                
                # Execute the script (run in a thread to not block)
                print(f"MorphProvider: Executing {language} script in sandbox {sandbox_id[:8]}...")
                # Display the first 150 characters of the script for debugging
                print(f"MorphProvider: Script snippet: {script[:150]}...")
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        sandbox.run_code,
                        script,
                        language=language,
                        timeout=SANDBOX_TIMEOUT
                    ),
                    timeout=ASYNCIO_TIMEOUT
                )
                
                # Parse the reward from the result, using text property like E2B does
                reward = 0.0
                try:
                    if hasattr(result, 'text') and result.text:
                        # Parse the text property directly (same as E2B provider)
                        reward = float(result.text)
                        print(f"MorphProvider: Sandbox {sandbox_id[:8]}... returned reward: {reward}")
                    elif result.stdout:
                        # Fallback to stdout if text is not available
                        lines = result.stdout.strip().split('\n')
                        if lines:
                            reward = float(lines[-1])
                            print(f"MorphProvider: Sandbox {sandbox_id[:8]}... returned reward from stdout: {reward}")
                except (ValueError, AttributeError) as e:
                    print(f"MorphProvider: Sandbox {sandbox_id[:8]}... failed to parse reward: {e}")
                    if hasattr(result, 'text') and result.text:
                        print(f"MorphProvider: Text output was: {result.text}")
                    elif result.stdout:
                        print(f"MorphProvider: Stdout was: {result.stdout.strip()}")
                
                return reward
                
            except asyncio.TimeoutError:
                print(f"MorphProvider: Sandbox {sandbox_id[:8] if sandbox_id else 'unknown'} operation timed out")
                return 0.0
            except Exception as e:
                print(f"MorphProvider: Error in sandbox {sandbox_id[:8] if sandbox_id else 'unknown'}: {e}")
                return 0.0
            finally:
                # Clean up the sandbox
                if sandbox:
                    try:
                        print(f"MorphProvider: Cleaning up sandbox {sandbox_id[:8] if sandbox_id else 'unknown'}")
                        await asyncio.to_thread(sandbox.close)
                        await asyncio.to_thread(sandbox.shutdown)
                    except Exception as e:
                        print(f"MorphProvider: Error cleaning up sandbox {sandbox_id[:8] if sandbox_id else 'unknown'}: {e}")


def get_provider(provider_type: str = "e2b", **kwargs) -> CodeExecutionProvider:
    """Factory function to get the appropriate code execution provider.
    
    Args:
        provider_type: Type of provider to use ("e2b", "local", "morph")
        **kwargs: Additional arguments to pass to the provider
    
    Returns:
        An instance of CodeExecutionProvider
    """
    if provider_type == "e2b":
        return E2BProvider(
            num_parallel=kwargs.get("num_parallel", 2),
            e2b_router_url=kwargs.get("e2b_router_url", None),
        )
    elif provider_type == "local":
        return LocalProvider(
            num_parallel=kwargs.get("num_parallel", 2),
        )
    elif provider_type == "morph":
        return MorphProvider(
            num_parallel=kwargs.get("num_parallel", 2),
            morph_router_url=kwargs.get("morph_router_url", None)
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")

import asyncio
import json
import logging
import os
import tempfile
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from open_r1.utils.import_utils import is_morph_available


# Replace direct imports with conditional imports
if is_morph_available():
    from morphcloud.api import Instance, InstanceExecResponse, MorphCloudClient
else:
    Instance = None
    InstanceExecResponse = None
    MorphCloudClient = None


# Silence verbose logs from dependencies
logging.getLogger("paramiko").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class MorphCloudError(Exception):
    pass


class MorphCloudExecutionClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        spans_log_path: Optional[str] = None,
    ):
        """
        Initialize the MorphCloud execution client.

        Args:
            api_key: Optional API key for MorphCloud. If not provided, will use MORPH_API_KEY env var.
            base_url: Optional base URL for MorphCloud API. If not provided, will use default.
            spans_log_path: Path to log API call spans to. Defaults to 'logs/morph_api_spans.jsonl'.
        """

        self.client = MorphCloudClient(api_key=api_key, base_url=base_url)
        self._snapshot_lock = asyncio.Lock()

    async def _prepare_instance(self, snapshot_id=None) -> Instance:
        """
        Prepare and start a MorphCloud instance.

        Args:
          snapshot_id: Optional snapshot ID to use. If None, will get or create base snapshot.

        Returns:
          Instance: The ready-to-use MorphCloud instance

        Raises:
          TimeoutError: If instance fails to start or become ready
        """

        if not snapshot_id:
            snapshot = await self._get_or_create_base_snapshot()
            snapshot_id = snapshot.id

        try:
            instance = await self.client.instances.astart(
                snapshot_id, ttl_seconds=600
            )  # Auto-terminate after 10 minutes
            await instance.await_until_ready(timeout=300)
            return instance
        except asyncio.TimeoutError as e:
            print(f"Timeout while preparing instance: {str(e)}")
            if instance:
                try:
                    await instance.astop()
                except Exception:
                    pass
            raise

    async def _prepare_files(self, data: Dict[str, Any], temp_dir: str) -> Tuple[str, Dict[str, Any], Dict[str, str]]:
        """
        Process files, determine problem ID, and prepare configuration.

        Args:
            data: Dictionary containing file information
            temp_dir: Local temporary directory for file operations

        Returns:
            tuple: (problem_id, grader_config, local_files)

        Raises:
            ValueError: If problem ID cannot be determined
        """
        # Extract problem ID
        problem_id = None
        graders_files = []
        for file in data["files"]:
            if file["name"].startswith("graders/") and file["name"].endswith(".cpp"):
                potential_id = os.path.basename(file["name"]).split(".")[0]
                if potential_id not in ["grader", "manager", "stub"]:
                    problem_id = potential_id

            if file["name"].startswith("graders/"):
                graders_files.append(file)

        if not problem_id:
            raise ValueError("Could not determine problem ID from files")

        grader_config = {
            "task_type": "Batch",
            "code": problem_id,
            "time_limit": data["run_timeout"] / 1000,
            "memory_limit": data["run_memory_limit"] * 1024 * 1024,
        }

        for file in graders_files:
            if "manager.cpp" in file["name"]:
                grader_config["task_type"] = "Communication"
                grader_config["task_type_parameters_Communication_num_processes"] = 1
                grader_config["task_type_parameters_Communication_user_io"] = "std_io"
                break

        config_path = os.path.join(temp_dir, "grader_config.json")
        with open(config_path, "w") as f:
            json.dump(grader_config, f)

        local_files = {"grader_config.json": config_path}

        for file in data["files"]:
            local_path = os.path.join(temp_dir, os.path.basename(file["name"]))
            with open(local_path, "w") as f:
                f.write(file["content"])
            local_files[file["name"]] = local_path

        return problem_id, grader_config, local_files

    async def _upload_files(self, instance: Instance, local_files: Dict[str, str]) -> bool:
        """
        Upload all necessary files to the instance.

        Args:
            instance: The MorphCloud instance
            local_files: Dictionary mapping remote paths to local file paths

        Returns:
            bool: True if all uploads were successful

        Raises:
            TimeoutError: If uploads time out
        """
        for remote_name, local_path in local_files.items():
            target_path = f"/workspace/{remote_name}"
            dir_path = os.path.dirname(target_path)

            if dir_path != "/workspace":
                await instance.aexec(f"mkdir -p {dir_path}")

            await instance.aupload(local_path, target_path)

        await instance.aupload(local_files["grader_config.json"], "/workspace/graders/grader_config.json")

        return True

    async def _compile_code(self, instance: Instance) -> InstanceExecResponse:
        """
        Compile the code on the instance.

        Args:
            instance: The MorphCloud instance

        Returns:
            InstanceExecResponse: Result of compilation

        Raises:
            RuntimeError: If compilation fails
        """
        compile_result = await instance.aexec("cd /workspace && ./compile")

        if compile_result.exit_code != 0:
            raise RuntimeError(f"Compilation error exit code {compile_result.exit_code}\n{compile_result.stderr}")

        return compile_result

    async def _run_tests(self, instance: Instance, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Run tests and evaluate results.

        Args:
            instance: The MorphCloud instance
            data: Dictionary containing runtime parameters

        Returns:
            tuple: (score, feedback)

        Raises:
            TimeoutError: If test execution times out
        """
        hard_timeout = data["run_timeout"] / 1000 + 3
        run_command = f"cd /workspace && timeout {hard_timeout}s ./run"

        run_result = await instance.aexec(run_command)

        if run_result.exit_code == 124 or run_result.exit_code == 137 or run_result.exit_code == 143:
            return "0", "Time limit exceeded"

        if run_result.exit_code != 0 and "Memory limit exceeded" in run_result.stderr:
            return "0", "Memory limit exceeded"

        if run_result.stdout:
            return run_result.stdout.strip(), run_result.stderr.strip()

        if run_result.exit_code != 0:
            return (
                "0",
                f"Runtime error with exit code {run_result.exit_code}\n{run_result.stderr}",
            )

        return "0", "Unknown error"

    async def _execute_with_instance(self, instance: Instance, data: Dict[str, Any], temp_dir: str) -> Tuple[str, str]:
        """Execute code using a prepared instance.

        Args:
            instance: Ready MorphCloud instance
            data: Execution data
            temp_dir: Temporary directory for file operations

        Returns:
            Tuple of (score, feedback)

        Raises:
            Exception: Passes through exceptions for retry handling
        """
        await instance.await_until_ready(timeout=300)

        problem_id, grader_config, local_files = await self._prepare_files(data, temp_dir)

        await self._upload_files(instance, local_files)

        try:
            await self._compile_code(instance)
        except RuntimeError as e:
            return "0", str(e)

        score, feedback = await self._run_tests(instance, data)
        return score, feedback

    async def _execute(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Internal implementation of execute with no retry logic.

        Args:
            data: Dictionary containing execution data

        Returns:
            Tuple of (score, feedback)

        Raises:
            Exception: If execution fails
        """
        instance = None

        # Set timeouts to ensure we don't block indefinitely
        # INSTANCE_TIMEOUT = 300  # 5 minutes for instance operations
        TOTAL_EXECUTION_TIMEOUT = 600  # 10 minutes total execution time

        with tempfile.TemporaryDirectory(prefix="morph_exec_") as temp_dir:
            snapshot = await self._get_or_create_base_snapshot()
            instance = await self.client.instances.astart(
                snapshot.id, ttl_seconds=600
            )  # Auto-terminate after 10 minutes

            async with instance:
                # Use asyncio.wait_for to add overall timeout to the execution process
                return await asyncio.wait_for(
                    self._execute_with_instance(instance, data, temp_dir),
                    timeout=TOTAL_EXECUTION_TIMEOUT,
                )

    async def execute(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Execute code on MorphCloud based on the provided data with enhanced debugging and recovery.

        Orchestrates the following steps with proper error handling and retries:
        1. Prepare an instance (with retry)
        2. Set up workspace (with retry)
        3. Prepare and upload files (with retry)
        4. Compile code (with retry)
        5. Run tests (with retry)

        Args:
            data: Dictionary containing:
                - files: List of file objects with name and content fields
                - run_timeout: Timeout in milliseconds
                - run_memory_limit: Memory limit in MB

        Returns:
            Tuple of (score, feedback) where:
                - score is a string representation of a float between 0.0 and 1.0
                - feedback is a string with execution details
        """
        # TODO: would be faster to pass info about the subtask as well to create a snapshot per subtask
        # would cache the uploads of all files other than the submission: input.txt, correct_output.txt, grader files
        # rather than reusing the snapshot that only has the compile/run scripts on it
        # currently, run_submission -> client.execute(data) does not easily pass subtask info

        # Retry configuration
        max_retries = 4
        base_delay = 1.0

        # Try execution with retries and exponential backoff
        for attempt in range(max_retries + 1):
            try:
                return await self._execute(data)

            except asyncio.TimeoutError:
                if attempt < max_retries:
                    print(f"Execution timed out, retrying ({attempt + 1}/{max_retries})")
                else:
                    return "0", "Execution timed out after multiple retries"

            except Exception as e:
                # Calculate exponential backoff
                if attempt < max_retries:
                    retry_delay = min(base_delay * (2**attempt), 30)  # Exponential backoff, capped at 30 seconds

                    print(
                        f"Execution failed with {type(e).__name__}: {str(e)}, retrying in {retry_delay:.2f}s ({attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(retry_delay)
                else:
                    print(f"Execution failed after {max_retries} retries: {type(e).__name__}: {str(e)}")
                    return "0", f"Execution failed after multiple retries: {str(e)}"

    async def _get_or_create_base_snapshot(self):
        """Get or create a snapshot with the necessary dependencies and scripts for evaluation."""

        async with self._snapshot_lock:
            base_snapshots = await self.client.snapshots.alist(digest="ioi-evaluation-morph")

            if not base_snapshots:
                print("Creating base snapshot with build-essential cmake and g++")

                # Create base snapshot with minimal specs
                base_snapshot = await self.client.snapshots.acreate(
                    vcpus=2,
                    memory=4096,
                    disk_size=10240,
                    metadata={"purpose": "ioi_evaluation"},
                )

                # Start a temporary instance from the base snapshot
                temp_instance = await self.client.instances.astart(
                    base_snapshot.id, ttl_seconds=900
                )  # Auto-terminate after 15 minutes

                try:
                    # Wait for the instance to be ready
                    await temp_instance.await_until_ready(timeout=300)

                    # Get script contents
                    compile_script = await self._get_compile_script()
                    run_script = await self._get_run_script()

                    # Use temporary directory to store scripts
                    with tempfile.TemporaryDirectory(prefix="morph_setup_") as temp_dir:
                        # Create paths for script files
                        compile_path = os.path.join(temp_dir, "compile.sh")
                        run_path = os.path.join(temp_dir, "run.sh")

                        # Write scripts to temp files
                        with open(compile_path, "w") as f:
                            f.write(compile_script)

                        with open(run_path, "w") as f:
                            f.write(run_script)

                        async with temp_instance:
                            # Install dependencies
                            await temp_instance.aexec("apt-get update && apt-get install -y build-essential cmake g++")

                            # Create workspace directory
                            await temp_instance.aexec(
                                "mkdir -p /workspace && mkdir -p /workspace/graders && chmod 777 /workspace"
                            )

                            # Upload scripts to instance
                            await temp_instance.aupload(compile_path, "/workspace/compile")
                            await temp_instance.aupload(run_path, "/workspace/run")

                            # Make scripts executable
                            await temp_instance.aexec("chmod +x /workspace/compile /workspace/run")

                            # Create snapshot from the prepared instance
                            final_snapshot = await temp_instance.asnapshot(digest="ioi-evaluation-morph")

                except Exception as e:
                    # Ensure instance is stopped if anything fails
                    await temp_instance.astop()
                    raise e
            else:
                final_snapshot = base_snapshots[0]

            return final_snapshot

    async def _get_compile_script(self):
        """Get the compile script content."""
        return """#!/bin/bash

manager_files=()  # Array to store manager filenames
current_dir="$(pwd)"

# Checker compilation path
checker_dir="$current_dir/checker"
checker_src="$checker_dir/checker.cpp"

if [ -e "$checker_src" ]; then
    echo "Compiling checker"
    checker_exe="$checker_dir/checker"
    g++ -x c++ -std=gnu++17 -O2 -o "$checker_exe" "$checker_src"
    chmod +x "$checker_exe"
    if [ $? -ne 0 ]; then
        echo "Could not compile checker" >&2
        exit 1
    fi
    echo "Compiled checker"
else
    echo "No checker found at $checker_src"
fi

# Graders path
graders_dir="$current_dir/graders"
if [ ! -e "$graders_dir" ]; then
    echo "Grader folder was not found" >&2
    exit 1
fi

# Find and compile manager if it exists
manager_src="$graders_dir/manager.cpp"
if [ -e "$manager_src" ]; then
    echo "Compiling manager"
    manager_exe="$graders_dir/manager"
    g++ -x c++ -std=gnu++17 -O2 -o "$manager_exe" "$manager_src"
    chmod +x "$manager_exe"
    if [ $? -ne 0 ]; then
        echo "Could not compile manager" >&2
        exit 1
    fi
    manager_files+=("manager")
fi

# Process other graders
graders_list=($(ls "$graders_dir" | grep -v 'manager.cpp'))
for grader_name in "${graders_list[@]}"; do
    manager_files+=("$grader_name")
done

# Extract problem name and compile necessary files
problem_name='?'
for file in "${manager_files[@]}"; do
    if [[ "$file" == *.h && "$file" != "testlib.h" ]]; then
        problem_name="${file%.h}"
        echo "Problem name: $problem_name"
        break
    fi
done

files_to_compile=("graders/$problem_name.cpp")
[ -e graders/grader.cpp ] && files_to_compile+=("graders/grader.cpp")
[ -e graders/stub.cpp ] && files_to_compile+=("graders/stub.cpp")

g++ -DEVAL -std=gnu++17 -O2 -pipe -s -o graders/"$problem_name" "${files_to_compile[@]}"
if [ $? -ne 0 ]; then
    echo "Failed to compile $problem_name" >&2
    exit 1
fi
chmod +x graders/"$problem_name"
echo "Compiled $problem_name from ${files_to_compile[@]} successfully"

echo "Manager files: ${manager_files[@]}"
"""

    async def _get_run_script(self):
        """Get the run script content."""
        return """#!/usr/bin/env bash
# disable stack limit so you don't get RE with recursion
ulimit -s unlimited
# some problems have 10MB+ input/output files in their test cases and you might get RE. uncomment if needed
# ulimit -f 2097152

# Check if grader_config.json exists
if [ ! -f "graders/grader_config.json" ]; then
    echo "Error: graders/grader_config.json not found" >&2
    echo "Current directory contents:" >&2
    find . -type f -o -type d | sed -e 's/[^-][^\/]*\//  |/g' -e 's/|\([^ ]\)/|-\1/' >&2
    exit 1
fi

# Read task type, code, and time limit from grader_config.json using grep and sed
TASK_TYPE=$(grep -o '"task_type":[^,}]*' graders/grader_config.json | sed 's/"task_type":\\s*"\\([^"]*\\)"/\\1/')
TASK_NAME=$(grep -o '"code":[^,}]*' graders/grader_config.json | sed 's/"code":\\s*"\\([^"]*\\)"/\\1/')
TIME_LIMIT=$(grep -o '"time_limit":[^,}]*' graders/grader_config.json | sed 's/"time_limit":\\s*\\([^,}]*\\)/\\1/')
MEMORY_LIMIT=$(grep -o '"memory_limit":[^,}]*' graders/grader_config.json | sed 's/"memory_limit":\\s*\\([^,}]*\\)/\\1/')
TASK_EXECUTABLE="graders/$TASK_NAME"

# Set memory limit in KB (convert from bytes)
MEMORY_LIMIT_KB=0
if [ -n "$MEMORY_LIMIT" ]; then
    MEMORY_LIMIT_KB=$(($MEMORY_LIMIT / 1024))
    # Set the memory limit for the entire script and all child processes
    ulimit -v $MEMORY_LIMIT_KB
fi

# "Securely" handle the correct output file
CORRECT_OUTPUT=""
if [ -f "correct_output.txt" ]; then
    # Read the content and immediately remove the file
    CORRECT_OUTPUT=$(cat correct_output.txt)
    rm -f correct_output.txt
fi

# Create a temporary file for solution output
SOLUTION_OUTPUT=$(mktemp)

# Global variables for process tracking
declare -a ALL_PIDS
declare -a FIFO_DIRS

# Define cleanup function - simplified assuming timeout exists
function cleanup {
    # Kill all tracked processes silently
    exec 2>/dev/null
    for pid in "${ALL_PIDS[@]:-}"; do
        kill -9 "$pid" 2>/dev/null || true
    done

    # Clean up FIFO directories
    for dir in "${FIFO_DIRS[@]:-}"; do
        [ -d "$dir" ] && rm -rf "$dir"
    done

    # Clean up temporary files
    rm -f "$SOLUTION_OUTPUT" || true
    exec 2>&2
}

# Set up signal handling
trap cleanup EXIT INT TERM

# Function to handle exit codes consistently across task types
function handle_exit_code {
    local exit_code=$1

    # Check for known timeout exit codes:
    # - 124: standard timeout exit code
    # - 137: SIGKILL (128+9), used for hard timeouts
    # - 143: SIGTERM (128+15), can also be used for timeouts
    if [ $exit_code -eq 124 ] || [ $exit_code -eq 137 ] || [ $exit_code -eq 143 ]; then
        echo "0"
        echo "Time limit exceeded (${TIME_LIMIT}s)" >&2
        return 124
    # All other non-zero exit codes should be treated as runtime errors
    elif [ $exit_code -ne 0 ]; then
        echo "0"
        echo "Runtime error with exit code $exit_code" >&2
        return $exit_code
    fi

    # Success case - return 0
    return 0
}

# Function to run a command with timeout (simplified assuming timeout exists)
function run_with_timeout {
    local soft_limit=$1; shift
    local command_to_run="$@"

    timeout --preserve-status "$soft_limit" "$@"
    return $?
}

case "$TASK_TYPE" in
    "Batch")
        # Simple batch execution with timeout
        run_with_timeout "$TIME_LIMIT" ./$TASK_EXECUTABLE < input.txt > "$SOLUTION_OUTPUT"
        exit_code=$?

        # Handle non-zero exit codes
        handle_exit_code $exit_code
        if [ $? -ne 0 ]; then
            exit $?
        fi

        # Check the output if we have a correct output
        if [ -n "$CORRECT_OUTPUT" ]; then
            # Restore the correct output file
            echo "$CORRECT_OUTPUT" > correct_output.txt

            # Check if there's a custom checker
            if [ -f "checker/checker" ]; then
                # Let the checker handle everything
                ./checker/checker input.txt correct_output.txt "$SOLUTION_OUTPUT"
                exit $?
            else
                # Simple diff-based checking
                if diff -bq <(echo "$CORRECT_OUTPUT") "$SOLUTION_OUTPUT" >/dev/null; then
                    echo "1"
                    echo "Output is correct (diff)" >&2
                else
                    echo "0"
                    echo "Output isn't correct (diff)" >&2
                    exit 0
                fi
            fi
        else
            # If no correct output was provided, just output the solution's output
            cat "$SOLUTION_OUTPUT"
        fi
        ;;

    "Communication")
        # Read Communication-specific parameters
        NUM_PROCESSES=$(grep -o '"task_type_parameters_Communication_num_processes":[^,}]*' graders/grader_config.json | sed 's/.*:\\s*\\([0-9]*\\)/\\1/' || true)
        if [ -z "$NUM_PROCESSES" ]; then
            NUM_PROCESSES=1
        fi
        USER_IO=$(grep -o '"task_type_parameters_Communication_user_io":[^,}]*' graders/grader_config.json | sed 's/.*:\\s*"\\([^"]*\\)"/\\1/' || echo "std_io")

        # Read custom manager arguments if they exist
        MANAGER_CUSTOM_ARGS=""
        if grep -q '"task_type_parameters_Communication_manager_args"' graders/grader_config.json; then
            MANAGER_CUSTOM_ARGS=$(grep -o '"task_type_parameters_Communication_manager_args":[^,}]*' graders/grader_config.json | sed 's/.*:\\s*"\\([^"]*\\)"/\\1/')
        fi

        # Create temporary directories for FIFOs
        for i in $(seq 0 $((NUM_PROCESSES-1))); do
            FIFO_DIRS[$i]=$(mktemp -d)

            # Create FIFOs for this process
            mkfifo "${FIFO_DIRS[$i]}/u${i}_to_m"
            mkfifo "${FIFO_DIRS[$i]}/m_to_u${i}"
            chmod 755 "${FIFO_DIRS[$i]}"
            chmod 666 "${FIFO_DIRS[$i]}/u${i}_to_m" "${FIFO_DIRS[$i]}/m_to_u${i}"
        done

        # Prepare manager arguments
        MANAGER_ARGS=""
        for i in $(seq 0 $((NUM_PROCESSES-1))); do
            MANAGER_ARGS="$MANAGER_ARGS ${FIFO_DIRS[$i]}/u${i}_to_m ${FIFO_DIRS[$i]}/m_to_u${i}"
        done

        # Add custom manager arguments if specified
        if [ -n "$MANAGER_CUSTOM_ARGS" ]; then
            MANAGER_ARGS="$MANAGER_ARGS $MANAGER_CUSTOM_ARGS"
        fi

        # Start all user processes first
        for i in $(seq 0 $((NUM_PROCESSES-1))); do
            if [ "$USER_IO" = "fifo_io" ]; then
                # Pass FIFOs as arguments
                ARGS="${FIFO_DIRS[$i]}/m_to_u${i} ${FIFO_DIRS[$i]}/u${i}_to_m"
                if [ "$NUM_PROCESSES" -ne 1 ]; then
                    ARGS="$ARGS $i"
                fi
                ./$TASK_EXECUTABLE $ARGS &
                ALL_PIDS+=($!)
            else
                # Use stdin/stdout redirection
                if [ "$NUM_PROCESSES" -ne 1 ]; then
                    ./$TASK_EXECUTABLE "$i" < "${FIFO_DIRS[$i]}/m_to_u${i}" > "${FIFO_DIRS[$i]}/u${i}_to_m" 2>/dev/null &
                    ALL_PIDS+=($!)
                else
                    ./$TASK_EXECUTABLE < "${FIFO_DIRS[$i]}/m_to_u${i}" > "${FIFO_DIRS[$i]}/u${i}_to_m" 2>/dev/null &
                    ALL_PIDS+=($!)
                fi
            fi
        done

        # Run the manager with timeout using direct pipe from input.txt
        run_with_timeout "$TIME_LIMIT" ./graders/manager $MANAGER_ARGS < input.txt > "$SOLUTION_OUTPUT"

        exit_code=$?

        # Handle non-zero exit codes
        handle_exit_code $exit_code
        if [ $? -ne 0 ]; then
            exit $?
        fi

        # Check the output if we have a correct output AND there's a checker (otherwise we assume the manager handles everything)
        if [ -n "$CORRECT_OUTPUT" ] && [ -f "checker/checker" ]; then
            # Restore the correct output file
            echo "$CORRECT_OUTPUT" > correct_output.txt

            # Let the checker handle it
            ./checker/checker input.txt correct_output.txt "$SOLUTION_OUTPUT"
            exit $?
        else
            # we assume the manager handles it
            cat "$SOLUTION_OUTPUT"
        fi
        ;;

    *)
        echo "0"
        echo "Unsupported task type \"$TASK_TYPE\"" >&2
        exit 1
        ;;
esac
"""


def get_morph_client_from_env(session=None) -> MorphCloudExecutionClient:
    """
    Creates a MorphCloudExecutionClient instance using environment variables.

    Environment variables:
        MORPH_API_KEY: API key for MorphCloud

    Args:
        session: Optional aiohttp.ClientSession to use for HTTP requests

    Returns:
        MorphCloudExecutionClient: A configured MorphCloud execution client
    """
    if not is_morph_available():
        raise ImportError(
            "MorphCloud is not available and required for this function. Please install MorphCloud with "
            "`pip install morphcloud` and add an API key to a `.env` file."
        )

    load_dotenv()
    api_key = os.environ.get("MORPH_API_KEY")
    if not api_key:
        raise ValueError("MORPH_API_KEY environment variable is required")

    return MorphCloudExecutionClient(api_key=api_key)


# noqa: W293

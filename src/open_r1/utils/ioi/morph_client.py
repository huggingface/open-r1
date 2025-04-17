import asyncio
import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, Any, Tuple, Optional


class PistonError(Exception):
    pass


class MorphCloudExecutionClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the MorphCloud execution client.
        
        Args:
            api_key: Optional API key for MorphCloud. If not provided, will use MORPH_API_KEY env var.
            base_url: Optional base URL for MorphCloud API. If not provided, will use default.
        """
        # Import here to avoid circular imports and unnecessary dependencies for users not using MorphCloud
        from morphcloud.api import MorphCloudClient
        
        self.client = MorphCloudClient(api_key=api_key, base_url=base_url)
        # You can customize these based on your needs
        # Cache for snapshots to avoid recreating them
        self.snapshot_cache = {}
        self._snapshot_lock = asyncio.Lock()
        
    async def execute(self, data: Dict[str, Any]) -> Tuple[str, str]:
        """
        Execute code on MorphCloud based on the provided data with enhanced debugging and cleanup.
        
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
        instance = None
        
        # Create a unique temporary directory for this execution
        import tempfile
        temp_dir = tempfile.mkdtemp(prefix=f"morph_exec_")
        print(f"[DEBUG] Created temporary directory: {temp_dir}")
        
        try:
            # Get or create a snapshot with the necessary dependencies
            print("[DEBUG] Attempting to acquire base snapshot...")
            snapshot = await self._get_or_create_base_snapshot()
            print(f"[DEBUG] Successfully acquired snapshot with ID: {snapshot.id}")
            
            # Start an instance from the snapshot
            print(f"[DEBUG] Starting instance from snapshot {snapshot.id}...")
            instance = await self.client.instances.astart(snapshot.id)
            print(f"[DEBUG] Instance started with ID: {instance.id}")
            
            try:
                print(f"[DEBUG] Waiting for instance {instance.id} to become ready...")
                await instance.await_until_ready(timeout=300)
                print(f"[DEBUG] Instance {instance.id} is now ready")

                # Create temporary directory to work with
                print("[DEBUG] Creating workspace directories...")
                await instance.aexec("mkdir -p /workspace")
                await instance.aexec("mkdir -p /workspace/graders")
                print("[DEBUG] Workspace directories created")
                
                # Extract problem ID
                problem_id = None
                graders_files = []
                print("[DEBUG] Analyzing input files to determine problem ID...")
                for file in data["files"]:
                    if file["name"].startswith("graders/") and file["name"].endswith(".cpp"):
                        # This might be the problem ID file (graders/<problem_id>.cpp)
                        potential_id = os.path.basename(file["name"]).split(".")[0]
                        if potential_id not in ["grader", "manager", "stub"]:
                            problem_id = potential_id
                            print(f"[DEBUG] Found problem ID: {problem_id}")
                    
                    if file["name"].startswith("graders/"):
                        graders_files.append(file)
                
                if not problem_id:
                    print("[DEBUG] ERROR: Could not determine problem ID from files")
                    return "0.0", "Could not determine problem ID from files"
                
                # Get compile and run scripts
                compile_script = await self._get_compile_script()
                run_script = await self._get_run_script()
                
                # Upload compile and run scripts and make them executable
                print("[DEBUG] Writing compile script to temporary file...")
                compile_path = os.path.join(temp_dir, "compile")
                with open(compile_path, "w") as f:
                    f.write(data.get("compile_script", compile_script))
                print("[DEBUG] Uploading compile script to instance...")
                await instance.aupload(compile_path, "/workspace/compile")
                print("[DEBUG] Making compile script executable...")
                await instance.aexec("chmod +x /workspace/compile")
                
                print("[DEBUG] Writing run script to temporary file...")
                run_path = os.path.join(temp_dir, "run")
                with open(run_path, "w") as f:
                    f.write(data.get("run_script", run_script))
                print("[DEBUG] Uploading run script to instance...")
                await instance.aupload(run_path, "/workspace/run")
                print("[DEBUG] Making run script executable...")
                await instance.aexec("chmod +x /workspace/run")
                
                # Create a grader_config.json file with task information
                print("[DEBUG] Creating grader configuration...")
                grader_config = {
                    "task_type": "Batch",  # Default to Batch, can be overridden
                    "code": problem_id,
                    "time_limit": data["run_timeout"] / 1000,  # Convert ms to seconds
                    "memory_limit": data["run_memory_limit"] * 1024 * 1024  # Convert MB to bytes
                }
                
                # Check if we need to handle Communication task type
                for file in graders_files:
                    if "manager.cpp" in file["name"]:
                        print("[DEBUG] Detected Communication task type")
                        grader_config["task_type"] = "Communication"
                        # Set default Communication params if needed
                        grader_config["task_type_parameters_Communication_num_processes"] = 1
                        grader_config["task_type_parameters_Communication_user_io"] = "std_io"
                        break
                
                print("[DEBUG] Writing grader configuration to temporary file...")
                config_path = os.path.join(temp_dir, "grader_config.json")
                with open(config_path, "w") as f:
                    json.dump(grader_config, f)
                print("[DEBUG] Uploading grader configuration to instance...")
                await instance.aupload(config_path, "/workspace/graders/grader_config.json")
                
                # Process all files to upload
                print(f"[DEBUG] Uploading {len(data['files'])} files to instance...")
                for index, file in enumerate(data["files"]):
                    file_path = os.path.join(temp_dir, os.path.basename(file["name"]))
                    with open(file_path, "w") as f:
                        f.write(file["content"])
                    
                    # Create directory structure if needed
                    target_path = "/workspace/" + file["name"]
                    dir_path = os.path.dirname(target_path)
                    await instance.aexec(f"mkdir -p {dir_path}")
                    
                    # Upload the file to the instance
                    await instance.aupload(file_path, target_path)
                    print(f"[DEBUG] Uploaded file {index+1}/{len(data['files'])}: {file['name']}")
                
                # Compile the code
                print("[DEBUG] Running compilation...")
                compile_result = await instance.aexec("cd /workspace && ./compile")
                print(f"[DEBUG] Compilation complete with exit code: {compile_result.exit_code}")
                if compile_result.exit_code != 0:
                    print(f"[DEBUG] Compilation error: {compile_result.stderr}")
                    return "0", f"Compilation error exit code {compile_result.exit_code}\n{compile_result.stderr}"
                
                # Run with the specified time limit and memory limit
                # Note: The run script already handles time limits internally
                hard_timeout = data["run_timeout"] / 1000 + 3  # Add 3 seconds as a hard limit
                run_command = f"cd /workspace && timeout {hard_timeout}s ./run"
                
                print(f"[DEBUG] Executing with timeout of {hard_timeout}s: {run_command}")
                run_result = await instance.aexec(run_command)
                print(f"[DEBUG] Execution complete with exit code: {run_result.exit_code}")
                
                # Parse the result based on various conditions
                if run_result.exit_code == 124 or run_result.exit_code == 137 or run_result.exit_code == 143:
                    # Timeout exit codes
                    print("[DEBUG] Detected time limit exceeded")
                    return "0", "Time limit exceeded"
                
                if run_result.exit_code != 0 and "Memory limit exceeded" in run_result.stderr:
                    print("[DEBUG] Detected memory limit exceeded")
                    return "0", "Memory limit exceeded"
                
                # If there's stdout, return it as the score with stderr as feedback
                if run_result.stdout:
                    print(f"[DEBUG] Execution successful with score: {run_result.stdout.strip()}")
                    return run_result.stdout.strip(), run_result.stderr.strip()
                
                # Other error cases
                if run_result.exit_code != 0:
                    print(f"[DEBUG] Runtime error with exit code: {run_result.exit_code}")
                    return "0", f"Runtime error with exit code {run_result.exit_code}\n{run_result.stderr}"
                
                # Default fallback
                print("[DEBUG] No output produced by execution, returning unknown error")
                return "0", "Unknown error"
                
            finally:
                # Always clean up the instance
                if instance:
                    print(f"[DEBUG] Cleaning up instance {instance.id}...")
                    try:
                        await instance.astop()
                        print(f"[DEBUG] Instance {instance.id} stopped")
                    except Exception as e:
                        if "404" in str(e) and "not found" in str(e).lower():
                            # Instance is already gone, that's fine
                            print(f"[DEBUG] Instance {instance.id} already removed")
                        else:
                            # Some other error occurred
                            print(f"[DEBUG] Error stopping instance {instance.id}: {str(e)}")
                    
        except Exception as e:
            # In case of any errors, return a failed score with the error message
            print(f"[DEBUG] Execution error: {str(e)}")
            print(f"[DEBUG] Error details: {type(e).__name__}: {str(e)}")
            import traceback
            print(f"[DEBUG] Traceback: {traceback.format_exc()}")
            return "0", f"Execution error: {str(e)}"
        finally:
            # Final cleanup in case the instance wasn't properly stopped earlier
            if instance:
                try:
                    print(f"[DEBUG] Final cleanup - ensuring instance {instance.id} is stopped...")
                    await instance.astop()
                    print(f"[DEBUG] Instance {instance.id} successfully stopped in final cleanup")
                except Exception as cleanup_error:
                    if "404" in str(cleanup_error) and "not found" in str(cleanup_error).lower():
                        # Instance is already gone, that's fine
                        print(f"[DEBUG] Instance {instance.id} already removed in final cleanup")
                    else:
                        print(f"[DEBUG] Error during final instance cleanup: {str(cleanup_error)}")

            # Clean up the temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
                print(f"[DEBUG] Cleaned up temporary directory: {temp_dir}")
            except Exception as e:
                print(f"[DEBUG] Error cleaning up temporary directory {temp_dir}: {str(e)}")

    async def _get_or_create_base_snapshot(self):
        """Get or create a snapshot with the necessary dependencies for evaluation."""
        # Check if we already have a cached snapshot without acquiring the lock
        if "base_evaluation_snapshot" in self.snapshot_cache:
            return self.snapshot_cache["base_evaluation_snapshot"]
        
        # If no snapshot in cache, acquire the lock before checking again and creating
        async with self._snapshot_lock:
            # Check again in case another task created the snapshot while we were waiting
            if "base_evaluation_snapshot" in self.snapshot_cache:
                return self.snapshot_cache["base_evaluation_snapshot"]
            
            # Create a base snapshot with the necessary tools
            print('Creating base snapshot with build-essential cmake and g++')
            base_snapshot = await self.client.snapshots.acreate(
                vcpus=2,
                memory=4096,  # 4GB
                disk_size=10240,  # 10GB
                metadata={"purpose": "ioi_evaluation"},
                digest="ioi_evaluation",
            )
            
            # Set up the snapshot with required dependencies
            setup_snapshot = await base_snapshot.aexec(
                "apt-get update && "
                "apt-get install -y build-essential cmake g++"
            )
            
            # Create workspace directory
            final_snapshot = await setup_snapshot.aexec("mkdir -p /workspace && chmod 777 /workspace")
            
            # Cache the snapshot for future use
            self.snapshot_cache["base_evaluation_snapshot"] = final_snapshot
            
            return final_snapshot



    async def _get_compile_script(self):
        """Get the compile script content."""
        # First check if there's a compile script in the pistonless directory
        try:
            with open(os.path.join(os.path.dirname(__file__), "../../../../pistonless/compile"), "r") as f:
                compile_script = f.read()
                print("[DEBUG] Using actual compile script from pistonless/compile")
                return compile_script
        except (FileNotFoundError, IOError) as e:
            print(f"[WARNING] Failed to load compile script from pistonless/compile: {e}")
            print("[WARNING] Using fallback compile script - this may cause compatibility issues with complex IOI problems")
            
            # Now try the morph directory as an alternative
            try:
                with open(os.path.join(os.path.dirname(__file__), "../../../../morph/compile"), "r") as f:
                    compile_script = f.read()
                    print("[DEBUG] Using compile script from morph/compile")
                    return compile_script
            except (FileNotFoundError, IOError) as e:
                print(f"[WARNING] Failed to load compile script from morph/compile: {e}")
                
            # Fallback to the full compile script - copied from pistonless/compile
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
        # First check if there's a run script in the pistonless directory
        try:
            with open(os.path.join(os.path.dirname(__file__), "../../../../pistonless/run"), "r") as f:
                run_script = f.read()
                print("[DEBUG] Using actual run script from pistonless/run")
                return run_script
        except (FileNotFoundError, IOError) as e:
            print(f"[WARNING] Failed to load run script from pistonless/run: {e}")
            print("[WARNING] Using fallback run script - this may cause compatibility issues with complex IOI problems")
            
            # Now try the morph directory as an alternative
            try:
                with open(os.path.join(os.path.dirname(__file__), "../../../../morph/run"), "r") as f:
                    run_script = f.read()
                    print("[DEBUG] Using run script from morph/run")
                    return run_script
            except (FileNotFoundError, IOError) as e:
                print(f"[WARNING] Failed to load run script from morph/run: {e}")
            
            # Fallback to the full run script - copied from pistonless/run
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
        echo "Unsupported task type \\"$TASK_TYPE\\"" >&2
        exit 1
        ;;
esac
"""


def get_morph_client_from_env(session=None) -> MorphCloudExecutionClient:
    """
    Creates a MorphCloudExecutionClient instance using environment variables.
    
    Environment variables:
        MORPH_API_KEY: API key for MorphCloud
        MORPH_BASE_URL: Optional base URL for MorphCloud API
        
    Args:
        session: Optional aiohttp.ClientSession to use for HTTP requests
        
    Returns:
        MorphCloudExecutionClient: A configured MorphCloud execution client
    """
    api_key = os.environ.get("MORPH_API_KEY")
    if not api_key:
        raise ValueError("MORPH_API_KEY environment variable is required")
    
    base_url = os.environ.get("MORPH_BASE_URL")
    
    # Create and return the MorphCloud client
    return MorphCloudExecutionClient(api_key=api_key, base_url=base_url)

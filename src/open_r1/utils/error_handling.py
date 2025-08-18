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

"""Enhanced error handling and logging utilities for Open-R1."""

import logging
import sys
import traceback
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Dict, Optional, Type, Union

# Optional wandb import
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


class OpenR1Error(Exception):
    """Base exception class for Open-R1 specific errors."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
    
    def __str__(self):
        if self.error_code:
            return f"[{self.error_code}] {super().__str__()}"
        return super().__str__()


class ModelTrainingError(OpenR1Error):
    """Raised when there's an error during model training."""
    pass


class DataGenerationError(OpenR1Error):
    """Raised when there's an error during data generation."""
    pass


class EvaluationError(OpenR1Error):
    """Raised when there's an error during model evaluation."""
    pass


class CodeExecutionError(OpenR1Error):
    """Raised when there's an error during code execution."""
    pass


def setup_logging(
    level: Union[str, int] = logging.INFO,
    log_file: Optional[str] = None,
    use_wandb: bool = True,
    project_name: str = "open-r1"
) -> logging.Logger:
    """Set up comprehensive logging for Open-R1.
    
    Args:
        level: Logging level
        log_file: Optional file path for logging
        use_wandb: Whether to integrate with Weights & Biases
        project_name: Name for the W&B project
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("open_r1")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    # W&B integration (only if available and requested)
    if use_wandb and WANDB_AVAILABLE and wandb.run is not None:
        wandb_handler = WandBHandler()
        wandb_handler.setLevel(level)
        logger.addHandler(wandb_handler)
    
    return logger


class WandBHandler(logging.Handler):
    """Custom logging handler for Weights & Biases integration."""
    
    def emit(self, record):
        if WANDB_AVAILABLE and wandb.run is not None:
            log_entry = self.format(record)
            wandb.log({
                f"log_{record.levelname.lower()}": log_entry,
                "log_level": record.levelname,
                "log_timestamp": record.created
            })


def safe_execute(
    func: Callable,
    *args,
    error_handler: Optional[Callable] = None,
    default_return: Any = None,
    log_errors: bool = True,
    **kwargs
) -> Any:
    """Safely execute a function with comprehensive error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        error_handler: Optional custom error handler function
        default_return: Value to return if execution fails
        log_errors: Whether to log errors
        **kwargs: Keyword arguments for the function
        
    Returns:
        Function result or default_return if execution fails
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            logger = logging.getLogger("open_r1")
            logger.error(f"Error executing {func.__name__}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
        
        if error_handler:
            try:
                return error_handler(e, *args, **kwargs)
            except Exception as handler_error:
                if log_errors:
                    logger = logging.getLogger("open_r1")
                    logger.error(f"Error handler failed: {str(handler_error)}")
        
        return default_return


@contextmanager
def error_context(
    operation: str,
    error_type: Type[OpenR1Error] = OpenR1Error,
    log_errors: bool = True,
    **context_kwargs
):
    """Context manager for error handling with automatic logging.
    
    Args:
        operation: Description of the operation being performed
        error_type: Type of exception to catch and re-raise
        log_errors: Whether to log errors
        **context_kwargs: Additional context information
    """
    logger = logging.getLogger("open_r1")
    logger.info(f"Starting operation: {operation}")
    
    try:
        yield
        logger.info(f"Successfully completed operation: {operation}")
    except Exception as e:
        if log_errors:
            logger.error(f"Operation '{operation}' failed: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            if context_kwargs:
                logger.debug(f"Context: {context_kwargs}")
        
        # Re-raise as the specified error type
        raise error_type(f"Operation '{operation}' failed: {str(e)}") from e


def retry_on_error(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff_factor: float = 2.0,
    exceptions: tuple = (Exception,),
    log_attempts: bool = True
):
    """Decorator to retry functions on specific exceptions.
    
    Args:
        max_attempts: Maximum number of retry attempts
        delay: Initial delay between attempts in seconds
        backoff_factor: Multiplier for delay on each retry
        exceptions: Tuple of exceptions to catch and retry
        log_attempts: Whether to log retry attempts
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = logging.getLogger("open_r1")
            last_exception = None
            current_delay = delay
            
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        if log_attempts:
                            logger.warning(
                                f"Attempt {attempt + 1}/{max_attempts} failed for {func.__name__}: {str(e)}. "
                                f"Retrying in {current_delay:.2f}s..."
                            )
                        
                        import time
                        time.sleep(current_delay)
                        current_delay *= backoff_factor
                    else:
                        if log_attempts:
                            logger.error(
                                f"All {max_attempts} attempts failed for {func.__name__}. "
                                f"Last error: {str(e)}"
                            )
            
            raise last_exception
        
        return wrapper
    return decorator


def validate_environment() -> Dict[str, bool]:
    """Validate the Open-R1 environment and dependencies.
    
    Returns:
        Dictionary mapping component names to availability status
    """
    status = {}
    
    # Check core ML libraries
    try:
        import torch
        status["torch"] = True
        status["torch_version"] = torch.__version__
    except ImportError:
        status["torch"] = False
    
    try:
        import transformers
        status["transformers"] = True
        status["transformers_version"] = transformers.__version__
    except ImportError:
        status["transformers"] = False
    
    # Check code execution providers
    try:
        from .import_utils import is_e2b_available, is_morph_available
        status["e2b"] = is_e2b_available()
        status["morph"] = is_morph_available()
    except ImportError:
        status["e2b"] = False
        status["morph"] = False
    
    # Check evaluation tools
    try:
        import lighteval
        status["lighteval"] = True
    except ImportError:
        status["lighteval"] = False
    
    # Check training tools
    try:
        import accelerate
        status["accelerate"] = True
    except ImportError:
        status["accelerate"] = False
    
    # Check W&B availability
    status["wandb"] = WANDB_AVAILABLE
    
    return status


def log_environment_info():
    """Log comprehensive environment information for debugging."""
    logger = logging.getLogger("open_r1")
    status = validate_environment()
    
    logger.info("Open-R1 Environment Status:")
    for component, available in status.items():
        if isinstance(available, bool):
            status_text = "✓ Available" if available else "✗ Not Available"
            logger.info(f"  {component}: {status_text}")
        else:
            logger.info(f"  {component}: {available}")
    
    # Log system information
    import platform
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Python: {platform.python_version()}")
    
    # Log GPU information if available
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"CUDA: {torch.version.cuda}")
            logger.info(f"GPU Count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        else:
            logger.info("CUDA: Not Available")
    except ImportError:
        logger.info("CUDA: PyTorch not available")


# Convenience functions for common error patterns
def handle_model_loading_error(model_path: str, error: Exception) -> None:
    """Handle errors during model loading with helpful suggestions."""
    logger = logging.getLogger("open_r1")
    logger.error(f"Failed to load model from {model_path}: {str(error)}")
    
    if "not found" in str(error).lower():
        logger.error("Model path not found. Please check:")
        logger.error("  1. Model path is correct")
        logger.error("  2. Model exists on Hugging Face Hub")
        logger.error("  3. You have proper access permissions")
    elif "out of memory" in str(error).lower():
        logger.error("Out of memory error. Try:")
        logger.error("  1. Reducing batch size")
        logger.error("  2. Using gradient checkpointing")
        logger.error("  3. Using DeepSpeed ZeRO optimization")


def handle_training_error(phase: str, error: Exception) -> None:
    """Handle errors during training with recovery suggestions."""
    logger = logging.getLogger("open_r1")
    logger.error(f"Training error during {phase}: {str(error)}")
    
    if "gradient" in str(error).lower():
        logger.error("Gradient-related error. Try:")
        logger.error("  1. Reducing learning rate")
        logger.error("  2. Adding gradient clipping")
        logger.error("  3. Using mixed precision training")
    elif "memory" in str(error).lower():
        logger.error("Memory error. Try:")
        logger.error("  1. Reducing batch size")
        logger.error("  2. Using gradient accumulation")
        logger.error("  3. Enabling gradient checkpointing")

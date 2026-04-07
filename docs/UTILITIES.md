# Open-R1 Utilities Documentation

This document provides comprehensive documentation for the utility modules in Open-R1, which enhance the robustness, monitoring, and configuration management of the project.

## Table of Contents

1. [Error Handling](#error-handling)
2. [Performance Monitoring](#performance-monitoring)
3. [Data Validation](#data-validation)
4. [Configuration Management](#configuration-management)
5. [Usage Examples](#usage-examples)
6. [Best Practices](#best-practices)

## Error Handling

The error handling utilities provide robust error management, logging, and recovery mechanisms for Open-R1 operations.

### Core Classes

#### `OpenR1Error`
Base exception class for Open-R1 specific errors with error codes and detailed context.

```python
from open_r1.utils import OpenR1Error

# Create a custom error
error = OpenR1Error(
    message="Training failed due to memory issues",
    error_code="MEMORY_ERROR",
    details={"gpu_memory": "8GB", "required": "12GB"}
)
```

#### Specific Error Types
- `ModelTrainingError`: Raised during model training failures
- `DataGenerationError`: Raised during data generation failures
- `EvaluationError`: Raised during model evaluation failures
- `CodeExecutionError`: Raised during code execution failures

### Functions

#### `setup_logging()`
Configure comprehensive logging with console, file, and W&B integration.

```python
from open_r1.utils import setup_logging

# Basic setup
logger = setup_logging(level="INFO")

# With file logging
logger = setup_logging(
    level="DEBUG",
    log_file="training.log",
    use_wandb=True,
    project_name="my-experiment"
)
```

#### `safe_execute()`
Safely execute functions with automatic error handling and fallback values.

```python
from open_r1.utils import safe_execute

def risky_function(x):
    if x < 0:
        raise ValueError("Negative numbers not allowed")
    return x ** 2

# Safe execution with fallback
result = safe_execute(
    risky_function, 
    -5, 
    default_return=0,
    log_errors=True
)
# Returns 0 instead of crashing
```

#### `error_context()`
Context manager for automatic error logging and re-raising with proper error types.

```python
from open_r1.utils import error_context, ModelTrainingError

with error_context("model training", ModelTrainingError):
    # Your training code here
    train_model()
    # If any error occurs, it's automatically logged and re-raised as ModelTrainingError
```

#### `retry_on_error()`
Decorator for automatic retry logic with exponential backoff.

```python
from open_r1.utils import retry_on_error

@retry_on_error(max_attempts=3, delay=1.0, backoff_factor=2.0)
def download_model():
    # This function will be retried up to 3 times with increasing delays
    # 1s, 2s, 4s between attempts
    return download_from_hub()
```

#### `validate_environment()`
Check the availability of required dependencies and system components.

```python
from open_r1.utils import validate_environment, log_environment_info

# Check environment status
status = validate_environment()
print(f"PyTorch available: {status['torch']}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Log comprehensive environment info
log_environment_info()
```

## Performance Monitoring

The performance monitoring utilities provide real-time tracking of training metrics, resource usage, and system performance.

### Core Classes

#### `PerformanceMonitor`
Main class for monitoring operations with automatic W&B integration.

```python
from open_r1.utils import PerformanceMonitor

# Create monitor
monitor = PerformanceMonitor(enable_wandb=True, log_interval=30.0)

# Monitor an operation
with monitor.monitor("training_epoch") as metrics:
    # Your training code here
    train_epoch()
    
    # Metrics are automatically collected and logged
    print(f"Operation took {metrics.duration:.2f} seconds")
    print(f"Peak memory: {metrics.peak_memory_mb:.1f} MB")

# Cleanup
monitor.stop()
```

#### `TrainingMetricsCollector`
Specialized collector for training-specific metrics with automatic W&B logging.

```python
from open_r1.utils import create_training_metrics_collector

# Create collector
collector = create_training_metrics_collector("my-model", "sft")

# Log training step
collector.log_training_step(
    loss=0.5,
    learning_rate=1e-4,
    gradient_norm=1.2,
    step_time=0.1,
    memory_usage=2048.0
)

# Log evaluation metrics
collector.log_evaluation_metrics(
    {"accuracy": 0.85, "f1": 0.82},
    split="validation"
)

# Get training summary
summary = collector.get_training_summary()
print(f"Average loss: {summary['loss_mean']:.4f}")
```

#### `monitor_performance()`
Decorator for automatic performance monitoring of functions.

```python
from open_r1.utils import monitor_performance

@monitor_performance("model_inference")
def generate_text(prompt):
    # This function will be automatically monitored
    return model.generate(prompt)
```

## Data Validation

The data validation utilities ensure data quality and consistency across the Open-R1 pipeline.

### Core Classes

#### `DataValidator`
Main class for validating datasets according to configurable rules.

```python
from open_r1.utils import DataValidator

# Create validator
validator = DataValidator(strict_mode=False)

# Define validation rules
rules = {
    "text_length": {
        "text_column": "text",
        "min_length": 10,
        "max_length": 10000
    },
    "format_compliance": {
        "text_column": "text",
        "required_format": "think_answer"
    },
    "duplicate_detection": {
        "text_column": "text",
        "similarity_threshold": 0.95
    }
}

# Validate dataset
report = validator.validate_dataset(dataset, rules)
print(report)
```

#### `validate_open_r1_dataset()`
Convenience function for quick dataset validation with predefined rules.

```python
from open_r1.utils import validate_open_r1_dataset

# Validate reasoning dataset
report = validate_open_r1_dataset(
    dataset,
    task_type="reasoning",
    text_column="text"
)

if report.quality_score < 0.8:
    print("Dataset quality is low. Recommendations:")
    for rec in report.recommendations:
        print(f"- {rec}")
```

### Validation Rules

#### Task-Specific Rules
- **reasoning**: Format compliance, text length, mathematical content
- **code**: Code blocks, language support, text length
- **math**: Mathematical notation, format compliance
- **general**: Basic structure validation

#### Custom Rules
```python
# Create custom validation rules
custom_rules = create_validation_rules(
    "reasoning",
    text_column="prompt",
    min_length=20,
    max_length=5000,
    required_patterns=[r"\\boxed\{.*?\}"],
    forbidden_patterns=[r"TODO", r"FIXME"]
)
```

## Configuration Management

The configuration management utilities provide structured, validated configuration handling for all Open-R1 operations.

### Core Classes

#### `OpenR1Config`
Main configuration class that combines all configuration aspects.

```python
from open_r1.utils import OpenR1Config, ModelConfig, TrainingConfig

# Create configuration
config = OpenR1Config(
    model=ModelConfig(
        model_name_or_path="Qwen/Qwen2.5-1.5B",
        model_type="qwen",
        torch_dtype="bfloat16"
    ),
    training=TrainingConfig(
        learning_rate=4e-5,
        num_train_epochs=3,
        per_device_train_batch_size=4
    ),
    # ... other configs
    experiment_name="my-experiment",
    description="Training Qwen model on reasoning tasks"
)

# Validate configuration
issues = config.validate()
if issues:
    print("Configuration issues found:")
    for issue in issues:
        print(f"- {issue}")

# Save configuration
config.save("config.yaml")

# Calculate resource requirements
memory_estimate = config.get_estimated_memory_usage()
print(f"Estimated memory usage: {memory_estimate['total']:.1f} GB")
```

#### `ConfigManager`
Manager for handling multiple configurations with template support.

```python
from open_r1.utils import ConfigManager

# Create manager
manager = ConfigManager("configs/")

# Create template configurations
sft_config = manager.create_template_config("my-sft", "sft")
grpo_config = manager.create_template_config("my-grpo", "grpo")

# List available configurations
configs = manager.list_configs()
print(f"Available configs: {configs}")

# Load configuration
config = manager.load_config("my-sft")
```

### Template Configurations

#### SFT Training
```python
from open_r1.utils import create_default_config

# Create SFT configuration
sft_config = create_default_config("sft")
sft_config.training.learning_rate = 5e-5
sft_config.dataset.dataset_name = "my-dataset"
sft_config.save("my_sft_config.yaml")
```

#### GRPO Training
```python
# Create GRPO configuration
grpo_config = create_default_config("grpo")
grpo_config.training.learning_rate = 1e-5
grpo_config.dataset.dataset_name = "code-dataset"
grpo_config.save("my_grpo_config.yaml")
```

#### Evaluation
```python
# Create evaluation configuration
eval_config = create_default_config("evaluation")
eval_config.evaluation.eval_datasets = ["aime24", "math_500"]
eval_config.save("my_eval_config.yaml")
```

## Usage Examples

### Complete Training Pipeline with Utilities

```python
from open_r1.utils import (
    setup_logging, PerformanceMonitor, TrainingMetricsCollector,
    error_context, safe_execute, create_default_config
)

# Setup logging
logger = setup_logging(level="INFO", log_file="training.log")

# Create configuration
config = create_default_config("sft")
config.model.model_name_or_path = "Qwen/Qwen2.5-1.5B"
config.dataset.dataset_name = "open-r1/Mixture-of-Thoughts"

# Setup monitoring
monitor = PerformanceMonitor(enable_wandb=True)
collector = create_training_metrics_collector("Qwen2.5-1.5B", "sft")

# Training loop with error handling and monitoring
with error_context("model training", ModelTrainingError):
    for epoch in range(config.training.num_train_epochs):
        with monitor.monitor(f"epoch_{epoch}") as epoch_metrics:
            
            # Training epoch
            for step, batch in enumerate(train_dataloader):
                try:
                    loss = train_step(batch)
                    
                    # Log metrics
                    collector.log_training_step(
                        loss=loss,
                        learning_rate=config.training.learning_rate,
                        step_time=step_time
                    )
                    
                except Exception as e:
                    logger.error(f"Training step {step} failed: {e}")
                    continue
            
            # Log epoch metrics
            epoch_metrics.custom_metrics["epoch"] = epoch
            epoch_metrics.custom_metrics["avg_loss"] = avg_loss

# Cleanup
monitor.stop()
logger.info("Training completed successfully")
```

### Data Quality Assurance

```python
from open_r1.utils import validate_open_r1_dataset, DataValidator

# Quick validation
report = validate_open_r1_dataset(
    dataset,
    task_type="reasoning",
    text_column="text"
)

print(f"Dataset quality score: {report.quality_score:.2%}")

# Detailed validation with custom rules
validator = DataValidator(strict_mode=True)
custom_rules = {
    "text_length": {
        "text_column": "text",
        "min_length": 50,
        "max_length": 5000
    },
    "format_compliance": {
        "text_column": "text",
        "required_format": "think_answer"
    },
    "mathematical_content": {
        "text_column": "text",
        "math_patterns": [r"\\boxed\{.*?\}", r"\\frac\{.*?\}\{.*?\}"]
    }
}

detailed_report = validator.validate_dataset(dataset, custom_rules)

if not detailed_report.quality_score > 0.9:
    print("Dataset needs improvement:")
    for result in detailed_report.validation_results:
        if not result.is_valid:
            print(f"- {result.message}")
```

### Configuration Management

```python
from open_r1.utils import ConfigManager, load_and_validate_config

# Create configuration manager
manager = ConfigManager("experiments/")

# Create experiment configurations
for model_size in ["1.5B", "7B", "14B"]:
    config = manager.create_template_config(f"qwen-{model_size}", "sft")
    config.model.model_name_or_path = f"Qwen/Qwen2.5-{model_size}"
    config.experiment_name = f"Qwen-{model_size}-SFT"
    config.description = f"Supervised fine-tuning of Qwen {model_size}"
    
    # Customize based on model size
    if model_size == "1.5B":
        config.training.per_device_train_batch_size = 8
        config.training.gradient_accumulation_steps = 2
    elif model_size == "7B":
        config.training.per_device_train_batch_size = 4
        config.training.gradient_accumulation_steps = 4
    else:  # 14B
        config.training.per_device_train_batch_size = 2
        config.training.gradient_accumulation_steps = 8
    
    manager.save_config(f"qwen-{model_size}-sft", config)

# Load and validate configuration
config, issues = load_and_validate_config(
    "experiments/qwen-7b-sft.yaml",
    strict=False
)

if issues:
    print("Configuration warnings:")
    for issue in issues:
        print(f"- {issue}")

# Use configuration
print(f"Training {config.model.model_name_or_path}")
print(f"Effective batch size: {config.get_effective_batch_size()}")
print(f"Estimated memory: {config.get_estimated_memory_usage()['total']:.1f} GB")
```

## Best Practices

### Error Handling
1. **Use specific error types** for different failure modes
2. **Implement retry logic** for transient failures
3. **Log errors with context** for debugging
4. **Provide fallback values** when possible
5. **Use error contexts** for automatic logging

### Performance Monitoring
1. **Monitor key operations** like training steps and data loading
2. **Set appropriate log intervals** to avoid overwhelming W&B
3. **Use custom metrics** for domain-specific measurements
4. **Clean up monitors** to prevent resource leaks
5. **Monitor system resources** for optimization opportunities

### Data Validation
1. **Validate early** in the pipeline
2. **Use task-specific rules** for relevant validation
3. **Set appropriate thresholds** for your use case
4. **Review recommendations** for data quality improvement
5. **Validate at multiple stages** of data processing

### Configuration Management
1. **Use templates** for common configurations
2. **Validate configurations** before use
3. **Version control** your configurations
4. **Document customizations** for reproducibility
5. **Estimate resource requirements** before training

### General Guidelines
1. **Import utilities** from the main utils module
2. **Handle exceptions gracefully** with appropriate fallbacks
3. **Log important events** for debugging and monitoring
4. **Use type hints** for better code clarity
5. **Test utilities** with your specific use cases

## Troubleshooting

### Common Issues

#### Import Errors
```python
# If you get import errors, ensure you're importing from the right place
from open_r1.utils import setup_logging  # Correct
from open_r1.utils.error_handling import setup_logging  # Also correct
```

#### W&B Integration Issues
```python
# If W&B logging fails, check your login status
import wandb
if wandb.run is None:
    wandb.login()
    # Or disable W&B integration
    monitor = PerformanceMonitor(enable_wandb=False)
```

#### Configuration Validation Failures
```python
# If configuration validation fails, check the specific issues
config, issues = load_and_validate_config("config.yaml", strict=False)
for issue in issues:
    print(f"Configuration issue: {issue}")
```

#### Performance Monitoring Overhead
```python
# If monitoring adds too much overhead, increase log intervals
monitor = PerformanceMonitor(log_interval=120.0)  # Log every 2 minutes
```

### Getting Help

1. **Check the logs** for detailed error messages
2. **Validate your configuration** using the validation utilities
3. **Test utilities individually** to isolate issues
4. **Review the test suite** for usage examples
5. **Check environment compatibility** with `validate_environment()`

## Contributing

When adding new utilities:

1. **Follow the existing patterns** for consistency
2. **Add comprehensive tests** to the test suite
3. **Update this documentation** with usage examples
4. **Use type hints** for better code clarity
5. **Add error handling** for robust operation
6. **Include logging** for debugging support
7. **Update the utils __init__.py** to export new functions

The utilities are designed to be extensible, so feel free to add new functionality that follows the established patterns.

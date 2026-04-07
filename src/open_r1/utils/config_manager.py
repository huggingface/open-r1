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

"""Enhanced configuration management utilities for Open-R1."""

import os
import yaml
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field, asdict
from copy import deepcopy

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    from transformers import TrainingArguments
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    TrainingArguments = None


@dataclass
class ModelConfig:
    """Configuration for model loading and setup."""
    
    model_name_or_path: str
    model_type: Optional[str] = None
    trust_remote_code: bool = True
    torch_dtype: str = "auto"
    device_map: Optional[str] = None
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    
    def __post_init__(self):
        if self.torch_dtype == "auto":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                self.torch_dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
            else:
                self.torch_dtype = "float32"


@dataclass
class TrainingConfig:
    """Configuration for training parameters."""
    
    # Basic training parameters
    learning_rate: float = 5e-5
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Optimization
    weight_decay: float = 0.01
    warmup_steps: int = 100
    lr_scheduler_type: str = "cosine"
    
    # Mixed precision
    bf16: bool = False  # Changed from True to False for compatibility
    fp16: bool = False
    
    # Memory optimization
    gradient_checkpointing: bool = True
    dataloader_pin_memory: bool = False
    
    # Logging and saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: Optional[int] = 3
    
    # Evaluation
    evaluation_strategy: str = "steps"
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False
    
    def __post_init__(self):
        """Set bf16 to True only if CUDA is available and supports bf16."""
        if self.bf16 and TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                self.bf16 = torch.cuda.is_bf16_supported()
            except:
                self.bf16 = False
        else:
            self.bf16 = False
    
    def to_training_arguments(self, output_dir: str):
        """Convert to HuggingFace TrainingArguments."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("transformers is required to create TrainingArguments")
        
        return TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=self.num_train_epochs,
            per_device_train_batch_size=self.per_device_train_batch_size,
            per_device_eval_batch_size=self.per_device_eval_batch_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            max_grad_norm=self.max_grad_norm,
            weight_decay=self.weight_decay,
            warmup_steps=self.warmup_steps,
            lr_scheduler_type=self.lr_scheduler_type,
            bf16=self.bf16,
            fp16=self.fp16,
            gradient_checkpointing=self.gradient_checkpointing,
            dataloader_pin_memory=self.dataloader_pin_memory,
            logging_steps=self.logging_steps,
            save_steps=self.save_steps,
            eval_steps=self.eval_steps,
            save_total_limit=self.save_total_limit,
            eval_strategy=self.evaluation_strategy,
            metric_for_best_model=self.metric_for_best_model,
            greater_is_better=self.greater_is_better,
        )


@dataclass
class DatasetConfig:
    """Configuration for dataset loading and processing."""
    
    dataset_name: str
    dataset_config: Optional[str] = None
    split: str = "train"
    text_column: str = "text"
    target_column: Optional[str] = None
    
    # Data processing
    max_seq_length: int = 2048
    truncation: bool = True
    padding: bool = True
    
    # Data filtering
    min_length: Optional[int] = None
    max_length: Optional[int] = None
    filter_duplicates: bool = True
    
    # Validation split
    validation_split: Optional[float] = 0.1
    validation_split_seed: int = 42


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Evaluation settings
    eval_batch_size: int = 8
    max_eval_samples: Optional[int] = None
    
    # Metrics
    metrics: List[str] = field(default_factory=lambda: ["accuracy", "f1", "precision", "recall"])
    
    # Generation parameters
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    do_sample: bool = True
    
    # Evaluation datasets
    eval_datasets: List[str] = field(default_factory=list)
    
    # Benchmark-specific settings
    benchmark_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class SystemConfig:
    """Configuration for system and hardware settings."""
    
    # Hardware
    num_gpus: int = 1
    gpu_memory_fraction: float = 0.9
    mixed_precision: str = "bf16"
    
    # Distributed training
    distributed_backend: str = "nccl"
    local_rank: int = -1
    world_size: int = 1
    
    # Memory optimization
    max_memory: Optional[Dict[str, str]] = None
    offload_folder: Optional[str] = None
    
    # Environment
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        if self.max_memory is None:
            self.max_memory = {"0": "40GB"} if TORCH_AVAILABLE and torch.cuda.is_available() else {}


@dataclass
class OpenR1Config:
    """Main configuration class for Open-R1."""
    
    # Core configurations
    model: ModelConfig
    training: TrainingConfig
    dataset: DatasetConfig
    evaluation: EvaluationConfig
    system: SystemConfig
    
    # Metadata
    experiment_name: str = "open-r1-experiment"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    # Paths
    output_dir: str = "outputs"
    cache_dir: Optional[str] = None
    log_dir: Optional[str] = None
    
    # W&B integration
    wandb_project: str = "open-r1"
    wandb_entity: Optional[str] = None
    wandb_run_name: Optional[str] = None
    
    def __post_init__(self):
        # Set default paths
        if self.cache_dir is None:
            self.cache_dir = os.path.join(self.output_dir, "cache")
        
        if self.log_dir is None:
            self.log_dir = os.path.join(self.output_dir, "logs")
        
        # Create directories
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    def save(self, filepath: Union[str, Path]):
        """Save configuration to file."""
        filepath = Path(filepath)
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'OpenR1Config':
        """Load configuration from file."""
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'OpenR1Config':
        """Create configuration from dictionary."""
        # Extract nested configurations
        model_config = ModelConfig(**config_dict.get('model', {}))
        training_config = TrainingConfig(**config_dict.get('training', {}))
        dataset_config = DatasetConfig(**config_dict.get('dataset', {}))
        evaluation_config = EvaluationConfig(**config_dict.get('evaluation', {}))
        system_config = SystemConfig(**config_dict.get('system', {}))
        
        # Remove nested configs from main dict
        main_config = {k: v for k, v in config_dict.items() 
                      if k not in ['model', 'training', 'dataset', 'evaluation', 'system']}
        
        return cls(
            model=model_config,
            training=training_config,
            dataset=dataset_config,
            evaluation=evaluation_config,
            system=system_config,
            **main_config
        )
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Validate model configuration
        if not self.model.model_name_or_path:
            issues.append("Model name or path is required")
        
        # Validate training configuration
        if self.training.learning_rate <= 0:
            issues.append("Learning rate must be positive")
        
        if self.training.num_train_epochs <= 0:
            issues.append("Number of training epochs must be positive")
        
        if self.training.per_device_train_batch_size <= 0:
            issues.append("Per device batch size must be positive")
        
        # Validate dataset configuration
        if not self.dataset.dataset_name:
            issues.append("Dataset name is required")
        
        if self.dataset.max_seq_length <= 0:
            issues.append("Max sequence length must be positive")
        
        # Validate system configuration
        if self.system.num_gpus < 0:
            issues.append("Number of GPUs cannot be negative")
        
        if not (0 < self.system.gpu_memory_fraction <= 1):
            issues.append("GPU memory fraction must be between 0 and 1")
        
        return issues
    
    def get_effective_batch_size(self) -> int:
        """Calculate effective batch size."""
        return (
            self.training.per_device_train_batch_size *
            self.training.gradient_accumulation_steps *
            self.system.num_gpus
        )
    
    def get_estimated_memory_usage(self) -> Dict[str, float]:
        """Estimate memory usage for the configuration."""
        # This is a simplified estimation
        batch_size = self.get_effective_batch_size()
        seq_length = self.dataset.max_seq_length
        
        # Rough estimation (in GB)
        estimated_memory = {
            "model_parameters": 7.0,  # Base model size
            "activations": batch_size * seq_length * 0.001,  # Rough estimate
            "gradients": batch_size * seq_length * 0.001,
            "optimizer_states": batch_size * seq_length * 0.002,
        }
        
        total_memory = sum(estimated_memory.values())
        estimated_memory["total"] = total_memory
        
        return estimated_memory


class ConfigManager:
    """Manager for handling multiple configurations."""
    
    def __init__(self, config_dir: Union[str, Path] = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.configs: Dict[str, OpenR1Config] = {}
    
    def load_config(self, name: str) -> OpenR1Config:
        """Load a configuration by name."""
        if name in self.configs:
            return self.configs[name]
        
        config_file = self.config_dir / f"{name}.yaml"
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration '{name}' not found: {config_file}")
        
        config = OpenR1Config.from_file(config_file)
        self.configs[name] = config
        return config
    
    def save_config(self, name: str, config: OpenR1Config):
        """Save a configuration with a given name."""
        config_file = self.config_dir / f"{name}.yaml"
        config.save(config_file)
        self.configs[name] = config
    
    def list_configs(self) -> List[str]:
        """List all available configuration names."""
        config_files = list(self.config_dir.glob("*.yaml")) + list(self.config_dir.glob("*.yml"))
        return [f.stem for f in config_files]
    
    def create_template_config(self, name: str, task_type: str = "general") -> OpenR1Config:
        """Create a template configuration for a specific task type."""
        if task_type == "sft":
            config = self._create_sft_template()
        elif task_type == "grpo":
            config = self._create_grpo_template()
        elif task_type == "evaluation":
            config = self._create_evaluation_template()
        else:
            config = self._create_general_template()
        
        config.experiment_name = name
        self.save_config(name, config)
        return config
    
    def _create_sft_template(self) -> OpenR1Config:
        """Create SFT training template configuration."""
        return OpenR1Config(
            model=ModelConfig(
                model_name_or_path="Qwen/Qwen2.5-1.5B",
                model_type="qwen",
            ),
            training=TrainingConfig(
                learning_rate=4e-5,
                num_train_epochs=3,
                per_device_train_batch_size=4,
                gradient_accumulation_steps=4,
                warmup_steps=100,
                logging_steps=10,
                save_steps=500,
                eval_steps=500,
            ),
            dataset=DatasetConfig(
                dataset_name="open-r1/Mixture-of-Thoughts",
                dataset_config="all",
                max_seq_length=2048,
            ),
            evaluation=EvaluationConfig(
                eval_batch_size=4,
                metrics=["accuracy", "f1"],
            ),
            system=SystemConfig(
                num_gpus=1,
                seed=42,
            ),
            experiment_name="sft-template",
            description="Template configuration for SFT training",
        )
    
    def _create_grpo_template(self) -> OpenR1Config:
        """Create GRPO training template configuration."""
        return OpenR1Config(
            model=ModelConfig(
                model_name_or_path="Qwen/Qwen2.5-1.5B-Instruct",
                model_type="qwen",
            ),
            training=TrainingConfig(
                learning_rate=1e-5,
                num_train_epochs=1,
                per_device_train_batch_size=2,
                gradient_accumulation_steps=8,
                warmup_steps=50,
                logging_steps=5,
                save_steps=100,
                eval_steps=100,
            ),
            dataset=DatasetConfig(
                dataset_name="open-r1/verifiable-coding-problems-python",
                max_seq_length=4096,
            ),
            evaluation=EvaluationConfig(
                eval_batch_size=2,
                metrics=["pass_rate", "execution_success"],
            ),
            system=SystemConfig(
                num_gpus=2,
                seed=42,
            ),
            experiment_name="grpo-template",
            description="Template configuration for GRPO training",
        )
    
    def _create_evaluation_template(self) -> OpenR1Config:
        """Create evaluation template configuration."""
        return OpenR1Config(
            model=ModelConfig(
                model_name_or_path="open-r1/OpenR1-Distill-7B",
                model_type="qwen",
            ),
            training=TrainingConfig(
                learning_rate=0.0,  # No training
                num_train_epochs=0,
                per_device_train_batch_size=1,
            ),
            dataset=DatasetConfig(
                dataset_name="open-r1/Mixture-of-Thoughts",
                split="test",
                max_seq_length=2048,
            ),
            evaluation=EvaluationConfig(
                eval_batch_size=8,
                max_eval_samples=1000,
                metrics=["accuracy", "f1", "precision", "recall"],
                eval_datasets=["aime24", "math_500", "gpqa"],
            ),
            system=SystemConfig(
                num_gpus=1,
                seed=42,
            ),
            experiment_name="evaluation-template",
            description="Template configuration for model evaluation",
        )
    
    def _create_general_template(self) -> OpenR1Config:
        """Create general template configuration."""
        return OpenR1Config(
            model=ModelConfig(
                model_name_or_path="Qwen/Qwen2.5-1.5B",
                model_type="qwen",
            ),
            training=TrainingConfig(),
            dataset=DatasetConfig(
                dataset_name="example-dataset",
                max_seq_length=2048,
            ),
            evaluation=EvaluationConfig(),
            system=SystemConfig(),
            experiment_name="general-template",
            description="General template configuration",
        )


def create_default_config(
    task_type: str = "general",
    output_path: Optional[Union[str, Path]] = None
) -> OpenR1Config:
    """Create a default configuration for the specified task type."""
    config_manager = ConfigManager()
    config = config_manager.create_template_config("default", task_type)
    
    if output_path:
        config.save(output_path)
    
    return config


def load_and_validate_config(
    config_path: Union[str, Path],
    strict: bool = False
) -> Tuple[OpenR1Config, List[str]]:
    """Load and validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        strict: Whether to raise error on validation issues
        
    Returns:
        Tuple of (config, validation_issues)
    """
    config = OpenR1Config.from_file(config_path)
    issues = config.validate()
    
    if strict and issues:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {issue}" for issue in issues))
    
    return config, issues

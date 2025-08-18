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

"""Tests for the new utility modules."""

import unittest
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch
from datasets import Dataset

from open_r1.utils import (
    # Error handling
    OpenR1Error,
    ModelTrainingError,
    DataGenerationError,
    EvaluationError,
    CodeExecutionError,
    setup_logging,
    safe_execute,
    error_context,
    retry_on_error,
    validate_environment,
    log_environment_info,
    handle_model_loading_error,
    handle_training_error,
    
    # Performance monitoring
    PerformanceMonitor,
    PerformanceMetrics,
    TrainingMetricsCollector,
    monitor_performance,
    create_training_metrics_collector,
    get_global_monitor,
    set_global_monitor,
    
    # Data validation
    DataValidator,
    ValidationResult,
    DataQualityReport,
    create_validation_rules,
    validate_open_r1_dataset,
    
    # Configuration management
    OpenR1Config,
    ModelConfig,
    TrainingConfig,
    DatasetConfig,
    EvaluationConfig,
    SystemConfig,
    ConfigManager,
    create_default_config,
    load_and_validate_config,
)


class TestErrorHandling(unittest.TestCase):
    """Test error handling utilities."""
    
    def test_open_r1_error_creation(self):
        """Test OpenR1Error creation and string representation."""
        error = OpenR1Error("Test error", "TEST001", {"detail": "test"})
        self.assertEqual(str(error), "[TEST001] Test error")
        self.assertEqual(error.error_code, "TEST001")
        self.assertEqual(error.details, {"detail": "test"})
    
    def test_specific_error_types(self):
        """Test specific error type creation."""
        training_error = ModelTrainingError("Training failed")
        self.assertIsInstance(training_error, OpenR1Error)
        self.assertIsInstance(training_error, ModelTrainingError)
        
        data_error = DataGenerationError("Data generation failed")
        self.assertIsInstance(data_error, DataGenerationError)
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function."""
        def test_func(x, y):
            return x + y
        
        result = safe_execute(test_func, 2, 3)
        self.assertEqual(result, 5)
    
    def test_safe_execute_failure(self):
        """Test safe_execute with failing function."""
        def test_func():
            raise ValueError("Test error")
        
        result = safe_execute(test_func, default_return="fallback")
        self.assertEqual(result, "fallback")
    
    def test_error_context_success(self):
        """Test error_context with successful operation."""
        with error_context("test operation"):
            result = "success"
        
        self.assertEqual(result, "success")
    
    def test_error_context_failure(self):
        """Test error_context with failing operation."""
        with self.assertRaises(OpenR1Error):
            with error_context("test operation"):
                raise ValueError("Test error")
    
    def test_retry_on_error_success(self):
        """Test retry_on_error decorator with eventual success."""
        attempt_count = 0
        
        @retry_on_error(max_attempts=3, delay=0.1)
        def test_func():
            nonlocal attempt_count
            attempt_count += 1
            if attempt_count < 3:
                raise ValueError("Temporary error")
            return "success"
        
        result = test_func()
        self.assertEqual(result, "success")
        self.assertEqual(attempt_count, 3)
    
    def test_retry_on_error_failure(self):
        """Test retry_on_error decorator with persistent failure."""
        @retry_on_error(max_attempts=2, delay=0.1)
        def test_func():
            raise ValueError("Persistent error")
        
        with self.assertRaises(ValueError):
            test_func()
    
    @patch('torch.cuda.is_available')
    def test_validate_environment(self, mock_cuda):
        """Test environment validation."""
        mock_cuda.return_value = False
        status = validate_environment()
        
        self.assertIn("torch", status)
        self.assertIn("transformers", status)
        self.assertIn("e2b", status)
        self.assertIn("morph", status)


class TestPerformanceMonitoring(unittest.TestCase):
    """Test performance monitoring utilities."""
    
    def test_performance_metrics_creation(self):
        """Test PerformanceMetrics creation and updates."""
        metrics = PerformanceMetrics()
        
        # Test memory update
        metrics.update_memory()
        self.assertGreater(metrics.current_memory_mb, 0)
        
        # Test GPU metrics update
        metrics.update_gpu_metrics()
        
        # Test finalization
        metrics.finalize()
        self.assertIsNotNone(metrics.duration)
        self.assertGreater(metrics.duration, 0)
    
    def test_performance_monitor_basic(self):
        """Test basic PerformanceMonitor functionality."""
        monitor = PerformanceMonitor(enable_wandb=False)
        
        # Test monitoring
        with monitor.monitor("test_operation") as metrics:
            import time
            time.sleep(0.1)
        
        # Check that metrics were recorded
        self.assertEqual(len(monitor.metrics_history), 1)
        self.assertGreater(monitor.metrics_history[0].duration, 0)
        
        # Cleanup
        monitor.stop()
    
    def test_training_metrics_collector(self):
        """Test TrainingMetricsCollector functionality."""
        collector = create_training_metrics_collector("test-model", "test-task")
        
        # Log training step
        collector.log_training_step(
            loss=0.5,
            learning_rate=1e-4,
            gradient_norm=1.0,
            step_time=0.1
        )
        
        # Check metrics
        self.assertEqual(collector.current_step, 1)
        self.assertEqual(len(collector.metrics["loss"]), 1)
        self.assertEqual(collector.metrics["loss"][0], 0.5)
        
        # Get summary
        summary = collector.get_training_summary()
        self.assertIn("loss_mean", summary)
        self.assertEqual(summary["loss_mean"], 0.5)


class TestDataValidation(unittest.TestCase):
    """Test data validation utilities."""
    
    def setUp(self):
        """Set up test data."""
        self.test_data = {
            "text": [
                "<think>Let me solve this step by step.</think><answer>The answer is 42.</answer>",
                "<think>I need to think about this.</think><answer>Result is 100.</answer>",
                "This is invalid text without tags",
                "<think>Another valid example</think><answer>Final answer</answer>"
            ],
            "label": [1, 1, 0, 1]
        }
        self.dataset = Dataset.from_dict(self.test_data)
    
    def test_validation_result_creation(self):
        """Test ValidationResult creation and string representation."""
        result = ValidationResult(
            is_valid=True,
            message="Test validation",
            details={"test": "data"}
        )
        
        self.assertTrue(result.is_valid)
        self.assertIn("✓ VALID", str(result))
        
        invalid_result = ValidationResult(
            is_valid=False,
            message="Test validation failed",
            details={"errors": ["error1", "error2"]},
            error_count=2
        )
        
        self.assertFalse(invalid_result.is_valid)
        self.assertIn("✗ INVALID", str(invalid_result))
        self.assertEqual(invalid_result.error_count, 2)
    
    def test_data_validator_basic(self):
        """Test basic DataValidator functionality."""
        validator = DataValidator()
        
        # Test text length validation
        rules = {
            "text_length": {
                "text_column": "text",
                "min_length": 10,
                "max_length": 1000
            }
        }
        
        report = validator.validate_dataset(self.dataset, rules)
        self.assertIsInstance(report, DataQualityReport)
        self.assertEqual(report.total_samples, 4)
    
    def test_format_compliance_validation(self):
        """Test format compliance validation."""
        validator = DataValidator()
        
        rules = {
            "format_compliance": {
                "text_column": "text",
                "required_format": "think_answer"
            }
        }
        
        report = validator.validate_dataset(self.dataset, rules)
        
        # Should have 3 valid and 1 invalid samples
        self.assertEqual(report.valid_samples, 3)  # 3 samples pass format validation
        self.assertEqual(report.invalid_samples, 1)  # 1 sample fails format validation
        
        # Check recommendations
        self.assertGreater(len(report.recommendations), 0)
    
    def test_create_validation_rules(self):
        """Test validation rules creation."""
        # Test reasoning rules
        reasoning_rules = create_validation_rules("reasoning")
        self.assertIn("format_compliance", reasoning_rules)
        self.assertIn("text_length", reasoning_rules)
        
        # Test code rules
        code_rules = create_validation_rules("code")
        self.assertIn("code_content", code_rules)
        self.assertIn("text_length", code_rules)
        
        # Test math rules
        math_rules = create_validation_rules("math")
        self.assertIn("mathematical_content", math_rules)
        self.assertIn("format_compliance", math_rules)
    
    def test_validate_open_r1_dataset(self):
        """Test convenience validation function."""
        report = validate_open_r1_dataset(self.dataset, "reasoning")
        self.assertIsInstance(report, DataQualityReport)
        self.assertGreater(len(report.validation_results), 0)


class TestConfigurationManagement(unittest.TestCase):
    """Test configuration management utilities."""
    
    def setUp(self):
        """Set up test configuration."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.temp_dir) / "configs"
        self.config_dir.mkdir()
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_model_config_creation(self):
        """Test ModelConfig creation and auto-detection."""
        config = ModelConfig(model_name_or_path="test-model")
        
        self.assertEqual(config.model_name_or_path, "test-model")
        self.assertTrue(config.trust_remote_code)
        
        # Test torch_dtype auto-detection
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.is_bf16_supported', return_value=True):
                config = ModelConfig(model_name_or_path="test-model")
                self.assertEqual(config.torch_dtype, "bfloat16")
    
    def test_training_config_conversion(self):
        """Test TrainingConfig to TrainingArguments conversion."""
        config = TrainingConfig(
            learning_rate=1e-4,
            num_train_epochs=5,
            per_device_train_batch_size=8
        )
        
        training_args = config.to_training_arguments("test_output")
        
        self.assertEqual(training_args.learning_rate, 1e-4)
        self.assertEqual(training_args.num_train_epochs, 5)
        self.assertEqual(training_args.per_device_train_batch_size, 8)
    
    def test_open_r1_config_creation(self):
        """Test OpenR1Config creation and validation."""
        model_config = ModelConfig(model_name_or_path="test-model")
        training_config = TrainingConfig()
        dataset_config = DatasetConfig(dataset_name="test-dataset")
        evaluation_config = EvaluationConfig()
        system_config = SystemConfig()
        
        config = OpenR1Config(
            model=model_config,
            training=training_config,
            dataset=dataset_config,
            evaluation=evaluation_config,
            system=system_config,
            experiment_name="test-experiment"
        )
        
        self.assertEqual(config.experiment_name, "test-experiment")
        self.assertEqual(config.model.model_name_or_path, "test-model")
        self.assertEqual(config.dataset.dataset_name, "test-dataset")
        
        # Test validation
        issues = config.validate()
        self.assertEqual(len(issues), 0)
    
    def test_config_serialization(self):
        """Test configuration serialization and deserialization."""
        model_config = ModelConfig(model_name_or_path="test-model")
        training_config = TrainingConfig(learning_rate=1e-4)
        dataset_config = DatasetConfig(dataset_name="test-dataset")
        evaluation_config = EvaluationConfig()
        system_config = SystemConfig()
        
        config = OpenR1Config(
            model=model_config,
            training=training_config,
            dataset=dataset_config,
            evaluation=evaluation_config,
            system=system_config
        )
        
        # Test YAML serialization
        yaml_path = self.config_dir / "test_config.yaml"
        config.save(yaml_path)
        
        # Test loading
        loaded_config = OpenR1Config.from_file(yaml_path)
        self.assertEqual(loaded_config.model.model_name_or_path, "test-model")
        self.assertEqual(loaded_config.training.learning_rate, 1e-4)
    
    def test_config_manager(self):
        """Test ConfigManager functionality."""
        manager = ConfigManager(self.config_dir)
        
        # Test template creation
        sft_config = manager.create_template_config("test-sft", "sft")
        self.assertEqual(sft_config.experiment_name, "test-sft")
        
        # Test listing configs
        configs = manager.list_configs()
        self.assertIn("test-sft", configs)
        
        # Test loading config
        loaded_config = manager.load_config("test-sft")
        self.assertEqual(loaded_config.experiment_name, "test-sft")
    
    def test_create_default_config(self):
        """Test default configuration creation."""
        # Test SFT config
        sft_config = create_default_config("sft")
        self.assertIsInstance(sft_config, OpenR1Config)
        self.assertEqual(sft_config.training.num_train_epochs, 3)
        
        # Test GRPO config
        grpo_config = create_default_config("grpo")
        self.assertIsInstance(grpo_config, OpenR1Config)
        self.assertEqual(grpo_config.training.num_train_epochs, 1)
    
    def test_load_and_validate_config(self):
        """Test configuration loading and validation."""
        # Create a valid config
        config = create_default_config("sft")
        config_path = self.config_dir / "valid_config.yaml"
        config.save(config_path)
        
        # Test loading and validation
        loaded_config, issues = load_and_validate_config(config_path)
        self.assertIsInstance(loaded_config, OpenR1Config)
        self.assertEqual(len(issues), 0)
        
        # Test strict validation
        loaded_config, issues = load_and_validate_config(config_path, strict=True)
        self.assertIsInstance(loaded_config, OpenR1Config)
    
    def test_config_validation_issues(self):
        """Test configuration validation error detection."""
        # Create config with validation issues
        model_config = ModelConfig(model_name_or_path="")  # Invalid: empty name
        training_config = TrainingConfig(learning_rate=-1.0)  # Invalid: negative LR
        dataset_config = DatasetConfig(dataset_name="test-dataset")
        evaluation_config = EvaluationConfig()
        system_config = SystemConfig()
        
        config = OpenR1Config(
            model=model_config,
            training=training_config,
            dataset=dataset_config,
            evaluation=evaluation_config,
            system=system_config
        )
        
        # Test validation
        issues = config.validate()
        self.assertGreater(len(issues), 0)
        self.assertIn("Model name or path is required", issues)
        self.assertIn("Learning rate must be positive", issues)
    
    def test_effective_batch_size_calculation(self):
        """Test effective batch size calculation."""
        config = create_default_config("sft")
        
        # Modify for testing
        config.training.per_device_train_batch_size = 4
        config.training.gradient_accumulation_steps = 2
        config.system.num_gpus = 2
        
        effective_batch_size = config.get_effective_batch_size()
        self.assertEqual(effective_batch_size, 4 * 2 * 2)  # 16
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        config = create_default_config("sft")
        config.dataset.max_seq_length = 2048
        config.training.per_device_train_batch_size = 4
        config.training.gradient_accumulation_steps = 1
        config.system.num_gpus = 1
        
        memory_estimate = config.get_estimated_memory_usage()
        
        self.assertIn("total", memory_estimate)
        self.assertGreater(memory_estimate["total"], 0)


if __name__ == "__main__":
    unittest.main()

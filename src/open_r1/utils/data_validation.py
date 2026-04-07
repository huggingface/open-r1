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

"""Data validation and quality check utilities for Open-R1."""

import re
import json
from typing import Any, Dict, List, Optional, Set, Tuple, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np
from datasets import Dataset, Features, Value, Sequence


@dataclass
class ValidationResult:
    """Result of a data validation check."""
    
    is_valid: bool
    message: str
    details: Dict[str, Any]
    error_count: int = 0
    warning_count: int = 0
    
    def __str__(self):
        status = "✓ VALID" if self.is_valid else "✗ INVALID"
        return f"{status}: {self.message}"


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    
    total_samples: int
    valid_samples: int
    invalid_samples: int
    validation_results: List[ValidationResult]
    quality_score: float
    recommendations: List[str]
    
    def __str__(self):
        return (
            f"Data Quality Report:\n"
            f"  Total Samples: {self.total_samples}\n"
            f"  Valid Samples: {self.valid_samples}\n"
            f"  Invalid Samples: {self.invalid_samples}\n"
            f"  Quality Score: {self.quality_score:.2%}\n"
            f"  Issues Found: {len([r for r in self.validation_results if not r.is_valid])}\n"
            f"  Recommendations: {len(self.recommendations)}"
        )


class DataValidator:
    """Main data validation class."""
    
    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self.validators = {}
        self._register_default_validators()
    
    def _register_default_validators(self):
        """Register default validation functions."""
        self.validators.update({
            "text_length": self._validate_text_length,
            "text_content": self._validate_text_content,
            "format_compliance": self._validate_format_compliance,
            "mathematical_content": self._validate_mathematical_content,
            "code_content": self._validate_code_content,
            "dataset_structure": self._validate_dataset_structure,
            "duplicate_detection": self._validate_duplicate_detection,
            "label_consistency": self._validate_label_consistency,
        })
    
    def validate_dataset(
        self,
        dataset: Dataset,
        validation_rules: Optional[Dict[str, Any]] = None,
        sample_size: Optional[int] = None
    ) -> DataQualityReport:
        """Validate a dataset according to specified rules.
        
        Args:
            dataset: HuggingFace dataset to validate
            validation_rules: Dictionary of validation rules
            sample_size: Number of samples to validate (None for all)
            
        Returns:
            DataQualityReport with validation results
        """
        if sample_size:
            dataset = dataset.select(range(min(sample_size, len(dataset))))
        
        validation_results = []
        total_samples = len(dataset)
        
        # Apply validation rules
        for rule_name, rule_config in (validation_rules or {}).items():
            if rule_name in self.validators:
                result = self.validators[rule_name](dataset, rule_config)
                validation_results.append(result)
        
        # Calculate overall validation status
        # A sample is considered valid if it passes ALL validation rules
        valid_samples = 0
        invalid_samples = 0
        
        # Check each sample against all validation rules
        for i in range(total_samples):
            sample_valid = True
            for result in validation_results:
                if not result.is_valid:
                    # Check if this sample is in the invalid samples list
                    if "invalid_samples" in result.details:
                        for invalid_sample in result.details["invalid_samples"]:
                            if invalid_sample.get("index") == i:
                                sample_valid = False
                                break
                    if not sample_valid:
                        break
            
            if sample_valid:
                valid_samples += 1
            else:
                invalid_samples += 1
        
        # Calculate quality score
        quality_score = valid_samples / total_samples if total_samples > 0 else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(validation_results)
        
        return DataQualityReport(
            total_samples=total_samples,
            valid_samples=valid_samples,
            invalid_samples=invalid_samples,
            validation_results=validation_results,
            quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _validate_text_length(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate text length constraints."""
        min_length = config.get("min_length", 1)
        max_length = config.get("max_length", float("inf"))
        text_column = config.get("text_column", "text")
        
        if text_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Text column '{text_column}' not found in dataset",
                details={"missing_column": text_column}
            )
        
        invalid_samples = []
        for i, text in enumerate(dataset[text_column]):
            if not isinstance(text, str):
                invalid_samples.append({"index": i, "value": text, "issue": "not_string"})
            elif len(text) < min_length:
                invalid_samples.append({"index": i, "length": len(text), "min_required": min_length, "issue": "too_short"})
            elif len(text) > max_length:
                invalid_samples.append({"index": i, "length": len(text), "max_allowed": max_length, "issue": "too_long"})
        
        is_valid = len(invalid_samples) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Text length validation: {len(invalid_samples)} samples failed",
            details={"invalid_samples": invalid_samples, "min_length": min_length, "max_length": max_length},
            error_count=len(invalid_samples)
        )
    
    def _validate_text_content(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate text content quality."""
        text_column = config.get("text_column", "text")
        required_patterns = config.get("required_patterns", [])
        forbidden_patterns = config.get("forbidden_patterns", [])
        
        if text_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Text column '{text_column}' not found in dataset",
                details={"missing_column": text_column}
            )
        
        invalid_samples = []
        for i, text in enumerate(dataset[text_column]):
            if not isinstance(text, str):
                invalid_samples.append({"index": i, "value": text, "issue": "not_string"})
                continue
            
            # Check required patterns
            for pattern in required_patterns:
                if not re.search(pattern, text, re.IGNORECASE):
                    invalid_samples.append({"index": i, "pattern": pattern, "issue": "missing_required_pattern"})
            
            # Check forbidden patterns
            for pattern in forbidden_patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    invalid_samples.append({"index": i, "pattern": pattern, "issue": "forbidden_pattern_found"})
        
        is_valid = len(invalid_samples) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Text content validation: {len(invalid_samples)} samples failed",
            details={"invalid_samples": invalid_samples},
            error_count=len(invalid_samples)
        )
    
    def _validate_format_compliance(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate format compliance (e.g., think/answer tags)."""
        text_column = config.get("text_column", "text")
        required_format = config.get("required_format", "think_answer")
        
        if text_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Text column '{text_column}' not found in dataset",
                details={"missing_column": text_column}
            )
        
        invalid_samples = []
        for i, text in enumerate(dataset[text_column]):
            if not isinstance(text, str):
                invalid_samples.append({"index": i, "value": text, "issue": "not_string"})
                continue
            
            if required_format == "think_answer":
                # Check for think/answer format
                think_pattern = r"<think>.*?</think>"
                answer_pattern = r"<answer>.*?</answer>"
                
                has_think = bool(re.search(think_pattern, text, re.DOTALL))
                has_answer = bool(re.search(answer_pattern, text, re.DOTALL))
                
                if not has_think or not has_answer:
                    invalid_samples.append({
                        "index": i,
                        "has_think": has_think,
                        "has_answer": has_answer,
                        "issue": "missing_required_tags"
                    })
        
        is_valid = len(invalid_samples) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Format compliance validation: {len(invalid_samples)} samples failed",
            details={"invalid_samples": invalid_samples, "required_format": required_format},
            error_count=len(invalid_samples)
        )
    
    def _validate_mathematical_content(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate mathematical content quality."""
        text_column = config.get("text_column", "text")
        math_patterns = config.get("math_patterns", [r"\\boxed\{.*?\}", r"\\frac\{.*?\}\{.*?\}"])
        
        if text_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Text column '{text_column}' not found in dataset",
                details={"missing_column": text_column}
            )
        
        invalid_samples = []
        for i, text in enumerate(dataset[text_column]):
            if not isinstance(text, str):
                invalid_samples.append({"index": i, "value": text, "issue": "not_string"})
                continue
            
            # Check for mathematical content
            has_math = any(re.search(pattern, text) for pattern in math_patterns)
            if not has_math:
                invalid_samples.append({"index": i, "issue": "no_mathematical_content"})
        
        is_valid = len(invalid_samples) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Mathematical content validation: {len(invalid_samples)} samples failed",
            details={"invalid_samples": invalid_samples, "math_patterns": math_patterns},
            error_count=len(invalid_samples)
        )
    
    def _validate_code_content(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate code content quality."""
        text_column = config.get("text_column", "text")
        code_languages = config.get("code_languages", ["python", "cpp", "java"])
        
        if text_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Text column '{text_column}' not found in dataset",
                details={"missing_column": text_column}
            )
        
        invalid_samples = []
        for i, text in enumerate(dataset[text_column]):
            if not isinstance(text, str):
                invalid_samples.append({"index": i, "value": text, "issue": "not_string"})
                continue
            
            # Check for code blocks
            code_block_pattern = r"```(\w+)?\n(.*?)\n```"
            code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
            
            if not code_blocks:
                invalid_samples.append({"index": i, "issue": "no_code_blocks"})
            else:
                # Validate code language if specified
                for lang, code in code_blocks:
                    if lang and lang.lower() not in [l.lower() for l in code_languages]:
                        invalid_samples.append({
                            "index": i,
                            "language": lang,
                            "allowed_languages": code_languages,
                            "issue": "unsupported_language"
                        })
        
        is_valid = len(invalid_samples) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Code content validation: {len(invalid_samples)} samples failed",
            details={"invalid_samples": invalid_samples, "code_languages": code_languages},
            error_count=len(invalid_samples)
        )
    
    def _validate_dataset_structure(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate dataset structure and schema."""
        required_columns = config.get("required_columns", [])
        column_types = config.get("column_types", {})
        
        # Check required columns
        missing_columns = [col for col in required_columns if col not in dataset.column_names]
        if missing_columns:
            return ValidationResult(
                is_valid=False,
                message=f"Missing required columns: {missing_columns}",
                details={"missing_columns": missing_columns, "available_columns": dataset.column_names}
            )
        
        # Check column types
        invalid_types = []
        for column, expected_type in column_types.items():
            if column in dataset.column_names:
                actual_type = type(dataset[column][0]).__name__ if len(dataset) > 0 else "unknown"
                if actual_type != expected_type:
                    invalid_types.append({
                        "column": column,
                        "expected": expected_type,
                        "actual": actual_type
                    })
        
        is_valid = len(invalid_types) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Dataset structure validation: {len(invalid_types)} type mismatches",
            details={"invalid_types": invalid_types, "required_columns": required_columns},
            error_count=len(invalid_types)
        )
    
    def _validate_duplicate_detection(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Detect duplicate samples in the dataset."""
        text_column = config.get("text_column", "text")
        similarity_threshold = config.get("similarity_threshold", 0.95)
        
        if text_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Text column '{text_column}' not found in dataset",
                details={"missing_column": text_column}
            )
        
        # Simple duplicate detection based on exact text matching
        texts = dataset[text_column]
        unique_texts = set()
        duplicates = []
        
        for i, text in enumerate(texts):
            if text in unique_texts:
                duplicates.append({"index": i, "text": text[:100] + "..." if len(text) > 100 else text})
            else:
                unique_texts.add(text)
        
        is_valid = len(duplicates) == 0
        return ValidationResult(
            is_valid=is_valid,
            message=f"Duplicate detection: {len(duplicates)} duplicates found",
            details={"duplicates": duplicates, "total_samples": len(texts), "unique_samples": len(unique_texts)},
            error_count=len(duplicates)
        )
    
    def _validate_label_consistency(
        self,
        dataset: Dataset,
        config: Dict[str, Any]
    ) -> ValidationResult:
        """Validate label consistency and distribution."""
        label_column = config.get("label_column", "label")
        expected_labels = config.get("expected_labels", [])
        min_samples_per_label = config.get("min_samples_per_label", 1)
        
        if label_column not in dataset.column_names:
            return ValidationResult(
                is_valid=False,
                message=f"Label column '{label_column}' not found in dataset",
                details={"missing_column": label_column}
            )
        
        # Count label distribution
        label_counts = defaultdict(int)
        invalid_labels = []
        
        for i, label in enumerate(dataset[label_column]):
            if expected_labels and label not in expected_labels:
                invalid_labels.append({"index": i, "label": label, "expected": expected_labels})
            label_counts[label] += 1
        
        # Check minimum samples per label
        insufficient_labels = [
            label for label, count in label_counts.items()
            if count < min_samples_per_label
        ]
        
        issues = len(invalid_labels) + len(insufficient_labels)
        is_valid = issues == 0
        
        return ValidationResult(
            is_valid=is_valid,
            message=f"Label consistency validation: {issues} issues found",
            details={
                "label_distribution": dict(label_counts),
                "invalid_labels": invalid_labels,
                "insufficient_labels": insufficient_labels,
                "expected_labels": expected_labels,
                "min_samples_per_label": min_samples_per_label
            },
            error_count=issues
        )
    
    def _generate_recommendations(self, validation_results: List[ValidationResult]) -> List[str]:
        """Generate recommendations based on validation results."""
        recommendations = []
        
        for result in validation_results:
            if not result.is_valid:
                if "missing_column" in result.details:
                    recommendations.append(f"Add missing column: {result.details['missing_column']}")
                
                if "invalid_samples" in result.details:
                    count = len(result.details["invalid_samples"])
                    if count > 0:
                        recommendations.append(f"Review and fix {count} invalid samples")
                
                if "duplicates" in result.details:
                    count = len(result.details["duplicates"])
                    if count > 0:
                        recommendations.append(f"Remove {count} duplicate samples")
                
                if "insufficient_labels" in result.details:
                    labels = result.details["insufficient_labels"]
                    recommendations.append(f"Collect more samples for labels: {labels}")
        
        return list(set(recommendations))  # Remove duplicates


def create_validation_rules(
    task_type: str,
    **kwargs
) -> Dict[str, Any]:
    """Create validation rules for specific task types.
    
    Args:
        task_type: Type of task ("reasoning", "code", "math", "general")
        **kwargs: Additional configuration parameters
        
    Returns:
        Dictionary of validation rules
    """
    base_rules = {
        "dataset_structure": {
            "required_columns": ["text"],
            "column_types": {"text": "str"}
        }
    }
    
    if task_type == "reasoning":
        base_rules.update({
            "format_compliance": {
                "text_column": "text",
                "required_format": "think_answer"
            },
            "text_length": {
                "text_column": "text",
                "min_length": 10,
                "max_length": 10000
            }
        })
    
    elif task_type == "code":
        base_rules.update({
            "code_content": {
                "text_column": "text",
                "code_languages": ["python", "cpp", "java", "javascript"]
            },
            "text_length": {
                "text_column": "text",
                "min_length": 20,
                "max_length": 50000
            }
        })
    
    elif task_type == "math":
        base_rules.update({
            "mathematical_content": {
                "text_column": "text",
                "math_patterns": [r"\\boxed\{.*?\}", r"\\frac\{.*?\}\{.*?\}", r"\\sqrt\{.*?\}"]
            },
            "format_compliance": {
                "text_column": "text",
                "required_format": "think_answer"
            }
        })
    
    # Override with custom kwargs
    for key, value in kwargs.items():
        if key in base_rules:
            base_rules[key].update(value)
        else:
            base_rules[key] = value
    
    return base_rules


def validate_open_r1_dataset(
    dataset: Dataset,
    task_type: str = "general",
    **kwargs
) -> DataQualityReport:
    """Convenience function to validate an Open-R1 dataset.
    
    Args:
        dataset: Dataset to validate
        task_type: Type of task for validation rules
        **kwargs: Additional validation parameters
        
    Returns:
        DataQualityReport with validation results
    """
    validator = DataValidator()
    rules = create_validation_rules(task_type, **kwargs)
    return validator.validate_dataset(dataset, rules)

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

"""Performance monitoring and metrics collection utilities for Open-R1."""

import time
import psutil
import threading
from collections import defaultdict, deque
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from functools import wraps

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    wandb = None


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Timing metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration: Optional[float] = None
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    
    # GPU metrics (if available)
    gpu_memory_mb: Dict[int, float] = field(default_factory=dict)
    gpu_utilization: Dict[int, float] = field(default_factory=dict)
    
    # Throughput metrics
    samples_per_second: float = 0.0
    tokens_per_second: float = 0.0
    
    # Custom metrics
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = self.end_time - self.start_time
    
    def update_memory(self):
        """Update current memory usage."""
        process = psutil.Process()
        memory_info = process.memory_info()
        self.current_memory_mb = memory_info.rss / 1024 / 1024
        self.peak_memory_mb = max(self.peak_memory_mb, self.current_memory_mb)
    
    def update_gpu_metrics(self):
        """Update GPU metrics if available."""
        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    memory_allocated = torch.cuda.memory_allocated(i) / 1024 / 1024
                    self.gpu_memory_mb[i] = memory_allocated
                    
                    # Note: GPU utilization requires nvidia-smi or similar
                    # This is a placeholder for future implementation
                    self.gpu_utilization[i] = 0.0
                except Exception:
                    pass
    
    def finalize(self):
        """Finalize metrics collection."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.update_memory()
        self.update_gpu_metrics()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for logging."""
        metrics_dict = {
            "duration_seconds": self.duration,
            "peak_memory_mb": self.peak_memory_mb,
            "current_memory_mb": self.current_memory_mb,
            "samples_per_second": self.samples_per_second,
            "tokens_per_second": self.tokens_per_second,
        }
        
        # Add GPU metrics
        for gpu_id, memory in self.gpu_memory_mb.items():
            metrics_dict[f"gpu_{gpu_id}_memory_mb"] = memory
        
        # Add custom metrics
        metrics_dict.update(self.custom_metrics)
        
        return metrics_dict


class PerformanceMonitor:
    """Main performance monitoring class."""
    
    def __init__(self, enable_wandb: bool = True, log_interval: float = 60.0):
        self.enable_wandb = enable_wandb and WANDB_AVAILABLE
        self.log_interval = log_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.active_monitors: Dict[str, PerformanceMetrics] = {}
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_monitoring = threading.Event()
        
        # Performance counters
        self.counters = defaultdict(int)
        self.timers = defaultdict(list)
        
        # Start background monitoring
        self._start_background_monitoring()
    
    def _start_background_monitoring(self):
        """Start background monitoring thread."""
        if self._monitoring_thread is None or not self._monitoring_thread.is_alive():
            self._stop_monitoring.clear()
            self._monitoring_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitoring_thread.start()
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._stop_monitoring.wait(self.log_interval):
            try:
                self._log_current_metrics()
            except Exception as e:
                print(f"Error in performance monitoring: {e}")
    
    def _log_current_metrics(self):
        """Log current metrics to W&B if enabled."""
        if not self.enable_wandb or not WANDB_AVAILABLE or wandb.run is None:
            return
        
        # Log system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory_percent = psutil.virtual_memory().percent
        
        wandb.log({
            "system/cpu_percent": cpu_percent,
            "system/memory_percent": memory_percent,
            "system/active_monitors": len(self.active_monitors),
        })
        
        # Log active monitor metrics
        for monitor_name, metrics in self.active_monitors.items():
            if metrics.duration:
                wandb.log({
                    f"monitors/{monitor_name}/duration": metrics.duration,
                    f"monitors/{monitor_name}/memory_mb": metrics.current_memory_mb,
                })
    
    def start_monitor(self, name: str) -> PerformanceMetrics:
        """Start monitoring a new operation."""
        metrics = PerformanceMetrics()
        self.active_monitors[name] = metrics
        return metrics
    
    def stop_monitor(self, name: str) -> Optional[PerformanceMetrics]:
        """Stop monitoring an operation and return final metrics."""
        if name in self.active_monitors:
            metrics = self.active_monitors[name]
            metrics.finalize()
            self.metrics_history.append(metrics)
            del self.active_monitors[name]
            return metrics
        return None
    
    @contextmanager
    def monitor(self, name: str):
        """Context manager for monitoring operations."""
        metrics = self.start_monitor(name)
        try:
            yield metrics
        finally:
            self.stop_monitor(name)
    
    def increment_counter(self, name: str, value: int = 1):
        """Increment a performance counter."""
        self.counters[name] += value
    
    def record_timer(self, name: str, duration: float):
        """Record a timing measurement."""
        self.timers[name].append(duration)
    
    def get_counter_stats(self) -> Dict[str, int]:
        """Get current counter values."""
        return dict(self.counters)
    
    def get_timer_stats(self) -> Dict[str, Dict[str, float]]:
        """Get timer statistics."""
        stats = {}
        for name, times in self.timers.items():
            if times:
                stats[name] = {
                    "count": len(times),
                    "total": sum(times),
                    "mean": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return stats
    
    def log_metrics_to_wandb(self, step: Optional[int] = None):
        """Log all current metrics to Weights & Biases."""
        if not self.enable_wandb or not WANDB_AVAILABLE or wandb.run is None:
            return
        
        # Log counter stats
        counter_stats = self.get_counter_stats()
        for name, value in counter_stats.items():
            wandb.log({f"counters/{name}": value}, step=step)
        
        # Log timer stats
        timer_stats = self.get_timer_stats()
        for name, stats in timer_stats.items():
            for stat_name, stat_value in stats.items():
                wandb.log({f"timers/{name}/{stat_name}": stat_value}, step=step)
        
        # Log system metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        wandb.log({
            "system/cpu_percent": cpu_percent,
            "system/memory_percent": memory.percent,
            "system/memory_available_gb": memory.available / 1024 / 1024 / 1024,
            "system/memory_used_gb": memory.used / 1024 / 1024 / 1024,
        }, step=step)
    
    def stop(self):
        """Stop the performance monitor."""
        self._stop_monitoring.set()
        if self._monitoring_thread and self._monitoring_thread.is_alive():
            self._monitoring_thread.join(timeout=5.0)


def monitor_performance(name: str, enable_wandb: bool = True):
    """Decorator to monitor function performance."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            monitor = PerformanceMonitor(enable_wandb=enable_wandb)
            with monitor.monitor(name):
                result = func(*args, **kwargs)
                return result
        return wrapper
    return decorator


class TrainingMetricsCollector:
    """Specialized metrics collector for training operations."""
    
    def __init__(self, model_name: str, task_name: str):
        self.model_name = model_name
        self.task_name = task_name
        self.metrics = defaultdict(list)
        self.current_step = 0
        
    def log_training_step(
        self,
        loss: float,
        learning_rate: float,
        gradient_norm: Optional[float] = None,
        step_time: Optional[float] = None,
        memory_usage: Optional[float] = None,
        **additional_metrics
    ):
        """Log metrics for a training step."""
        step_metrics = {
            "step": self.current_step,
            "loss": loss,
            "learning_rate": learning_rate,
            "timestamp": time.time(),
        }
        
        if gradient_norm is not None:
            step_metrics["gradient_norm"] = gradient_norm
        
        if step_time is not None:
            step_metrics["step_time"] = step_time
        
        if memory_usage is not None:
            step_metrics["memory_usage_mb"] = memory_usage
        
        step_metrics.update(additional_metrics)
        
        # Store metrics
        for key, value in step_metrics.items():
            self.metrics[key].append(value)
        
        # Log to W&B if available
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(step_metrics, step=self.current_step)
        
        self.current_step += 1
    
    def log_evaluation_metrics(self, metrics: Dict[str, float], split: str = "validation"):
        """Log evaluation metrics."""
        eval_metrics = {
            f"eval/{split}/{key}": value
            for key, value in metrics.items()
        }
        
        if WANDB_AVAILABLE and wandb.run is not None:
            wandb.log(eval_metrics, step=self.current_step)
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get a summary of training metrics."""
        summary = {}
        
        for metric_name, values in self.metrics.items():
            if metric_name in ["step", "timestamp"]:
                continue
            
            if isinstance(values[0], (int, float)):
                summary[f"{metric_name}_mean"] = sum(values) / len(values)
                summary[f"{metric_name}_min"] = min(values)
                summary[f"{metric_name}_max"] = max(values)
                summary[f"{metric_name}_latest"] = values[-1]
        
        return summary


def create_training_metrics_collector(model_name: str, task_name: str) -> TrainingMetricsCollector:
    """Factory function to create a training metrics collector."""
    return TrainingMetricsCollector(model_name, task_name)


# Global performance monitor instance
_global_monitor: Optional[PerformanceMonitor] = None


def get_global_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor


def set_global_monitor(monitor: PerformanceMonitor):
    """Set the global performance monitor instance."""
    global _global_monitor
    _global_monitor = monitor

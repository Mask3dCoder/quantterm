"""
Observability utilities for QuantTerm.
Provides metrics collection, structured logging, and telemetry.
"""
from contextlib import contextmanager
from typing import Dict, Any, Optional, List
import time
import functools
import logging
import asyncio


class MetricsCollector:
    """
    Prometheus-style metrics for operational visibility.
    """
    def __init__(self):
        self.counters: Dict[str, int] = {}
        self.gauges: Dict[str, float] = {}
        self.histograms: Dict[str, List[float]] = {}
        self.timers: Dict[str, List[float]] = {}
    
    def increment(self, name: str, labels: Optional[Dict[str, str]] = None, value: int = 1):
        """Increment a counter metric."""
        key = self._format_key(name, labels)
        self.counters[key] = self.counters.get(key, 0) + value
    
    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric."""
        key = self._format_key(name, labels)
        self.gauges[key] = value
    
    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation."""
        key = self._format_key(name, labels)
        if key not in self.histograms:
            self.histograms[key] = []
        self.histograms[key].append(value)
    
    @contextmanager
    def timer(self, name: str, labels: Optional[Dict[str, str]] = None):
        """Time a block of code."""
        start = time.time()
        try:
            yield
        finally:
            duration = time.time() - start
            self.histogram(name, duration, labels)
    
    def _format_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics."""
        return {
            'counters': self.counters,
            'gauges': self.gauges,
            'histograms': {
                k: {
                    'count': len(v),
                    'mean': sum(v) / len(v) if v else 0,
                    'min': min(v) if v else 0,
                    'max': max(v) if v else 0
                }
                for k, v in self.histograms.items()
            }
        }


# Global metrics instance
metrics = MetricsCollector()


def instrumented(func):
    """Decorator to automatically instrument function calls."""
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__
        try:
            result = await func(*args, **kwargs)
            metrics.increment(
                "function_calls_total",
                {"function": func_name, "status": "success"}
            )
            return result
        except Exception as e:
            metrics.increment(
                "function_calls_total",
                {"function": func_name, "status": "error", "error_type": type(e).__name__}
            )
            raise
        finally:
            duration = time.time() - start
            metrics.histogram(
                "function_duration_seconds",
                duration,
                {"function": func_name}
            )
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        func_name = func.__name__
        try:
            result = func(*args, **kwargs)
            metrics.increment(
                "function_calls_total",
                {"function": func_name, "status": "success"}
            )
            return result
        except Exception as e:
            metrics.increment(
                "function_calls_total",
                {"function": func_name, "status": "error", "error_type": type(e).__name__}
            )
            raise
        finally:
            duration = time.time() - start
            metrics.histogram(
                "function_duration_seconds",
                duration,
                {"function": func_name}
            )
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


# Structured logging setup
def setup_logging(level: str = "INFO"):
    """Configure structured logging."""
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    return logging.getLogger("quantterm")


# Create logger instance
logger = setup_logging()

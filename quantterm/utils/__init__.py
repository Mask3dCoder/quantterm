"""
QuantTerm utilities package.
Provides resilience, bulkhead, and telemetry utilities.
"""
from quantterm.utils.resilience import (
    CircuitBreaker,
    CircuitBreakerOpenError,
    CircuitState,
    RetryPolicy,
    resilient
)
from quantterm.utils.bulkhead import Bulkhead, YAHOO_BULKHEAD, OPTIONS_BULKHEAD
from quantterm.utils.telemetry import (
    MetricsCollector,
    metrics,
    instrumented,
    setup_logging,
    logger
)
from quantterm.utils.security import (
    SecurityError,
    InvalidModelError,
    safe_joblib_load,
    safe_joblib_dump,
    validate_model_file,
    validate_ticker,
    validate_portfolio_weights,
    is_allowed_class,
    ALLOWED_SKLEARN_CLASSES,
)

__all__ = [
    # Resilience
    'CircuitBreaker',
    'CircuitBreakerOpenError', 
    'CircuitState',
    'RetryPolicy',
    'resilient',
    # Bulkhead
    'Bulkhead',
    'YAHOO_BULKHEAD',
    'OPTIONS_BULKHEAD',
    # Telemetry
    'MetricsCollector',
    'metrics',
    'instrumented',
    'setup_logging',
    'logger',
    # Security
    'SecurityError',
    'InvalidModelError',
    'safe_joblib_load',
    'safe_joblib_dump',
    'validate_model_file',
    'validate_ticker',
    'validate_portfolio_weights',
    'is_allowed_class',
    'ALLOWED_SKLEARN_CLASSES',
]

"""Custom exceptions for QuantTerm."""


class QuantTermError(Exception):
    """Base exception for QuantTerm."""
    pass


class DataError(QuantTermError):
    """Data-related errors (missing data, invalid data, etc.)."""
    pass


class CalculationError(QuantTermError):
    """Calculation errors (numerical issues, overflow, etc.)."""
    pass


class ValidationError(QuantTermError):
    """Validation errors (invalid inputs, etc.)."""
    pass


class ConfigurationError(QuantTermError):
    """Configuration errors."""
    pass


class APIError(QuantTermError):
    """API-related errors."""
    pass


class DataProviderError(APIError):
    """Data provider errors."""
    pass


class NetworkError(APIError):
    """Network connectivity errors."""
    pass


class RateLimitError(APIError):
    """Rate limit exceeded."""
    pass


class AuthenticationError(APIError):
    """Authentication failed."""
    pass


class ModelError(QuantTermError):
    """Model-specific errors."""
    pass


class OptionPricingError(ModelError):
    """Option pricing errors."""
    pass


class OptimizationError(ModelError):
    """Optimization errors."""
    pass


class BacktestError(QuantTermError):
    """Backtesting errors."""
    pass


class RiskCalculationError(QuantTermError):
    """Risk calculation errors."""
    pass

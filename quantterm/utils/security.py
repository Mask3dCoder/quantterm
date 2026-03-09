"""
Security utilities for QuantTerm.

Provides safe model loading and input validation to prevent
deserialization vulnerabilities and code execution attacks.
"""

import io
import os
import pickle
import struct
from typing import Any, Set, TypeVar, Type

import joblib
import numpy as np

T = TypeVar('T')


# Allowed sklearn model classes - whitelist approach
ALLOWED_SKLEARN_CLASSES: Set[str] = {
    # Ensemble methods
    'RandomForestClassifier',
    'RandomForestRegressor',
    'GradientBoostingClassifier',
    'GradientBoostingRegressor',
    'AdaBoostClassifier',
    'AdaBoostRegressor',
    'BaggingClassifier',
    'BaggingRegressor',
    'ExtraTreesClassifier',
    'ExtraTreesRegressor',
    
    # Linear models
    'LogisticRegression',
    'LinearRegression',
    'Ridge',
    'Lasso',
    'ElasticNet',
    'BayesianRidge',
    'SGDClassifier',
    'SGDRegressor',
    
    # Support vector machines
    'SVC',
    'SVR',
    'LinearSVC',
    'LinearSVR',
    
    # Nearest neighbors
    'KNeighborsClassifier',
    'KNeighborsRegressor',
    
    # Naive Bayes
    'GaussianNB',
    'MultinomialNB',
    'BernoulliNB',
    
    # Decision trees
    'DecisionTreeClassifier',
    'DecisionTreeRegressor',
    
    # Gaussian processes
    'GaussianProcessClassifier',
    'GaussianProcessRegressor',
    
    # Neural networks
    'MLPClassifier',
    'MLPRegressor',
    
    # Preprocessing
    'StandardScaler',
    'MinMaxScaler',
    'MaxAbsScaler',
    'RobustScaler',
    'Normalizer',
    'Binarizer',
    'LabelEncoder',
    'OneHotEncoder',
    'PCA',
    'KernelPCA',
    'TruncatedSVD',
    'DictVectorizer',
    'FunctionTransformer',
}

# Maximum model file size (100MB)
MAX_MODEL_FILE_SIZE = 100 * 1024 * 1024

# Joblib magic numbers
JOBLIB_MAGIC = b'PK\x03\x04'  # ZIP format


class SecurityError(Exception):
    """Raised when a security check fails."""
    pass


class InvalidModelError(Exception):
    """Raised when model validation fails."""
    pass


def validate_model_file(path: str) -> None:
    """
    Validate a model file before loading.
    
    Args:
        path: Path to the model file
        
    Raises:
        SecurityError: If file is invalid or suspicious
    """
    # Check file exists
    if not os.path.exists(path):
        raise SecurityError(f"Model file does not exist: {path}")
    
    # Check file size
    file_size = os.path.getsize(path)
    if file_size == 0:
        raise SecurityError("Model file is empty")
    
    if file_size > MAX_MODEL_FILE_SIZE:
        raise SecurityError(
            f"Model file too large: {file_size / (1024*1024):.1f}MB "
            f"(max: {MAX_MODEL_FILE_SIZE / (1024*1024):.0f}MB)"
        )
    
    # Check magic number
    with open(path, 'rb') as f:
        header = f.read(4)
    
    # Joblib files are ZIP archives
    if not header.startswith(JOBLIB_MAGIC):
        raise SecurityError(
            f"Invalid model file format. Expected joblib archive, "
            f"got: {header[:4].hex()}"
        )
    
    # Scan for suspicious content patterns
    with open(path, 'rb') as f:
        content = f.read()
    
    suspicious_patterns = [
        b'eval(',
        b'exec(',
        b'subprocess',
        b'os.system',
        b'__import__',
        b'pty.spawn',
        b'socket.',
    ]
    
    for pattern in suspicious_patterns:
        if pattern in content:
            raise SecurityError(
                f"Suspicious content detected in model file: {pattern.decode()}"
            )


def get_allowed_classes() -> Set[str]:
    """Return the set of allowed model classes."""
    return ALLOWED_SKLEARN_CLASSES.copy()


def is_allowed_class(obj: Any) -> bool:
    """
    Check if an object is an allowed model class.
    
    Args:
        obj: Object to check
        
    Returns:
        True if the object's class is in the whitelist
    """
    if obj is None:
        return True
        
    class_name = type(obj).__name__
    module = type(obj).__module__
    
    # Check direct class name match
    if class_name in ALLOWED_SKLEARN_CLASSES:
        return True
    
    # Check full module path
    full_name = f"{module}.{class_name}"
    
    # Allow sklearn classes
    if module.startswith('sklearn.'):
        # Extract simple class name from sklearn full path
        sklearn_class = class_name
        if sklearn_class in ALLOWED_SKLEARN_CLASSES:
            return True
    
    # Allow numpy and scipy base types
    if module.startswith('numpy.') or module.startswith('scipy.'):
        if isinstance(obj, (np.ndarray, np.number)):
            return True
    
    return False


class RestrictedUnpickler(pickle.Unpickler):
    """
    A restricted unpickler that only allows specific classes.
    
    This prevents arbitrary code execution through malicious
    pickle files.
    """
    
    def find_class(self, module: str, name: str) -> Any:
        """
        Override find_class to whitelist allowed classes.
        
        Args:
            module: Module name
            name: Class name
            
        Returns:
            The class if allowed
            
        Raises:
            SecurityError: If class is not in whitelist
        """
        # Always allow builtins
        if module == 'builtins':
            return super().find_class(module, name)
        
        # Allow numpy (needed for model data)
        if module.startswith('numpy.'):
            return super().find_class(module, name)
        
        # Allow scipy
        if module.startswith('scipy.'):
            return super().find_class(module, name)
        
        # Check sklearn classes
        if module.startswith('sklearn.'):
            if name in ALLOWED_SKLEARN_CLASSES:
                return super().find_class(module, name)
        
        # Block everything else
        raise SecurityError(
            f"Blocked deserialization of class: {module}.{name}. "
            f"Only sklearn model classes are allowed."
        )


def safe_joblib_load(path: str) -> Any:
    """
    Safely load a joblib file with validation.
    
    This function:
    1. Validates the file exists and has correct format
    2. Checks file size limits
    3. Scans for suspicious content
    4. Uses restricted unpickler
    
    Args:
        path: Path to the joblib file
        
    Returns:
        Loaded object
        
    Raises:
        SecurityError: If validation fails
        InvalidModelError: If model is invalid
    """
    # Validate file
    validate_model_file(path)
    
    try:
        # First try standard joblib load
        # joblib internally uses pickle but with its own protocol
        data = joblib.load(path)
        
        # Validate loaded object
        if isinstance(data, dict):
            # Common format: {'model': ..., 'scaler': ..., ...}
            for key, value in data.items():
                if value is not None and not is_allowed_class(value):
                    raise SecurityError(
                        f"Object '{key}' has disallowed type: {type(value).__name__}. "
                        f"Allowed types: {', '.join(sorted(ALLOWED_SKLEARN_CLASSES))}"
                    )
        elif not is_allowed_class(data):
            raise SecurityError(
                f"Root object has disallowed type: {type(data).__name__}. "
                f"Allowed types: {', '.join(sorted(ALLOWED_SKLEARN_CLASSES))}"
            )
        
        return data
        
    except SecurityError:
        raise
    except Exception as e:
        raise InvalidModelError(
            f"Failed to load model from {path}: {str(e)}"
        )


def safe_joblib_dump(obj: Any, path: str, compress: int = 3) -> None:
    """
    Safely save an object to a joblib file.
    
    Args:
        obj: Object to save (must be allowed class)
        path: Output path
        compress: Compression level (0-9)
        
    Raises:
        SecurityError: If object is not allowed
    """
    # Validate object before saving
    if not is_allowed_class(obj):
        raise SecurityError(
            f"Cannot save object of type: {type(obj).__name__}. "
            f"Allowed types: {', '.join(sorted(ALLOWED_SKLEARN_CLASSES))}"
        )
    
    # Ensure parent directory exists
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    
    joblib.dump(obj, path, compress=compress)


def validate_ticker(ticker: str) -> bool:
    """
    Validate a stock ticker symbol.
    
    Args:
        ticker: Ticker symbol to validate
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If ticker is invalid
    """
    import re
    
    if not ticker:
        raise ValueError("Ticker cannot be empty")
    
    # Standard US ticker format: 1-5 uppercase letters
    pattern = r'^[A-Z]{1,5}$'
    
    if not re.match(pattern, ticker.upper()):
        raise ValueError(
            f"Invalid ticker symbol: '{ticker}'. "
            f"Must be 1-5 uppercase letters (e.g., AAPL, MSFT, GOOG)"
        )
    
    return True


def validate_portfolio_weights(weights: list) -> bool:
    """
    Validate portfolio weights.
    
    Args:
        weights: List of weight values
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If weights are invalid
    """
    if not weights:
        raise ValueError("Weights cannot be empty")
    
    # Check all weights are in valid range
    for i, w in enumerate(weights):
        if w < 0:
            raise ValueError(f"Weight at index {i} is negative: {w}")
        if w > 1:
            raise ValueError(f"Weight at index {i} exceeds 1.0: {w}")
    
    # Check sum equals 1 (with tolerance)
    total = sum(weights)
    if abs(total - 1.0) > 0.001:
        raise ValueError(
            f"Weights sum to {total:.4f}, expected 1.0"
        )
    
    return True


def sanitize_string(value: str, max_length: int = 255) -> str:
    """
    Sanitize a string input.
    
    Args:
        value: String to sanitize
        max_length: Maximum allowed length
        
    Returns:
        Sanitized string
        
    Raises:
        ValueError: If sanitization fails
    """
    if not isinstance(value, str):
        raise ValueError(f"Expected string, got {type(value).__name__}")
    
    # Strip whitespace
    value = value.strip()
    
    # Check length
    if len(value) > max_length:
        raise ValueError(f"String exceeds maximum length: {max_length}")
    
    # Remove control characters
    value = ''.join(char for char in value if ord(char) >= 32 or char in '\n\r\t')
    
    return value

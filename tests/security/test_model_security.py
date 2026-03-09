"""
Security tests for model loading functionality.

Tests that:
1. Valid sklearn models load correctly
2. Malicious pickle files are rejected
3. File size limits are enforced
4. Suspicious content is detected
"""

import os
import pickle
import tempfile
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from quantterm.utils.security import (
    SecurityError,
    InvalidModelError,
    validate_model_file,
    safe_joblib_load,
    safe_joblib_dump,
    is_allowed_class,
    validate_ticker,
    validate_portfolio_weights,
    MAX_MODEL_FILE_SIZE,
    ALLOWED_SKLEARN_CLASSES,
)


class TestAllowedClasses:
    """Test allowed class detection."""
    
    def test_random_forest_allowed(self):
        model = RandomForestClassifier(n_estimators=10)
        assert is_allowed_class(model)
    
    def test_standard_scaler_allowed(self):
        scaler = StandardScaler()
        assert is_allowed_class(scaler)
    
    def test_logistic_regression_allowed(self):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        assert is_allowed_class(model)
    
    def test_numpy_array_allowed(self):
        arr = np.array([1, 2, 3])
        assert is_allowed_class(arr)
    
    def test_malicious_class_blocked(self):
        # Simulate a malicious class
        class MaliciousClass:
            def __reduce__(self):
                # This would execute arbitrary code
                return (os.system, ("echo pwned",))
        
        assert not is_allowed_class(MaliciousClass())


class TestFileValidation:
    """Test file validation."""
    
    def test_nonexistent_file_raises_error(self):
        with pytest.raises(SecurityError, match="does not exist"):
            validate_model_file("/nonexistent/path/model.pkl")
    
    def test_empty_file_raises_error(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
        
        try:
            with pytest.raises(SecurityError, match="empty"):
                validate_model_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_oversized_file_raises_error(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            # Write more than MAX_MODEL_FILE_SIZE bytes
            f.write(b'x' * (MAX_MODEL_FILE_SIZE + 1))
        
        try:
            with pytest.raises(SecurityError, match="too large"):
                validate_model_file(temp_path)
        finally:
            os.unlink(temp_path)
    
    def test_invalid_magic_number_raises_error(self):
        with tempfile.NamedTemporaryFile(delete=False) as f:
            temp_path = f.name
            f.write(b'NOT_A_VALID_FILE')  # Invalid magic number
        
        try:
            with pytest.raises(SecurityError, match="Invalid model file format"):
                validate_model_file(temp_path)
        finally:
            os.unlink(temp_path)


class TestSafeJoblibLoad:
    """Test safe joblib loading."""
    
    def test_valid_model_loads(self):
        """Test that valid sklearn models load correctly."""
        # Create a valid model
        model = RandomForestClassifier(n_estimators=5, random_state=42)
        scaler = StandardScaler()
        
        # Create test data
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        
        # Fit
        model.fit(X, y)
        scaler.fit(X)
        
        # Save with safe function
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            safe_joblib_dump({
                'model': model,
                'scaler': scaler,
                'feature_names': ['feature1', 'feature2'],
                'model_type': 'random_forest'
            }, temp_path)
            
            # Load with safe function
            data = safe_joblib_load(temp_path)
            
            assert isinstance(data['model'], RandomForestClassifier)
            assert isinstance(data['scaler'], StandardScaler)
            assert data['feature_names'] == ['feature1', 'feature2']
            assert data['model_type'] == 'random_forest'
        
        finally:
            os.unlink(temp_path)
    
    def test_malicious_pickle_rejected(self):
        """Test that malicious pickle files are rejected."""
        
        # Create a malicious pickle that would execute code
        class MaliciousPayload:
            def __reduce__(self):
                # This would execute: print("EXPLOIT!")
                return (print, ("EXPLOIT!",))
        
        # Try to save malicious content using pickle (not joblib)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # Write a malicious pickle directly
            with open(temp_path, 'wb') as f:
                pickle.dump({'model': MaliciousPayload()}, f)
            
            # Should be caught by validation
            with pytest.raises(SecurityError, match="Invalid model file format"):
                safe_joblib_load(temp_path)
        
        finally:
            os.unlink(temp_path)
    
    def test_suspicious_content_rejected(self):
        """Test that files with suspicious patterns are rejected."""
        
        # Create a joblib file with suspicious content
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as f:
            temp_path = f.name
        
        try:
            # Write a valid joblib header + suspicious content
            with open(temp_path, 'wb') as f:
                f.write(b'PK\x03\x04')  # ZIP magic
                f.write(b'\x00' * 100)  # Padding
                f.write(b'eval(')  # Suspicious pattern
            
            with pytest.raises(SecurityError, match="Suspicious content"):
                validate_model_file(temp_path)
        
        finally:
            os.unlink(temp_path)


class TestInputValidation:
    """Test input validation functions."""
    
    def test_valid_ticker(self):
        assert validate_ticker("AAPL")
        assert validate_ticker("MSFT")
        assert validate_ticker("aapl")  # Should be uppercased
    
    def test_invalid_ticker_too_long(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            validate_ticker("TOOLONG")
    
    def test_invalid_ticker_special_chars(self):
        with pytest.raises(ValueError, match="Invalid ticker"):
            validate_ticker("AAPL!")
    
    def test_valid_weights(self):
        assert validate_portfolio_weights([0.5, 0.3, 0.2])
        assert validate_portfolio_weights([1.0])
    
    def test_invalid_weights_negative(self):
        with pytest.raises(ValueError, match="negative"):
            validate_portfolio_weights([-0.5, 1.5])
    
    def test_invalid_weights_sum(self):
        with pytest.raises(ValueError, match="sum to"):
            validate_portfolio_weights([0.5, 0.3])  # Sum to 0.8


class TestAllowedClassesList:
    """Test the allowed classes list."""
    
    def test_contains_expected_classes(self):
        assert 'RandomForestClassifier' in ALLOWED_SKLEARN_CLASSES
        assert 'LogisticRegression' in ALLOWED_SKLEARN_CLASSES
        assert 'StandardScaler' in ALLOWED_SKLEARN_CLASSES
        assert 'PCA' in ALLOWED_SKLEARN_CLASSES
    
    def test_no_dangerous_classes(self):
        dangerous = ['os', 'sys', 'subprocess', 'socket', 'pty']
        for cls in ALLOWED_SKLEARN_CLASSES:
            for dangerous_word in dangerous:
                assert dangerous_word not in cls.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

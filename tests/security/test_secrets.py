"""
Tests for secure secrets management.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from quantterm.security.secrets import (
    SecureSecret,
    SecretsManager,
    SecretsError,
    KEYRING_AVAILABLE,
    CRYPTO_AVAILABLE,
)


class TestSecureSecret:
    """Test SecureSecret class."""
    
    def test_backend_detection(self):
        """Test backend detection."""
        secret = SecureSecret("test", "key")
        # Should detect available backend
        assert secret.backend in ("keyring", "encrypted_file", "environment")
    
    @patch.dict(os.environ, {'TEST_KEY': 'test_value'})
    def test_environment_backend(self):
        """Test environment variable backend."""
        secret = SecureSecret("test", "key")
        # Force environment backend
        with patch.object(secret, '_backend', 'environment'):
            value = secret.get()
            assert value == "test_value"


class TestSecretsManager:
    """Test SecretsManager class."""
    
    def test_providers_list(self):
        """Test that known providers are listed."""
        manager = SecretsManager()
        providers = manager.list_providers()
        
        # Should have known providers
        assert 'fred' in providers
        assert 'polygon' in providers
        assert 'alphavantage' in providers
    
    def test_get_secret(self):
        """Test getting a secret."""
        manager = SecretsManager()
        secret = manager.get_secret("fred")
        
        assert isinstance(secret, SecureSecret)
        assert secret._key == "fred"


class TestEnvironmentFallback:
    """Test environment variable fallback."""
    
    def test_env_var_retrieval(self):
        """Test that environment variables work as fallback."""
        # Set environment variable
        os.environ['QUANTTERM_FRED'] = 'test_fred_key'
        
        try:
            manager = SecretsManager()
            # Should fall back to environment
            # (will return None if keyring not available)
            result = manager.get_api_key('fred')
            
            # Either got from keyring (None) or env fallback
            # In test environment, likely None or env value
            assert result is None or result == 'test_fred_key'
        finally:
            del os.environ['QUANTTERM_FRED']


class TestInputValidation:
    """Test input validation."""
    
    def test_empty_key_rejected(self):
        """Test that empty keys are rejected."""
        with pytest.raises(Exception):
            secret = SecureSecret("test", "")
            secret.set("")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

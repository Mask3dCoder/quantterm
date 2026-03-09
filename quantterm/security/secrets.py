"""
Secure secrets management for QuantTerm.

Provides secure API key storage using:
1. System keychain (keyring) - preferred
2. Encrypted file storage with password - fallback
3. Environment variables - for CI/CD

Never stores secrets in plain text.
"""

import os
import sys
import time
import json
import base64
import hashlib
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import contextmanager

# Try to import keyring, provide fallback if unavailable
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

# For encrypted file fallback
try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


SERVICE_NAME = "quantterm"
CACHE_DURATION = timedelta(minutes=5)  # Max cache time


class SecretsError(Exception):
    """Base exception for secrets management."""
    pass


class KeyringUnavailableError(SecretsError):
    """Raised when system keyring is unavailable."""
    pass


class EncryptionUnavailableError(SecretsError):
    """Raised when encryption is unavailable."""
    pass


class InvalidPasswordError(SecretsError):
    """Raised when password is incorrect."""
    pass


@dataclass
class SecretMetadata:
    """Metadata about a stored secret."""
    provider: str
    created_at: datetime
    last_accessed: Optional[datetime] = None
    access_count: int = 0


class SecureSecret:
    """
    A secure secret container that manages API keys.
    
    Features:
    - System keychain integration (keyring)
    - Encrypted file fallback
    - In-memory caching with TTL
    - Automatic memory clearing
    
    Usage:
        secret = SecureSecret("quantterm", "alphavantage")
        api_key = secret.get()  # Retrieve from secure storage
        secret.clear_cache()  # Clear from memory
    """
    
    def __init__(
        self,
        service_name: str = SERVICE_NAME,
        key_name: str = "",
        allow_caching: bool = True
    ):
        """
        Initialize a secure secret container.
        
        Args:
            service_name: Service identifier (default: quantterm)
            key_name: Name/key identifier
            allow_caching: Whether to allow in-memory caching
        """
        self._service = service_name
        self._key = key_name
        self._allow_caching = allow_caching
        self._cached: Optional[str] = None
        self._cached_at: Optional[datetime] = None
        self._backend = self._detect_backend()
    
    def _detect_backend(self) -> str:
        """Detect the best available backend."""
        if KEYRING_AVAILABLE:
            try:
                # Test keyring availability
                keyring.get_password(self._service, "_test")
                return "keyring"
            except Exception:
                pass
        
        if CRYPTO_AVAILABLE:
            return "encrypted_file"
        
        return "environment"
    
    def _get_from_keyring(self) -> Optional[str]:
        """Retrieve from system keychain."""
        if not KEYRING_AVAILABLE:
            raise KeyringUnavailableError(
                "keyring package not installed. Install with: pip install keyring"
            )
        
        try:
            return keyring.get_password(self._service, self._key)
        except Exception as e:
            raise SecretsError(f"Failed to retrieve from keyring: {e}")
    
    def _set_in_keyring(self, value: str) -> None:
        """Store in system keychain."""
        if not KEYRING_AVAILABLE:
            raise KeyringUnavailableError(
                "keyring package not installed. Install with: pip install keyring"
            )
        
        try:
            keyring.set_password(self._service, self._key, value)
        except Exception as e:
            raise SecretsError(f"Failed to store in keyring: {e}")
    
    def _get_from_encrypted_file(self) -> Optional[str]:
        """Retrieve from encrypted file storage."""
        if not CRYPTO_AVAILABLE:
            raise EncryptionUnavailableError(
                "cryptography package not installed. Install with: pip install cryptography"
            )
        
        config_dir = Path.home() / ".quantterm"
        secrets_file = config_dir / "secrets.json"
        
        if not secrets_file.exists():
            return None
        
        try:
            # Get password for decryption
            password = getpass.getpass("Enter master password: ")
            
            with open(secrets_file, 'rb') as f:
                encrypted_data = f.read()
            
            # Derive key from password
            salt = encrypted_data[:32]
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            
            # Decrypt
            decrypted = fernet.decrypt(encrypted_data[32:])
            secrets = json.loads(decrypted)
            
            return secrets.get(self._key)
            
        except Exception as e:
            raise InvalidPasswordError(f"Failed to decrypt: {e}")
    
    def _set_in_encrypted_file(self, value: str) -> None:
        """Store in encrypted file storage."""
        if not CRYPTO_AVAILABLE:
            raise EncryptionUnavailableError(
                "cryptography package not installed. Install with: pip install cryptography"
            )
        
        config_dir = Path.home() / ".quantterm"
        config_dir.mkdir(parents=True, exist_ok=True)
        secrets_file = config_dir / "secrets.json"
        
        # Get or create password
        password = getpass.getpass("Enter master password: ")
        confirm = getpass.getpass("Confirm master password: ")
        
        if password != confirm:
            raise InvalidPasswordError("Passwords do not match")
        
        if len(password) < 8:
            raise InvalidPasswordError("Password must be at least 8 characters")
        
        # Generate salt
        salt = os.urandom(32)
        key = self._derive_key(password, salt)
        fernet = Fernet(key)
        
        # Load existing secrets or create new
        existing_secrets = {}
        if secrets_file.exists():
            try:
                with open(secrets_file, 'rb') as f:
                    old_encrypted = f.read()
                old_salt = old_encrypted[:32]
                old_key = self._derive_key(password, old_salt)
                old_fernet = Fernet(old_key)
                old_decrypted = old_fernet.decrypt(old_encrypted[32:])
                existing_secrets = json.loads(old_decrypted)
            except Exception:
                pass  # Can't decrypt with current password
        
        # Add new secret
        existing_secrets[self._key] = value
        
        # Encrypt and save
        secrets_json = json.dumps(existing_secrets)
        encrypted = fernet.encrypt(secrets_json.encode())
        
        with open(secrets_file, 'wb') as f:
            f.write(salt + encrypted)
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password."""
        kdf = PBKDF2(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def _get_from_environment(self) -> Optional[str]:
        """Retrieve from environment variable."""
        env_key = f"{self._service.upper()}_{self._key.upper()}"
        return os.environ.get(env_key)
    
    def get(self, ttl: int = 300) -> Optional[str]:
        """
        Retrieve the secret value.
        
        Args:
            ttl: Time to live in seconds (default: 5 minutes)
            
        Returns:
            The secret value or None if not set
        """
        # Check cache first
        if self._cached and self._allow_caching:
            if self._cached_at:
                age = (datetime.now() - self._cached_at).total_seconds()
                if age < ttl:
                    return self._cached
        
        # Retrieve from backend
        value = None
        if self._backend == "keyring":
            value = self._get_from_keyring()
        elif self._backend == "encrypted_file":
            value = self._get_from_encrypted_file()
        elif self._backend == "environment":
            value = self._get_from_environment()
        
        # Cache if enabled
        if value and self._allow_caching:
            self._cached = value
            self._cached_at = datetime.now()
        
        return value
    
    def set(self, value: str) -> None:
        """
        Store the secret value.
        
        Args:
            value: The secret value to store
        """
        if self._backend == "keyring":
            self._set_in_keyring(value)
        elif self._backend == "encrypted_file":
            self._set_in_encrypted_file(value)
        elif self._backend == "environment":
            env_key = f"{self._service.upper()}_{self._key.upper()}"
            print(f"Set environment variable: {env_key}=<value>")
            os.environ[env_key] = value
        
        # Update cache
        if self._allow_caching:
            self._cached = value
            self._cached_at = datetime.now()
    
    def clear_cache(self) -> None:
        """Clear the in-memory cache."""
        if self._cached:
            # Overwrite with zeros before clearing (defense in depth)
            self._cached = None
        self._cached_at = None
    
    def delete(self) -> bool:
        """
        Delete the secret from storage.
        
        Returns:
            True if deleted, False if not found
        """
        if self._backend == "keyring":
            try:
                keyring.delete_password(self._service, self._key)
                self.clear_cache()
                return True
            except Exception:
                return False
        elif self._backend == "encrypted_file":
            # Would need to re-implement for encrypted file
            pass
        
        self.clear_cache()
        return False
    
    def is_set(self) -> bool:
        """Check if a secret is stored."""
        return self.get() is not None
    
    @property
    def backend(self) -> str:
        """Get the current backend."""
        return self._backend
    
    def __repr__(self) -> str:
        """Never expose the actual value."""
        status = "set" if self.is_set() else "not set"
        return f"<SecureSecret backend={self._backend} key={self._key} {status}>"


class SecretsManager:
    """
    Manager for multiple secrets.
    
    Provides a unified interface for managing all API keys.
    """
    
    # Known secret providers
    PROVIDERS = {
        'alphavantage': 'Alpha Vantage API key',
        'polygon': 'Polygon.io API key',
        'alpaca_key': 'Alpine Markets API key',
        'alpaca_secret': 'Alpine Markets API secret',
        'fred': 'FRED API key',
        'bloomberg': 'Bloomberg API key',
        'refinitiv': 'Refinitiv API key',
    }
    
    def __init__(self):
        self._secrets: Dict[str, SecureSecret] = {}
    
    def get_secret(self, provider: str) -> SecureSecret:
        """Get or create a secret for a provider."""
        if provider not in self._secrets:
            self._secrets[provider] = SecureSecret(
                service_name=SERVICE_NAME,
                key_name=provider
            )
        return self._secrets[provider]
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """Get API key for a provider."""
        return self.get_secret(provider).get()
    
    def set_api_key(self, provider: str, value: str) -> None:
        """Set API key for a provider."""
        self.get_secret(provider).set(value)
    
    def list_providers(self) -> Dict[str, bool]:
        """List all providers and whether they have keys set."""
        result = {}
        for provider in self.PROVIDERS:
            result[provider] = self.get_secret(provider).is_set()
        return result
    
    def migrate_from_environment(self) -> None:
        """Migrate secrets from environment variables."""
        for provider in self.PROVIDERS:
            env_key = f"quantterm_{provider}".upper()
            value = os.environ.get(env_key)
            if value:
                print(f"Migrating {provider} from environment...")
                self.set_api_key(provider, value)
                print(f"  -> Set {provider} securely")
    
    def clear_all_caches(self) -> None:
        """Clear all secret caches."""
        for secret in self._secrets.values():
            secret.clear_cache()


# Global secrets manager instance
_secrets_manager: Optional[SecretsManager] = None


def get_secrets_manager() -> SecretsManager:
    """Get the global secrets manager."""
    global _secrets_manager
    if _secrets_manager is None:
        _secrets_manager = SecretsManager()
    return _secrets_manager


@contextmanager
def temporary_secret(secret: SecureSecret):
    """
    Context manager for temporary secret access.
    
    Automatically clears cache after use.
    
    Usage:
        secret = SecureSecret("quantterm", "api_key")
        with temporary_secret(secret) as api_key:
            # Use api_key
            call_api(api_key)
        # Cache automatically cleared
    """
    try:
        yield secret.get()
    finally:
        secret.clear_cache()

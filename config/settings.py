"""Configuration settings for QuantTerm.

SECURITY: API keys are retrieved on-demand from secure storage.
Use 'quantterm config set-key' to configure API keys.
"""
from pathlib import Path
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings with environment variable validation."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = "QuantTerm"
    app_version: str = "0.1.0"
    debug: bool = False

    # Paths
    data_dir: Path = Field(default=Path.home() / ".quantterm" / "data")
    cache_dir: Path = Field(default=Path.home() / ".quantterm" / "cache")
    log_dir: Path = Field(default=Path.home() / ".quantterm" / "logs")

    # API Keys - These are deprecated in favor of secure storage
    # Use 'quantterm config set-key <provider>' to configure
    # These are kept for backward compatibility only
    _deprecated_keys = {
        'bloomberg_api_key': 'bloomberg',
        'refinitiv_api_key': 'refinitiv',
        'polygon_api_key': 'polygon',
        'alpaca_key': 'alpaca_key',
        'alpaca_secret': 'alpaca_secret',
        'fred_api_key': 'fred',
    }
    
    # These fields are kept for .env file backward compatibility
    # but will show a deprecation warning
    bloomberg_api_key: Optional[str] = None
    refinitiv_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    alpaca_key: Optional[str] = None
    alpaca_secret: Optional[str] = None
    fred_api_key: Optional[str] = None

    # Database
    redis_url: str = "redis://localhost:6379/0"
    postgres_url: Optional[str] = None
    timescaledb_url: Optional[str] = None

    # Risk Parameters
    default_var_confidence: float = 0.99
    default_var_horizon: int = 1  # days
    default_risk_free_rate: float = 0.05  # 5%

    # Calculation Settings
    mc_simulations: int = 100000
    numba_enabled: bool = True
    parallel_workers: int = 4

    # Trading Parameters
    default_commission: float = 0.001  # 0.1%
    default_slippage: float = 0.0005   # 0.05%

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a provider from secure storage.
        
        This method retrieves API keys from secure storage (keyring),
        falling back to environment variables or deprecated .env values.
        
        Args:
            provider: Provider name (e.g., 'fred', 'polygon')
            
        Returns:
            API key string or None if not configured
        """
        # Try secure storage first
        try:
            from quantterm.security import get_secrets_manager
            manager = get_secrets_manager()
            key = manager.get_api_key(provider)
            if key:
                return key
        except Exception:
            pass
        
        # Fallback to environment variables
        import os
        env_key = f"QUANTTERM_{provider.upper()}"
        env_value = os.environ.get(env_key)
        if env_value:
            return env_value
        
        # Fallback to deprecated .env values (with warning)
        deprecated_map = {
            'bloomberg': 'bloomberg_api_key',
            'refinitiv': 'refinitiv_api_key',
            'polygon': 'polygon_api_key',
            'alpaca_key': 'alpaca_key',
            'alpaca_secret': 'alpaca_secret',
            'fred': 'fred_api_key',
        }
        
        if provider in deprecated_map:
            attr = deprecated_map[provider]
            value = getattr(self, attr, None)
            if value:
                import warnings
                warnings.warn(
                    f"API key for {provider} is stored in plain text. "
                    f"Use 'quantterm config set-key {provider}' to migrate to secure storage.",
                    DeprecationWarning,
                    stacklevel=2
                )
                return value
        
        return None
    
    def __repr__(self) -> str:
        """Never expose API keys in repr."""
        # Return a safe representation without keys
        return "<Settings (api_keys_hidden)>"


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

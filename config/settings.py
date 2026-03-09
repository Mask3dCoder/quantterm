"""Configuration settings for QuantTerm."""
from pathlib import Path
from typing import Optional

from pydantic import Field
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

    # Data Providers API Keys (optional)
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


# Global settings instance
settings = Settings()


def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings

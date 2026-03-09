"""Integration tests for cached data handler.

These tests use mocking to avoid real Yahoo Finance API calls.
"""

import shutil
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from quantterm.backtesting.data_handler import DataHandler, IntradayDataHandler, MultiSymbolDataHandler
from quantterm.data.cache import DataCache, reset_cache


# Sample mock data for Yahoo Finance responses
MOCK_DAILY_DATA = pd.DataFrame({
    "Date": pd.date_range("2024-01-01", periods=20, freq="D"),
    "Open": [100 + i for i in range(20)],
    "High": [105 + i for i in range(20)],
    "Low": [95 + i for i in range(20)],
    "Close": [102 + i for i in range(20)],
    "Volume": [1000000 for _ in range(20)],
})


def create_mock_yfinance_download(dataframe):
    """Create a mock yfinance.download function that tracks call count."""
    mock_func = MagicMock(side_effect=lambda *args, **kwargs: dataframe.copy())
    return mock_func


@pytest.fixture
def cache_dir(tmp_path):
    """Create a temporary cache directory and return path."""
    cache_path = tmp_path / "cache"
    yield cache_path
    # Force close any remaining handles on Windows
    try:
        shutil.rmtree(cache_path, ignore_errors=True)
    except Exception:
        pass


class TestDataHandlerCaching:
    """Test caching in DataHandler."""

    def test_first_call_hits_api(self, cache_dir):
        """Test first call hits the API (no cache)."""
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            result = handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            
            # Should have called the API
            assert mock_download.called
            assert len(result) > 0
        
        cache.close()

    def test_second_call_hits_cache(self, cache_dir):
        """Test second identical call uses cache."""
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            # First call - hits API
            result1 = handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            first_call_count = mock_download.call_count
            
            # Second call - should hit cache
            result2 = handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            second_call_count = mock_download.call_count
            
            # Should not have called API again
            assert first_call_count == second_call_count
            # Results should be identical
            assert result1.equals(result2)
        
        cache.close()

    def test_cache_statistics(self, cache_dir):
        """Test cache statistics update correctly."""
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            # First call
            handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            
            # Second call - cache hit
            handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            
            stats = cache.get_stats()
            
            # Should have 1 hit, 1 miss (first call)
            assert stats.hits >= 1
            assert stats.entries >= 1
        
        cache.close()

    def test_different_params_different_cache(self, cache_dir):
        """Test different date ranges use different cache entries."""
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            # Different date ranges
            handler.get_bars("SPY", "2024-01-01", "2024-01-15")
            handler.get_bars("SPY", "2024-01-16", "2024-01-31")
            
            stats = cache.get_stats()
            
            # Should have 2 entries
            assert stats.entries >= 2
        
        cache.close()


class TestMultiSymbolDataHandlerCaching:
    """Test caching in MultiSymbolDataHandler."""

    def test_multi_symbol_caching(self, cache_dir):
        """Test multi-symbol handler uses caching."""
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = MultiSymbolDataHandler()
            handler._cache = cache
            
            # First call with multiple symbols
            result1 = handler.get_bars(["SPY", "AAPL"], "2024-01-01", "2024-01-31")
            first_call_count = mock_download.call_count
            
            # Second call - should use cache
            result2 = handler.get_bars(["SPY", "AAPL"], "2024-01-01", "2024-01-31")
            second_call_count = mock_download.call_count
            
            # Should not have called API again
            assert first_call_count == second_call_count
        
        cache.close()


class TestIntradayDataHandlerCaching:
    """Test caching in IntradayDataHandler."""

    def test_intraday_caching(self, cache_dir):
        """Test intraday handler uses caching."""
        intraday_mock = MOCK_DAILY_DATA.copy()
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(intraday_mock)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = IntradayDataHandler(interval="5m")
            handler._cache = cache
            
            # First call
            result1 = handler.get_bars("SPY", "2024-01-01", "2024-01-31", "5m")
            first_call_count = mock_download.call_count
            
            # Second call - should use cache
            result2 = handler.get_bars("SPY", "2024-01-01", "2024-01-31", "5m")
            second_call_count = mock_download.call_count
            
            # Should not have called API again
            assert first_call_count == second_call_count
        
        cache.close()


class TestPerformance:
    """Test caching performance."""

    def test_cache_speedup(self, cache_dir):
        """Test cache provides significant speedup."""
        # Create mock that simulates network delay
        def slow_download(*args, **kwargs):
            time.sleep(0.1)  # Simulate 100ms network delay (shorter for tests)
            return MOCK_DAILY_DATA.copy()
        
        mock_download = MagicMock(side_effect=slow_download)
        cache = DataCache(cache_dir=cache_dir)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            # First call - hits API (should take ~0.1s)
            start = time.time()
            handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            first_time = time.time() - start
            
            # Second call - hits cache (should be instant)
            start = time.time()
            handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            second_time = time.time() - start
            
            # Cache should be at least 10x faster
            assert second_time < first_time / 10
            assert second_time < 0.1  # Should be under 100ms
        
        cache.close()


class TestOfflineMode:
    """Test offline mode functionality."""

    def test_use_cache_false_disables(self, cache_dir):
        """Test use_cache=False disables caching."""
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler(use_cache=False)
            handler._cache = None
            
            # First call
            handler.get_bars("SPY", "2024-01-01", "2024-01-31", use_cache=False)
            first_count = mock_download.call_count
            
            # Second call - should still hit API
            handler.get_bars("SPY", "2024-01-01", "2024-01-31", use_cache=False)
            second_count = mock_download.call_count
            
            # Should have called API both times
            assert second_count == 2

    def test_cache_works_without_network(self, cache_dir):
        """Test cached data works when network is unavailable."""
        cache = DataCache(cache_dir=cache_dir)
        
        # First, populate the cache with mock data
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            handler.get_bars("SPY", "2024-01-01", "2024-01-31")
        
        # Now simulate offline - mock yfinance to raise exception
        def offline_download(*args, **kwargs):
            raise ConnectionError("Network unavailable")
        
        mock_offline = MagicMock(side_effect=offline_download)
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_offline
            
            handler = DataHandler()
            handler._cache = cache
            
            # This should work from cache even though network is down
            result = handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            
            # Should have returned cached data
            assert result is not None
            assert len(result) > 0
        
        cache.close()


class TestBackwardsCompatibility:
    """Test backwards compatibility with existing API."""

    def test_default_use_cache_true(self):
        """Test use_cache defaults to True."""
        handler = DataHandler()
        assert handler._use_cache is True

    def test_existing_api_unchanged(self, cache_dir):
        """Test existing method signatures work."""
        cache = DataCache(cache_dir=cache_dir)
        mock_download = create_mock_yfinance_download(MOCK_DAILY_DATA)
        
        with patch("quantterm.backtesting.data_handler.yf") as mock_yf:
            mock_yf.download = mock_download
            
            handler = DataHandler()
            handler._cache = cache
            
            # Call without use_cache parameter - should still work
            result = handler.get_bars("SPY", "2024-01-01", "2024-01-31")
            
            assert result is not None
            assert len(result) > 0
        
        cache.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

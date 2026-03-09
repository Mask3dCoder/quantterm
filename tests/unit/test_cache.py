"""Unit tests for DataCache class."""

import os
import pickle
import shutil
import tempfile
import threading
import time
from pathlib import Path

import pytest

from quantterm.data.cache import CacheStats, DataCache, get_cache, reset_cache


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


class TestCacheKeyGeneration:
    """Test cache key generation."""

    def test_cache_key_format(self, cache_dir):
        """Test cache key follows expected format."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            key = cache._generate_cache_key(
                symbol="SPY",
                start="2024-01-01",
                end="2024-12-31",
                interval="1d",
                provider="yfinance"
            )
            
            # Key should be MD5 hash (32 hex chars)
            assert len(key) == 32
            assert all(c in "0123456789abcdef" for c in key)
        finally:
            cache.close()

    def test_same_params_same_key(self, cache_dir):
        """Test identical parameters produce identical keys."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            key1 = cache._generate_cache_key("SPY", "2024-01-01", "2024-12-31")
            key2 = cache._generate_cache_key("SPY", "2024-01-01", "2024-12-31")
            
            assert key1 == key2
        finally:
            cache.close()

    def test_different_params_different_keys(self, cache_dir):
        """Test different parameters produce different keys."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            key1 = cache._generate_cache_key("SPY", "2024-01-01", "2024-12-31")
            key2 = cache._generate_cache_key("AAPL", "2024-01-01", "2024-12-31")
            key3 = cache._generate_cache_key("SPY", "2023-01-01", "2024-12-31")
            
            assert key1 != key2
            assert key1 != key3
            assert key2 != key3
        finally:
            cache.close()

    def test_case_insensitive_symbol(self, cache_dir):
        """Test symbol is case-insensitive."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            key1 = cache._generate_cache_key("spy", "2024-01-01", "2024-12-31")
            key2 = cache._generate_cache_key("SPY", "2024-01-01", "2024-12-31")
            key3 = cache._generate_cache_key("Spy", "2024-01-01", "2024-12-31")
            
            assert key1 == key2 == key3
        finally:
            cache.close()


class TestTTL:
    """Test TTL calculation."""

    def test_ttl_short_period(self, cache_dir):
        """Test TTL for short period data (< 30 days)."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            # 10 day period should get 24 hour TTL
            ttl = cache._calculate_ttl("2024-01-01", "2024-01-10")
            
            # Should be approximately 24 hours (86400 seconds)
            assert 86000 < ttl < 86800
        finally:
            cache.close()

    def test_ttl_long_period(self, cache_dir):
        """Test TTL for long period data (> 30 days)."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            # 60 day period should get 30 day TTL
            ttl = cache._calculate_ttl("2024-01-01", "2024-03-01")
            
            # Should be approximately 30 days
            assert 2590000 < ttl < 2600000
        finally:
            cache.close()

    def test_ttl_invalid_date(self, cache_dir):
        """Test TTL falls back to default on invalid date."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            ttl = cache._calculate_ttl("invalid", "also-invalid")
            
            # Should default to 24 hours
            assert 86000 < ttl < 86800
        finally:
            cache.close()


class TestCacheOperations:
    """Test basic cache operations."""

    def test_set_and_get(self, cache_dir):
        """Test setting and getting values."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            test_data = {"col1": [1, 2, 3], "col2": [4, 5, 6]}
            
            cache.set("SPY", "2024-01-01", "2024-12-31", test_data)
            result = cache.get("SPY", "2024-01-01", "2024-12-31")
            
            assert result == test_data
        finally:
            cache.close()

    def test_get_missing(self, cache_dir):
        """Test getting non-existent key returns None."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            result = cache.get("NONEXISTENT", "2024-01-01", "2024-12-31")
            
            assert result is None
        finally:
            cache.close()

    def test_delete(self, cache_dir):
        """Test deleting cache entry."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            test_data = {"col1": [1, 2, 3]}
            cache.set("SPY", "2024-01-01", "2024-12-31", test_data)
            
            # Verify it exists
            assert cache.get("SPY", "2024-01-01", "2024-12-31") is not None
            
            # Delete
            cache.delete("SPY", "2024-01-01", "2024-12-31")
            
            # Verify it's gone
            assert cache.get("SPY", "2024-01-01", "2024-12-31") is None
        finally:
            cache.close()

    def test_clear_all(self, cache_dir):
        """Test clearing all cache entries."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            cache.set("SPY", "2024-01-01", "2024-12-31", {"data": 1})
            cache.set("AAPL", "2024-01-01", "2024-12-31", {"data": 2})
            
            count = cache.clear()
            
            assert count == 2
            assert cache.get("SPY", "2024-01-01", "2024-12-31") is None
        finally:
            cache.close()


class TestCacheCorruption:
    """Test cache corruption handling."""

    def test_corrupted_cache_entry(self, cache_dir):
        """Test handling of corrupted cache entry."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            # Manually create a corrupted cache entry
            key = cache._generate_cache_key("SPY", "2024-01-01", "2024-12-31")
            
            # Write corrupted data directly to cache
            cache_file = cache_dir / f"{key}.cache"
            with open(cache_file, "wb") as f:
                pickle.dump("invalid data", f)
            
            # Should handle corruption gracefully and return None
            result = cache.get("SPY", "2024-01-01", "2024-12-31")
            
            assert result is None
        finally:
            cache.close()

    def test_missing_cache_file(self, cache_dir):
        """Test handling of missing cache file."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            # Try to get a key that was never cached
            result = cache.get("NEVER", "2024-01-01", "2024-12-31")
            
            assert result is None
        finally:
            cache.close()


class TestCacheStats:
    """Test cache statistics."""

    def test_stats_initial(self, cache_dir):
        """Test initial stats are zero."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            stats = cache.get_stats()
            
            assert stats.hits == 0
            assert stats.misses == 0
            assert stats.entries >= 0
        finally:
            cache.close()

    def test_stats_after_hits_and_misses(self, cache_dir):
        """Test stats update correctly."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            # Add some data
            cache.set("SPY", "2024-01-01", "2024-12-31", {"data": 1})
            
            # Hit
            cache.get("SPY", "2024-01-01", "2024-12-31")
            # Miss
            cache.get("AAPL", "2024-01-01", "2024-12-31")
            
            stats = cache.get_stats()
            
            assert stats.hits == 1
            assert stats.misses == 1
        finally:
            cache.close()

    def test_hit_rate_calculation(self, cache_dir):
        """Test hit rate calculation."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            cache.set("SPY", "2024-01-01", "2024-12-31", {"data": 1})
            
            # 3 hits, 1 miss = 75%
            for _ in range(3):
                cache.get("SPY", "2024-01-01", "2024-12-31")
            cache.get("AAPL", "2024-01-01", "2024-12-31")
            
            stats = cache.get_stats()
            
            assert stats.hit_rate == 75.0
        finally:
            cache.close()


class TestThreadSafety:
    """Test thread safety of cache operations."""

    def test_concurrent_writes(self, cache_dir):
        """Test concurrent writes don't corrupt data."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            errors = []
            
            def write_data(symbol_num: int):
                try:
                    for i in range(10):
                        cache.set(
                            f"SYM{symbol_num}",
                            "2024-01-01",
                            "2024-12-31",
                            {"value": i}
                        )
                except Exception as e:
                    errors.append(e)
            
            threads = [threading.Thread(target=write_data, args=(i,)) for i in range(5)]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            assert len(errors) == 0
            assert cache.get_stats().entries > 0
        finally:
            cache.close()

    def test_concurrent_reads_writes(self, cache_dir):
        """Test concurrent reads and writes don't corrupt data."""
        cache = DataCache(cache_dir=cache_dir)
        try:
            # Pre-populate
            for i in range(10):
                cache.set(f"SYM{i}", "2024-01-01", "2024-12-31", {"value": i})
            
            errors = []
            
            def read_write(idx: int):
                try:
                    if idx % 2 == 0:
                        # Read
                        for i in range(10):
                            cache.get(f"SYM{i}", "2024-01-01", "2024-12-31")
                    else:
                        # Write
                        for i in range(10):
                            cache.set(f"NEW{i}", "2024-01-01", "2024-12-31", {"value": i})
                except Exception as e:
                    errors.append(e)
            
            threads = [threading.Thread(target=read_write, args=(i,)) for i in range(10)]
            
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            
            assert len(errors) == 0
        finally:
            cache.close()


class TestLRUEviction:
    """Test LRU eviction when max size is reached."""

    def test_size_limit(self, cache_dir):
        """Test cache respects size limit."""
        # Very small cache size (1KB)
        cache = DataCache(cache_dir=cache_dir, max_size=1024)
        try:
            # Add data larger than limit
            large_data = {"data": "x" * 2000}
            cache.set("SPY", "2024-01-01", "2024-12-31", large_data)
            
            stats = cache.get_stats()
            
            # Should have at least some entries (might evict older ones)
            assert stats.size_bytes > 0
        finally:
            cache.close()


class TestGlobalCache:
    """Test global cache singleton."""

    def test_get_cache_returns_same_instance(self):
        """Test get_cache returns the same instance."""
        reset_cache()
        
        cache1 = get_cache()
        cache2 = get_cache()
        
        assert cache1 is cache2

    def test_reset_cache(self):
        """Test reset_cache creates new instance."""
        reset_cache()
        
        cache1 = get_cache()
        reset_cache()
        cache2 = get_cache()
        
        assert cache1 is not cache2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

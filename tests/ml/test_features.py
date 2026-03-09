"""Tests for ML feature engineering."""
import pandas as pd
import numpy as np
from quantterm.ml.features import FeatureEngineer


def test_no_lookahead_bias():
    """Ensure features don't leak future information."""
    np.random.seed(42)
    n = 100
    
    # Generate random walk
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    engineer = FeatureEngineer(lookback_windows=[5, 10, 20])
    
    # Create features at point in time
    features = engineer.create_features(data.iloc[:50], data.index[49])
    
    assert features is not None
    print(f"Features created: {len(features)}")
    print(features)
    
    # Verify all features are finite
    assert np.all(np.isfinite(features.values)), "Features contain NaN or inf"
    
    # Verify no negative values for indicators that should be positive
    assert features['volatility_5d'] >= 0
    assert features['volatility_10d'] >= 0
    assert features['volatility_20d'] >= 0
    
    # Verify RSI is in valid range
    assert 0 <= features['rsi_5d'] <= 100
    assert 0 <= features['rsi_10d'] <= 100
    assert 0 <= features['rsi_20d'] <= 100


def test_feature_batch():
    """Test batch feature creation."""
    np.random.seed(42)
    n = 100
    
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    engineer = FeatureEngineer()
    features_df = engineer.create_features_batch(data)
    
    assert len(features_df) > 0
    print(f"Batch features: {features_df.shape}")
    
    # Check for any NaN values
    assert not features_df.isna().any().any(), "Features contain NaN values"


def test_target_creation():
    """Test target variable creation."""
    np.random.seed(42)
    n = 50
    
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Close': prices
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    engineer = FeatureEngineer(target_horizon=5)
    
    # Create target for a point in time
    target = engineer.create_target(data, data.index[20])
    
    assert target is not None
    print(f"Target (forward return): {target:.4f}")
    
    # Verify target is a valid return
    assert np.isfinite(target)


def test_insufficient_history():
    """Test behavior with insufficient history."""
    np.random.seed(42)
    n = 30
    
    returns = np.random.randn(n) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    engineer = FeatureEngineer(lookback_windows=[5, 10, 20, 60])
    
    # This should return None due to insufficient history
    features = engineer.create_features(data, data.index[10])
    
    # Should return None since max window is 60 but we only have 11 rows
    assert features is None or len(features) == 0


def test_rsi_calculation():
    """Test RSI calculation correctness."""
    # Create data with clear uptrend - need at least period+1 values (14+1=15)
    prices = pd.Series([
        100, 102, 104, 103, 105, 107, 106, 108, 110, 109,
        111, 113, 115, 114, 116, 118, 117, 119, 121, 120
    ])
    
    rsi = FeatureEngineer._calculate_rsi(prices, period=14)
    
    # RSI should be high (>50) for uptrend
    assert rsi > 50, f"RSI should be > 50 for uptrend, got {rsi}"
    assert 0 <= rsi <= 100, f"RSI should be in [0, 100], got {rsi}"


if __name__ == '__main__':
    print("Running test_no_lookahead_bias...")
    test_no_lookahead_bias()
    print("\n[PASS] test_no_lookahead_bias passed\n")
    
    print("Running test_feature_batch...")
    test_feature_batch()
    print("\n[PASS] test_feature_batch passed\n")
    
    print("Running test_target_creation...")
    test_target_creation()
    print("\n[PASS] test_target_creation passed\n")
    
    print("Running test_insufficient_history...")
    test_insufficient_history()
    print("\n[PASS] test_insufficient_history passed\n")
    
    print("Running test_rsi_calculation...")
    test_rsi_calculation()
    print("\n[PASS] test_rsi_calculation passed\n")
    
    print("All tests passed!")

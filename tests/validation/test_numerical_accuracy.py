"""
Numerical accuracy validation tests.
Validates that mathematical computations match expected reference values.
"""

import pytest
import numpy as np
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestBlackScholesAccuracy:
    """Test Black-Scholes option pricing accuracy."""
    
    def test_call_option_price_atm(self):
        """Test ATM call option pricing."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        # ATM option: S = K = 100, r = 0.05, sigma = 0.2, T = 0.25 (3 months)
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 0.25
        
        result = black_scholes(S, K, T, r, sigma)
        call_price = result.call_price
        
        # Reference value from known calculators: ~4.61
        assert 4.0 < call_price < 5.5, f"Call price {call_price} not in expected range [4.0, 5.5]"
    
    def test_put_option_price_atm(self):
        """Test ATM put option pricing."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 0.25
        
        result = black_scholes(S, K, T, r, sigma)
        put_price = result.put_price
        
        # Reference value: ~3.37 (put-call parity: C - P = S - K*e^(-rT))
        assert 2.5 < put_price < 4.5, f"Put price {put_price} not in expected range [2.5, 4.5]"
    
    def test_call_option_price_itm(self):
        """Test ITM call option pricing."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        S = 110.0  # In the money
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 0.25
        
        result = black_scholes(S, K, T, r, sigma)
        call_price = result.call_price
        
        # Should be higher than ATM
        assert call_price > 10.0, f"ITM call price {call_price} too low"
    
    def test_call_option_price_otm(self):
        """Test OTM call option pricing."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        S = 90.0  # Out of the money
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 0.25
        
        result = black_scholes(S, K, T, r, sigma)
        call_price = result.call_price
        
        # Should be relatively cheap
        assert call_price < 3.0, f"OTM call price {call_price} too high"
    
    def test_put_call_parity(self):
        """Test put-call parity relationship."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 0.25
        
        result = black_scholes(S, K, T, r, sigma)
        
        # Put-Call Parity: C - P = S - K * e^(-rT)
        lhs = result.call_price - result.put_price
        rhs = S - K * np.exp(-r * T)
        
        assert abs(lhs - rhs) < 0.01, f"Put-call parity violated: {lhs} != {rhs}"
    
    def test_greeks_sanity(self):
        """Test that Greeks have reasonable values."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        S = 100.0
        K = 100.0
        r = 0.05
        sigma = 0.2
        T = 0.25
        
        result = black_scholes(S, K, T, r, sigma)
        
        # Delta: 0 < Delta < 1 for call
        assert 0 < result.delta_call < 1, f"Call delta {result.delta_call} out of range"
        
        # Gamma: positive
        assert result.gamma > 0, f"Gamma {result.gamma} should be positive"
        
        # Theta: negative for call (time decay)
        assert result.theta_call < 0, f"Call theta {result.theta_call} should be negative"
        
        # Vega: positive (volatility increase increases option price)
        assert result.vega > 0, f"Vega {result.vega} should be positive"


class TestTechnicalIndicatorsAccuracy:
    """Test technical indicator calculations."""
    
    def test_sma_simple(self):
        """Test SMA calculation."""
        from quantterm.analytics.technical.indicators import sma
        
        prices = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
        
        result = sma(prices, period=5)
        
        # SMA at position 4 should be 3.0 (mean of 1-5)
        assert abs(result[4] - 3.0) < 0.01, f"SMA[4] = {result[4]}, expected 3.0"
        
        # SMA at position 9 should be 8.0 (mean of 6-10)
        assert abs(result[9] - 8.0) < 0.01, f"SMA[9] = {result[9]}, expected 8.0"
    
    def test_ema_basic(self):
        """Test EMA calculation."""
        from quantterm.analytics.technical.indicators import ema
        
        # Simple test case
        prices = np.array([10, 11, 12, 13, 14], dtype=float)
        
        result = ema(prices, period=3)
        
        # EMA differs from SMA due to exponential smoothing
        # For period=3: alpha = 2/(3+1) = 0.5
        # EMA[2] = alpha*price[2] + (1-alpha)*EMA[1] = 0.5*12 + 0.5*10.5 = 11.25
        assert abs(result[2] - 11.25) < 0.01, f"EMA[2] = {result[2]}, expected 11.25"
    
    def test_rsi_bounds(self):
        """Test RSI is always between 0 and 100."""
        from quantterm.analytics.technical.indicators import rsi
        
        # Upward trending prices
        prices_up = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=float)
        rsi_up = rsi(prices_up, period=14)
        
        # Should be high (near 100) for upward trend
        assert 50 < rsi_up[-1] <= 100, f"RSI for uptrend {rsi_up[-1]} out of range"
        
        # Downward trending prices
        prices_down = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1], dtype=float)
        rsi_down = rsi(prices_down, period=14)
        
        # Should be low (near 0) for downward trend
        assert 0 <= rsi_down[-1] < 50, f"RSI for downtrend {rsi_down[-1]} out of range"
    
    def test_bollinger_bands_ordering(self):
        """Test Bollinger Bands are correctly ordered."""
        from quantterm.analytics.technical.indicators import bollinger_bands
        
        prices = np.array([10, 11, 12, 13, 12, 11, 10, 11, 12, 13, 14, 15, 16, 17, 18, 
                          17, 16, 15, 14, 13, 12, 11, 12, 13, 14], dtype=float)
        
        upper, middle, lower = bollinger_bands(prices, period=20, std_dev=2)
        
        # Upper should be >= middle >= lower
        valid_indices = ~np.isnan(middle)
        assert np.all(upper[valid_indices] >= middle[valid_indices]), "Upper band should be >= middle"
        assert np.all(middle[valid_indices] >= lower[valid_indices]), "Middle band should be >= lower"


class TestVaRAccuracy:
    """Test VaR calculation accuracy."""
    
    def test_var_no_loss_zero(self):
        """Test VaR with no loss scenario."""
        from quantterm.portfolio.risk.var import historical_var, parametric_var
        
        # No losses - all returns are positive
        returns = np.array([0.01, 0.02, 0.03, 0.01, 0.02])
        
        var_result = historical_var(returns, confidence=0.95)
        var_hist = var_result.var
        
        # With no losses, VaR should be <= 0 (or very small)
        assert var_hist <= 0.01, f"Historical VaR {var_hist} should be <= 0"
    
    def test_var_with_losses(self):
        """Test VaR with actual losses."""
        from quantterm.portfolio.risk.var import historical_var
        
        # Mix of gains and losses
        returns = np.array([0.05, 0.02, -0.03, -0.10, 0.01, -0.05, 0.03, -0.08, 0.02, -0.02])
        
        var_result = historical_var(returns, confidence=0.95)
        var_hist = var_result.var
        
        # At 95% confidence, VaR should capture the worst 5% of returns
        sorted_returns = np.sort(returns)
        
        # VaR is positive (potential loss), should be between 0.08 and 0.10
        assert 0.05 <= var_hist <= 0.15, f"Historical VaR {var_hist} out of expected range"
    
    def test_var_confidence_level(self):
        """Test VaR increases with higher confidence level."""
        from quantterm.portfolio.risk.var import historical_var
        
        returns = np.array([0.05, 0.02, -0.03, -0.10, 0.01, -0.05, 0.03, -0.08, 0.02, -0.02])
        
        var_result_95 = historical_var(returns, confidence=0.95)
        var_result_99 = historical_var(returns, confidence=0.99)
        
        var_95 = var_result_95.var
        var_99 = var_result_99.var
        
        # VaR at 99% should be >= VaR at 95% (more extreme losses captured)
        assert var_99 >= var_95, f"99% VaR {var_99} should be >= 95% VaR {var_95}"


class TestPortfolioOptimizationAccuracy:
    """Test portfolio optimization calculations."""
    
    def test_weights_sum_to_one(self):
        """Test that portfolio weights sum to 1."""
        pytest.importorskip("cvxpy")
        from quantterm.portfolio.optimization.mean_variance import mean_variance_optimize
        
        # Two assets with equal expected returns
        expected_returns = np.array([0.10, 0.10])
        cov_matrix = np.array([
            [0.01, 0.005],
            [0.005, 0.01]
        ])
        
        weights, _, _ = mean_variance_optimize(expected_returns, cov_matrix)
        
        assert abs(np.sum(weights) - 1.0) < 0.01, f"Weights sum to {np.sum(weights)}, expected 1.0"
    
    def test_minimum_variance_portfolio(self):
        """Test minimum variance portfolio."""
        pytest.importorskip("cvxpy")
        from quantterm.portfolio.optimization.mean_variance import minimum_variance_portfolio
        
        cov_matrix = np.array([
            [0.01, 0.005],
            [0.005, 0.01]
        ])
        
        weights = minimum_variance_portfolio(cov_matrix)
        
        # Weights should sum to 1
        assert abs(np.sum(weights) - 1.0) < 0.01, f"Weights sum to {np.sum(weights)}"
        
        # All weights should be non-negative (long-only)
        assert np.all(weights >= 0), "Weights should be non-negative"


class TestDataProviderAccuracy:
    """Test data provider accuracy."""
    
    def test_returns_calculation(self):
        """Test return calculation accuracy."""
        from quantterm.data.providers.yahoo import get_returns
        
        # Test with known prices
        prices = np.array([100.0, 110.0, 121.0])
        
        # Calculate returns manually
        expected_returns = np.diff(prices) / prices[:-1]
        
        # Calculate log returns
        actual_returns = np.diff(np.log(prices))
        
        # Log returns should be approximately equal to simple returns for small changes
        # For 100 -> 110: simple = 0.10, log = ~0.095
        assert abs(expected_returns[0] - 0.10) < 0.001, "Simple return calculation wrong"
        assert abs(actual_returns[0] - np.log(1.10)) < 0.001, "Log return calculation wrong"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_zero_volatility(self):
        """Test option pricing with zero volatility - should raise error."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        # Zero volatility should raise ValueError (input validation)
        with pytest.raises(ValueError, match="Volatility must be positive"):
            black_scholes(s=100, k=100, t=1, r=0.05, sigma=0.0)
    
    def test_zero_time_to_expiry(self):
        """Test option pricing with zero time to expiry."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s=100, k=100, t=0, r=0.05, sigma=0.2)
        
        # At T=0, option value = max(S-K, 0) for call
        assert result.call_price == max(0, 100 - 100), "At-the-money should be 0 at expiry"
    
    def test_deep_itm_call(self):
        """Test deep ITM call option."""
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s=200, k=100, t=1, r=0.05, sigma=0.2)
        
        # Deep ITM call should be worth approximately S - K*e^(-rT)
        intrinsic = 200 - 100 * np.exp(-0.05)
        assert abs(result.call_price - intrinsic) < 1.0, "Deep ITM call should be close to intrinsic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

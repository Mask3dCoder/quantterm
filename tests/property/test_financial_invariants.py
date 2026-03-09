"""
Property-based tests for financial mathematical invariants.
These catch bugs that example-based tests miss.
"""
import pytest
import numpy as np
from hypothesis import given, strategies as st, settings, assume, HealthCheck


def hypothesis_settings():
    """Configure hypothesis settings for all tests."""
    settings.register_profile("fast", max_examples=10, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    settings.load_profile("fast")


class TestFinancialInvariants:
    """
    Mathematical properties that must always hold.
    """
    
    @given(
        s=st.floats(min_value=1.0, max_value=10000.0),
        k=st.floats(min_value=1.0, max_value=10000.0),
        t=st.floats(min_value=0.01, max_value=5.0),
        r=st.floats(min_value=-0.1, max_value=0.3),
        sigma=st.floats(min_value=0.01, max_value=2.0)
    )
    @settings(max_examples=1000, deadline=None)
    def test_put_call_parity(self, s, k, t, r, sigma):
        """
        C - P = S - K*e^(-rT)
        This is a fundamental no-arbitrage condition.
        """
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s, k, t, r, sigma)
        
        lhs = result.call_price - result.put_price
        rhs = s - k * np.exp(-r * t)
        
        # Allow for numerical precision errors
        assert np.isclose(lhs, rhs, rtol=1e-5), \
            f"Put-call parity violated: {lhs:.8f} != {rhs:.8f}"
    
    @given(
        prices=st.lists(
            st.floats(min_value=1.0, max_value=1000.0),
            min_size=30,
            max_size=200
        ),
        period=st.integers(min_value=2, max_value=30)
    )
    @settings(max_examples=50, deadline=None)
    def test_rsi_bounds(self, prices, period):
        """
        RSI must always be between 0 and 100.
        """
        from quantterm.analytics.technical.indicators import rsi
        
        assume(len(prices) > period + 1)
        
        prices_arr = np.array(prices)
        result = rsi(prices_arr, period)
        
        valid_results = result[~np.isnan(result)]
        assert len(valid_results) > 0
        
        assert np.all(valid_results >= 0), f"RSI below 0: {np.min(valid_results)}"
        assert np.all(valid_results <= 100), f"RSI above 100: {np.max(valid_results)}"
    
    @given(
        returns=st.lists(
            st.floats(min_value=-0.5, max_value=0.5),
            min_size=100,
            max_size=500
        )
    )
    @settings(max_examples=100)
    def test_var_monotonicity(self, returns):
        """
        Higher confidence → higher VaR (more extreme quantile).
        """
        from quantterm.portfolio.risk.var import historical_var
        
        returns_arr = np.array(returns)
        assume(np.std(returns_arr) > 0.001)  # Ensure non-constant
        
        var_90 = historical_var(returns_arr, confidence=0.90)
        var_95 = historical_var(returns_arr, confidence=0.95)
        var_99 = historical_var(returns_arr, confidence=0.99)
        
        # VaR is typically negative (loss), so 99% should be more negative (larger in absolute value)
        # Actually, VaRResult.var is positive for losses, so 99% should be >= 95% >= 90%
        assert var_99.var >= var_95.var >= var_90.var, \
            f"VaR not monotonic: 90%={var_90.var}, 95%={var_95.var}, 99%={var_99.var}"
    
    @given(
        n=st.integers(min_value=2, max_value=10)
    )
    @settings(max_examples=20, deadline=None)
    def test_minimum_variance_weights_sum_to_one(self, n):
        """
        Minimum variance portfolio weights must sum to 1.
        """
        pytest.importorskip("cvxpy")
        
        from quantterm.portfolio.optimization.mean_variance import minimum_variance_portfolio
        
        # Create valid covariance matrix (positive semi-definite)
        A = np.random.randn(n, n)
        cov = A @ A.T * 0.01  # Scale to reasonable values
        
        result = minimum_variance_portfolio(cov)
        
        assert np.isclose(np.sum(result), 1.0, atol=1e-5), f"Weights sum to {np.sum(result)}"
        assert np.all(result >= -1e-6), f"Negative weights: {result}"
    
    @given(
        prices=st.lists(
            st.floats(min_value=1.0, max_value=1000.0),
            min_size=20,
            max_size=100
        ),
        period=st.integers(min_value=5, max_value=30)
    )
    @settings(max_examples=20, deadline=None, suppress_health_check=[HealthCheck.too_slow])
    def test_sma_in_range(self, prices, period):
        """
        SMA must be within price range.
        """
        from quantterm.analytics.technical.indicators import sma
        
        assume(len(prices) > period)
        
        prices_arr = np.array(prices)
        result = sma(prices_arr, period)
        
        valid_result = result[~np.isnan(result)]
        assert len(valid_result) > 0
        
        # SMA should be between min and max of the window
        assert np.all((valid_result >= prices_arr.min() - 0.01) | np.isnan(result[~np.isnan(result)])), \
            "SMA below minimum price"
        assert np.all((valid_result <= prices_arr.max() + 0.01) | np.isnan(result[~np.isnan(result)])), \
            "SMA above maximum price"


class TestOptionGreeks:
    """
    Greeks properties that must hold.
    """
    
    @given(
        s=st.floats(min_value=10.0, max_value=1000.0),
        k=st.floats(min_value=10.0, max_value=1000.0),
        t=st.floats(min_value=0.01, max_value=2.0),
        r=st.floats(min_value=0.0, max_value=0.2),
        sigma=st.floats(min_value=0.01, max_value=1.0)
    )
    @settings(max_examples=500)
    def test_delta_call_bounds(self, s, k, t, r, sigma):
        """
        Call delta must be between 0 and 1.
        """
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s, k, t, r, sigma)
        
        assert 0 <= result.delta_call <= 1, \
            f"Call delta {result.delta_call} out of [0,1]"
    
    @given(
        s=st.floats(min_value=10.0, max_value=1000.0),
        k=st.floats(min_value=10.0, max_value=1000.0),
        t=st.floats(min_value=0.01, max_value=2.0),
        r=st.floats(min_value=0.0, max_value=0.2),
        sigma=st.floats(min_value=0.01, max_value=1.0)
    )
    @settings(max_examples=500)
    def test_delta_put_bounds(self, s, k, t, r, sigma):
        """
        Put delta must be between -1 and 0.
        """
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s, k, t, r, sigma)
        
        assert -1 <= result.delta_put <= 0, \
            f"Put delta {result.delta_put} out of [-1,0]"
    
    @given(
        s=st.floats(min_value=10.0, max_value=1000.0),
        k=st.floats(min_value=10.0, max_value=1000.0),
        t=st.floats(min_value=0.01, max_value=2.0),
        r=st.floats(min_value=0.0, max_value=0.2),
        sigma=st.floats(min_value=0.01, max_value=1.0)
    )
    @settings(max_examples=500, deadline=None)
    def test_gamma_positive(self, s, k, t, r, sigma):
        """
        Gamma must be non-negative (convexity).
        Note: Can be exactly 0 for deep OTM options due to numerical precision.
        """
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s, k, t, r, sigma)
        
        assert result.gamma >= 0, f"Gamma {result.gamma} should be non-negative"
    
    @given(
        s=st.floats(min_value=10.0, max_value=1000.0),
        k=st.floats(min_value=10.0, max_value=1000.0),
        t=st.floats(min_value=0.01, max_value=2.0),
        r=st.floats(min_value=0.0, max_value=0.2),
        sigma=st.floats(min_value=0.01, max_value=1.0)
    )
    @settings(max_examples=500, deadline=None)
    def test_vega_positive(self, s, k, t, r, sigma):
        """
        Vega must be non-negative (higher vol = higher option price).
        Note: Can be exactly 0 for deep OTM options due to numerical precision.
        """
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        
        result = black_scholes(s, k, t, r, sigma)
        
        assert result.vega >= 0, f"Vega {result.vega} should be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

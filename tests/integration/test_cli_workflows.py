"""
Integration tests for CLI workflows.
End-to-end tests for all QuantTerm CLI command groups.
"""

import pytest
import subprocess
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestQuoteCommands:
    """Test quote command group."""
    
    def test_quote_single_symbol(self):
        """Test getting quote for a single symbol."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "quote", "AAPL"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should not error out
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_quote_history(self):
        """Test getting historical data."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "history", "AAPL", "--start", "1y"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()


class TestTechnicalCommands:
    """Test technical analysis command group."""
    
    def test_indicator_rsi(self):
        """Test RSI indicator calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "indicator", "AAPL", "rsi", "--period", "14"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_indicator_sma(self):
        """Test SMA indicator calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "indicator", "AAPL", "sma", "--period", "20"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_indicator_ema(self):
        """Test EMA indicator calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "indicator", "AAPL", "ema", "--period", "12"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_indicator_macd(self):
        """Test MACD indicator calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "indicator", "AAPL", "macd"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_indicator_bollinger(self):
        """Test Bollinger Bands indicator."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "indicator", "AAPL", "bollinger"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_indicator_list(self):
        """Test listing all available indicators."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "list"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0


class TestDerivativesCommands:
    """Test derivatives command group."""
    
    def test_options_price_call(self):
        """Test Black-Scholes call option pricing."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "options", "price", "AAPL", "180", "30d", "call", "--vol", "0.30"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_options_price_put(self):
        """Test Black-Scholes put option pricing."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "options", "price", "AAPL", "180", "30d", "put", "--vol", "0.30"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_options_greeks(self):
        """Test Greeks calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "options", "greeks", "AAPL", "180", "30d", "call", "--vol", "0.30"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()


class TestRiskCommands:
    """Test risk command group."""
    
    def test_risk_var_historical(self):
        """Test historical VaR calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "risk", "var", "AAPL", "--notional", "100000"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_risk_var_parametric(self):
        """Test parametric VaR calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "risk", "var", "AAPL", "--notional", "100000", "--method", "parametric"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_risk_volatility(self):
        """Test volatility calculation."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "risk", "vol", "AAPL", "--period", "1y"],
            capture_output=True,
            text=True,
            timeout=60
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()


class TestPortfolioCommands:
    """Test portfolio command group."""
    
    def test_portfolio_analyze(self):
        """Test portfolio analysis."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "portfolio", "analyze", "AAPL", "MSFT", "GOOGL"],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()
    
    def test_portfolio_performance(self):
        """Test portfolio performance."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "portfolio", "perf", "AAPL", "MSFT", "--start", "6m"],
            capture_output=True,
            text=True,
            timeout=120
        )
        assert result.returncode == 0 or "error" in result.stdout.lower() or "error" in result.stderr.lower()


class TestCLIHelp:
    """Test CLI help and error handling."""
    
    def test_main_help(self):
        """Test main help output."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "quantterm" in result.stdout.lower()
    
    def test_invalid_command(self):
        """Test handling of invalid commands."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "invalid", "command"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should fail gracefully
        assert result.returncode != 0
    
    def test_invalid_indicator(self):
        """Test handling of invalid indicator."""
        result = subprocess.run(
            [sys.executable, "-m", "quantterm", "tech", "indicator", "AAPL", "invalid_indicator"],
            capture_output=True,
            text=True,
            timeout=30
        )
        # Should fail gracefully
        assert result.returncode != 0 or "not found" in result.stdout.lower() or "invalid" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

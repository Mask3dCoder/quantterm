"""Tests for paper trading engine."""
from quantterm.live.paper_trading import PaperTradingEngine


def test_market_order():
    """Test market order execution."""
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Buy 100 shares at $100
    trade = engine.execute_market_order('SPY', 'buy', 100, 100.0)
    
    assert trade.price > 100.0  # Slippage applied
    assert engine.cash < 100000  # Cash deducted
    assert engine.get_position('SPY') == 100
    
    print(f"Buy executed at ${trade.price:.2f}")
    print(f"Cash remaining: ${engine.cash:.2f}")


def test_pnl_calculation():
    """Test P&L tracking."""
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Buy at $100
    engine.execute_market_order('SPY', 'buy', 100, 100.0)
    
    # Price rises to $110
    engine.update_market_prices({'SPY': 110.0})
    
    summary = engine.get_summary({'SPY': 110.0})
    
    assert summary['total_pnl'] > 0  # Should be profit
    print(f"P&L: ${summary['total_pnl']:.2f}")


def test_flatten():
    """Test flattening positions."""
    engine = PaperTradingEngine(initial_capital=100000)
    
    # Buy and sell
    engine.execute_market_order('SPY', 'buy', 100, 100.0)
    engine.flatten({'SPY': 110.0})
    
    assert engine.get_position('SPY') == 0
    print(f"Position after flatten: {engine.get_position('SPY')}")


if __name__ == '__main__':
    test_market_order()
    test_pnl_calculation()
    test_flatten()
    print("All paper trading tests passed!")

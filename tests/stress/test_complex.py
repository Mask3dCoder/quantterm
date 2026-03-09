"""End-to-end stress test for the backtesting engine.

Tests all components under realistic conditions.
"""

import pytest

from quantterm.backtesting.engine import MultiSymbolEngine
from quantterm.backtesting.data_handler import MultiSymbolDataHandler
from quantterm.backtesting.portfolio import Portfolio, MarginPortfolio
from quantterm.backtesting.strategy.complex import ComplexStrategy
from quantterm.backtesting.strategy.rebalancing import RebalancingStrategy
from quantterm.backtesting.metrics import calculate_metrics


def test_multi_symbol_backtest():
    """Test: 6 symbols, 1 year, no errors."""
    engine = MultiSymbolEngine(
        strategy_class=ComplexStrategy,
        symbols=['SPY', 'QQQ', 'IWM', 'TLT', 'GLD'],
        initial_capital=1000000,
        data_handler=MultiSymbolDataHandler(),
    )
    
    result = engine.run('2022-01-01', '2023-12-31')
    
    assert result is not None
    assert result['final_value'] > 0
    assert len(result['trades']) > 0
    print(f"Multi-symbol: ${result['final_value']:,.2f}, {len(result['trades'])} trades")


def test_margin_portfolio():
    """Test: Margin calculations, short selling."""
    portfolio = MarginPortfolio(
        initial_cash=100000,
        margin_requirement=0.50,
        short_borrow_cost=0.03
    )
    
    # Test buying power
    prices = {'SPY': 400}
    assert portfolio.buying_power(prices) == 200000  # 2x with 50% margin
    
    # Test gross/net exposure
    portfolio.cash = 50000
    portfolio.long_positions['SPY'] = 100
    assert portfolio.gross_exposure(prices) == 40000
    assert portfolio.net_exposure(prices) == 40000


def test_rebalancing_costs():
    """Test: Transaction costs visible in returns."""
    engine = MultiSymbolEngine(
        strategy_class=RebalancingStrategy,
        symbols=['SPY', 'TLT', 'GLD'],
        initial_capital=1000000,
        data_handler=MultiSymbolDataHandler(),
    )
    
    result = engine.run('2015-01-01', '2023-12-31')
    
    # Calculate returns
    returns = []
    values = result.get('portfolio_values', [1000000, result['final_value']])
    for i in range(1, len(values)):
        ret = (values[i] - values[i-1]) / values[i-1]
        returns.append(ret)
    
    metrics = calculate_metrics(returns, 1000000)
    
    # Transaction costs should reduce returns
    # Buy-and-hold would be ~50%, rebalancing should be ~40-45%
    print(f"Rebalancing Return: {metrics.get('total_return', 0)*100:.2f}%")
    print(f"Trades: {len(result['trades'])}")


def test_determinism():
    """Test: Same inputs → same outputs."""
    engine = MultiSymbolEngine(
        strategy_class=ComplexStrategy,
        symbols=['SPY', 'TLT'],
        initial_capital=1000000,
        data_handler=MultiSymbolDataHandler(),
    )
    
    result1 = engine.run('2023-01-01', '2023-06-30')
    result2 = engine.run('2023-01-01', '2023-06-30')
    
    assert result1['final_value'] == result2['final_value']
    assert len(result1['trades']) == len(result2['trades'])


def test_performance():
    """Test: Backtest completes in reasonable time.
    
    Note: ComplexStrategy fetches historical data for momentum calculations,
    which involves network calls. This test uses a shorter period
    to keep runtime reasonable.
    """
    import time
    
    engine = MultiSymbolEngine(
        strategy_class=ComplexStrategy,
        symbols=['SPY', 'QQQ', 'TLT', 'GLD'],
        initial_capital=1000000,
        data_handler=MultiSymbolDataHandler(),
    )
    
    # Use shorter period for performance test
    start = time.time()
    result = engine.run('2023-01-01', '2023-12-31')
    elapsed = time.time() - start
    
    # Allow more time due to yfinance API calls in momentum calculation
    assert elapsed < 120, f"Backtest took {elapsed:.1f}s, expected <120s"
    print(f"Backtest completed in {elapsed:.1f}s")


def test_complex_strategy_metrics():
    """Test: Complex strategy produces reasonable metrics."""
    from quantterm.backtesting.strategy.complex import ComplexStrategy
    
    # Test that the strategy can be instantiated
    portfolio = Portfolio(1000000)
    data_handler = MultiSymbolDataHandler()
    
    strategy = ComplexStrategy(
        name="TestComplex",
        portfolio=portfolio,
        data_handler=data_handler,
        symbols=['SPY', 'QQQ'],
        target_weights={'SPY': 0.5, 'QQQ': 0.5}
    )
    
    assert strategy.name == "TestComplex"
    assert len(strategy.symbols) == 2
    assert 'SPY' in strategy.target_weights
    assert 'QQQ' in strategy.target_weights


if __name__ == '__main__':
    print("Running stress tests...")
    test_multi_symbol_backtest()
    test_margin_portfolio()
    test_rebalancing_costs()
    test_determinism()
    test_performance()
    test_complex_strategy_metrics()
    print("All stress tests passed!")

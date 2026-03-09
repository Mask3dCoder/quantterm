"""Performance metrics for backtesting.

Provides functions to calculate key performance metrics including
Sharpe ratio, max drawdown, and win rate from backtest returns.
"""

import numpy as np
from typing import List, Dict, Optional


def calculate_returns(portfolio_values: List[float]) -> List[float]:
    """Calculate period-over-period returns from portfolio values.
    
    Args:
        portfolio_values: List of portfolio values over time.
        
    Returns:
        List of periodic returns.
    """
    if len(portfolio_values) < 2:
        return []
    
    returns = []
    for i in range(1, len(portfolio_values)):
        if portfolio_values[i-1] > 0:
            ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
            returns.append(ret)
    
    return returns


def calculate_metrics(returns: List[float], initial_value: float) -> Dict:
    """Calculate performance metrics from returns.
    
    Args:
        returns: List of periodic returns (e.g., daily returns).
        initial_value: Initial portfolio value.
        
    Returns:
        Dictionary with performance metrics:
        - total_return: Total return over the period
        - annualized_return: Annualized return (assuming 252 trading days)
        - volatility: Annualized volatility
        - sharpe_ratio: Sharpe ratio (assuming 0% risk-free rate)
        - max_drawdown: Maximum drawdown
        - calmar_ratio: Calmar ratio (annualized return / max drawdown)
        - win_rate: Percentage of positive returns
        - n_periods: Number of return periods
    """
    if not returns or initial_value <= 0:
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'n_periods': 0,
        }
    
    returns_array = np.array(returns)
    n_periods = len(returns)
    
    # Total return: compound the returns
    total_return = np.prod(1 + returns_array) - 1
    
    # Annualized return (assuming 252 trading days per year)
    years = n_periods / 252
    if years > 0:
        annualized_return = (1 + total_return) ** (1 / years) - 1
    else:
        annualized_return = total_return
    
    # Volatility (annualized)
    volatility = np.std(returns_array, ddof=1) * np.sqrt(252) if n_periods > 1 else 0.0
    
    # Sharpe ratio (assuming 0% risk-free rate)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0.0
    
    # Max drawdown from cumulative returns
    cumulative = np.cumprod(1 + returns_array)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = (cumulative - running_max) / running_max
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0
    
    # Calmar ratio (annualized return / max drawdown)
    calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0.0
    
    # Win rate
    win_rate = np.sum(returns_array > 0) / n_periods if n_periods > 0 else 0.0
    
    return {
        'total_return': float(total_return),
        'annualized_return': float(annualized_return),
        'volatility': float(volatility),
        'sharpe_ratio': float(sharpe_ratio),
        'max_drawdown': float(max_drawdown),
        'calmar_ratio': float(calmar_ratio),
        'win_rate': float(win_rate),
        'n_periods': n_periods,
    }


def format_metrics(metrics: Dict, initial_capital: float, final_value: float) -> str:
    """Format metrics into a readable string.
    
    Args:
        metrics: Dictionary of calculated metrics.
        initial_capital: Initial capital amount.
        final_value: Final portfolio value.
        
    Returns:
        Formatted string with metrics.
    """
    lines = [
        "=" * 50,
        "PERFORMANCE SUMMARY",
        "=" * 50,
        f"Initial Capital:  ${initial_capital:,.2f}",
        f"Final Value:      ${final_value:,.2f}",
        f"Total Return:     {metrics.get('total_return', 0) * 100:+.2f}%",
        f"Annualized Ret:   {metrics.get('annualized_return', 0) * 100:+.2f}%",
        "-" * 50,
        f"Sharpe Ratio:     {metrics.get('sharpe_ratio', 0):.2f}",
        f"Max Drawdown:    {metrics.get('max_drawdown', 0) * 100:.2f}%",
        f"Calmar Ratio:    {metrics.get('calmar_ratio', 0):.2f}",
        f"Win Rate:        {metrics.get('win_rate', 0) * 100:.2f}%",
        f"Volatility:      {metrics.get('volatility', 0) * 100:.2f}%",
        "-" * 50,
        f"Trading Periods:  {metrics.get('n_periods', 0)}",
        "=" * 50,
    ]
    return "\n".join(lines)

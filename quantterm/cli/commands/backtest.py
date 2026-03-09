"""Backtesting CLI commands."""
import typer

app = typer.Typer(name="backtest", help="Backtesting commands")


@app.command()
def run(
    symbol: str = typer.Option("SPY", "--symbol", "-s", help="Ticker symbol"),
    start: str = typer.Option("2020-01-01", "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2020-12-31", "--end", help="End date (YYYY-MM-DD)"),
    initial_capital: float = typer.Option(1000000.0, "--capital", "--initial-capital", help="Initial capital"),
    strategy: str = typer.Option("BuyAndHold", "--strategy", help="Strategy to use"),
):
    """Run backtest with event-driven engine.
    
    Executes a backtest using the specified parameters and outputs
    performance metrics including Sharpe ratio, max drawdown, and returns.
    """
    from rich.console import Console
    from rich.table import Table
    
    from quantterm.backtesting.engine import BacktestEngine
    from quantterm.backtesting.strategy.base import BuyAndHoldStrategy
    from quantterm.backtesting.data_handler import DataHandler
    from quantterm.backtesting.metrics import calculate_returns, calculate_metrics, format_metrics
    
    console = Console()
    
    console.print(f"[bold blue]Running backtest...[/bold blue]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Period: {start} to {end}")
    console.print(f"Capital: ${initial_capital:,.2f}")
    console.print(f"Strategy: {strategy}")
    console.print("-" * 50)
    
    # Initialize engine with buy-and-hold strategy
    engine = BacktestEngine(
        strategy_class=BuyAndHoldStrategy,
        initial_capital=initial_capital,
        data_handler=DataHandler()
    )
    
    # Run backtest
    result = engine.run(symbol, start, end)
    
    # Get portfolio values and calculate returns
    portfolio_values = result.get('portfolio_values', [])
    
    if not portfolio_values:
        console.print("[yellow]Warning: No portfolio values recorded[/yellow]")
        return
    
    # Prepend initial capital as starting value
    equity_curve = [initial_capital] + portfolio_values
    
    # Calculate returns and metrics
    returns = calculate_returns(equity_curve)
    metrics = calculate_metrics(returns, initial_capital)
    
    # Display results
    console.print("\n" + format_metrics(metrics, initial_capital, result['final_value']))
    
    # Additional details
    console.print(f"\n[bold]Trade Details:[/bold]")
    console.print(f"Total Trades: {len(result['trades'])}")
    console.print(f"Realized P&L: ${result['realized_pnl']:,.2f}")
    console.print(f"Unrealized P&L: ${result['unrealized_pnl']:,.2f}")


@app.command()
def walkforward(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy name"),
    train_window: int = typer.Option(252, "--train", help="Training window (days)"),
    test_window: int = typer.Option(63, "--test", help="Testing window (days)"),
):
    """Run walk-forward optimization."""
    from rich.console import Console

    console = Console()
    console.print(f"[bold blue]Running walk-forward analysis...[/bold blue]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Train window: {train_window}d, Test window: {test_window}d")

"""Portfolio CLI commands."""
import typer
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

app = typer.Typer(help="Portfolio management commands")
console = Console()


def parse_relative_date(date_str: str) -> str:
    """Convert relative dates to YYYY-MM-DD."""
    from datetime import datetime, timedelta
    
    if date_str in ('today', 'now'):
        return datetime.now().strftime('%Y-%m-%d')
    
    if date_str is None:
        return datetime.now().strftime('%Y-%m-%d')
    
    # Normalize shorthand: 6m -> 6mo, 3m -> 3mo, 1d -> 1d, etc.
    normalized = date_str.lower().strip()
    if len(normalized) >= 2 and normalized[-1] in ('d', 'w', 'm', 'y'):
        num = normalized[:-1]
        unit = normalized[-1]
        if unit == 'm':
            unit = 'mo'  # normalize m to mo
        normalized = num + unit
    
    mapping = {
        '1d': timedelta(days=1),
        '5d': timedelta(days=5),
        '1w': timedelta(weeks=1),
        '1mo': timedelta(days=30),
        '3mo': timedelta(days=90),
        '6mo': timedelta(days=180),
        '1y': timedelta(days=365),
        '2y': timedelta(days=730),
    }
    
    # Try the normalized key first, then the original
    if normalized in mapping:
        delta = mapping[normalized]
        return (datetime.now() - delta).strftime('%Y-%m-%d')
    
    # Try parsing as YYYY-MM-DD
    try:
        datetime.strptime(date_str, '%Y-%m-%d')
        return date_str
    except ValueError:
        pass
    
    # Return as-is if already in some other format
    return date_str


@app.command("optimize")
def optimize_portfolio(
    symbols: list[str] = typer.Argument(..., help="Portfolio symbols (e.g., AAPL MSFT GOOG)"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Initial weights (optional)"),
    target: str = typer.Option("sharpe", "--target", "-t", help="Target: sharpe, min_var, max_return"),
    lookback: str = typer.Option("1y", "--lookback", "-l", help="Historical lookback"),
    risk_free: float = typer.Option(0.05, "--rf", "-r", help="Risk-free rate"),
):
    """
    Optimize portfolio allocation.
    
    Example:
        quantterm portfolio optimize AAPL MSFT GOOG --target sharpe
    """
    try:
        from quantterm.portfolio.optimization.mean_variance import mean_variance_optimize
        from quantterm.data.providers import yahoo as yahoo_provider
        
        with console.status("Fetching historical data..."):
            start_date = parse_relative_date(lookback)
            returns_list = []
            
            for symbol in symbols:
                df = yahoo_provider.get_history(symbol, start=start_date)
                if df is None or df.empty:
                    console.print(f"[red]No data for {symbol}[/red]")
                    raise typer.Exit(1)
                rets = df['Close'].pct_change().dropna()
                returns_list.append(rets)
        
        # Align returns
        returns_df = pd.concat(returns_list, axis=1).dropna()
        returns_df.columns = symbols
        
        # Calculate expected returns and covariance
        expected_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        # Optimize
        with console.status("Optimizing portfolio..."):
            if target == "sharpe":
                # Maximize Sharpe ratio (minimize negative)
                result = mean_variance_optimize(
                    expected_returns.values,
                    cov_matrix.values,
                    risk_aversion=1.0 / (expected_returns.mean() / cov_matrix.values.diagonal().mean() ** 0.5 - risk_free),
                    long_only=True
                )
            elif target == "min_var":
                result = mean_variance_optimize(
                    expected_returns.values,
                    cov_matrix.values,
                    risk_aversion=10.0,
                    long_only=True
                )
            else:  # max_return
                result = mean_variance_optimize(
                    expected_returns.values,
                    cov_matrix.values,
                    risk_aversion=0.1,
                    long_only=True
                )
        
        optimal_weights = result[0]
        port_return = result[1]
        port_vol = result[2]
        
        # Display results
        console.print(Panel(
            f"[bold cyan]Portfolio Optimization Results[/bold cyan]\n"
            f"Target: {target.capitalize()}\n"
            f"Assets: {', '.join(symbols)}",
            border_style="purple"
        ))
        
        # Weights table
        weights_table = Table(title="Optimal Allocation", show_header=True, box=None)
        weights_table.add_column("Symbol", style="cyan")
        weights_table.add_column("Weight", justify="right")
        weights_table.add_column("Allocation", justify="right")
        
        for symbol, weight in zip(symbols, optimal_weights):
            weight_pct = weight * 100
            bar = "█" * int(weight_pct / 5) + "░" * (20 - int(weight_pct / 5))
            weights_table.add_row(
                symbol,
                f"{weight_pct:.1f}%",
                bar
            )
        
        console.print(weights_table)
        
        # Performance metrics
        sharpe = (port_return - risk_free) / port_vol if port_vol > 0 else 0
        
        perf_table = Table(show_header=False, box=None)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", justify="right")
        
        perf_table.add_row("Expected Return", f"{port_return:.2%}")
        perf_table.add_row("Volatility", f"{port_vol:.2%}")
        perf_table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
        
        console.print(perf_table)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("analyze")
def analyze_portfolio(
    symbols: list[str] = typer.Argument(..., help="Portfolio symbols"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Weights (e.g., 0.4 0.4 0.2)"),
    lookback: str = typer.Option("1y", "--lookback", "-l", help="Historical lookback"),
):
    """
    Analyze portfolio performance.
    
    Example:
        quantterm portfolio analyze AAPL MSFT GOOG --weights 0.4 0.4 0.2
    """
    try:
        from quantterm.data.providers import yahoo as yahoo_provider
        
        # Parse weights
        if weights:
            try:
                weight_list = [float(w) for w in weights.split()]
                if len(weight_list) != len(symbols):
                    console.print("[red]Number of weights must match number of symbols[/red]")
                    raise typer.Exit(1)
            except ValueError:
                console.print("[red]Invalid weights format[/red]")
                raise typer.Exit(1)
        else:
            weight_list = [1.0 / len(symbols)] * len(symbols)
        
        weights_arr = np.array(weight_list)
        weights_arr = weights_arr / weights_arr.sum()
        
        with console.status("Fetching historical data..."):
            start_date = parse_relative_date(lookback)
            returns_list = []
            
            for symbol in symbols:
                df = yahoo_provider.get_history(symbol, start=start_date)
                if df is None or df.empty:
                    console.print(f"[red]No data for {symbol}[/red]")
                    raise typer.Exit(1)
                rets = df['Close'].pct_change().dropna()
                returns_list.append(rets)
        
        # Align returns
        returns_df = pd.concat(returns_list, axis=1).dropna()
        returns_df.columns = symbols
        
        # Calculate portfolio returns
        port_returns = (returns_df * weights_arr).sum(axis=1)
        
        # Display results
        console.print(Panel(
            f"[bold cyan]Portfolio Analysis[/bold cyan]\n"
            f"Assets: {', '.join(symbols)}",
            border_style="purple"
        ))
        
        # Performance stats
        mean_ret = port_returns.mean() * 252
        vol = port_returns.std() * np.sqrt(252)
        sharpe = mean_ret / vol if vol > 0 else 0
        
        # Cumulative returns
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        perf_table = Table(show_header=False, box=None)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", justify="right")
        
        perf_table.add_row("Annual Return", f"{mean_ret:.2%}")
        perf_table.add_row("Annual Volatility", f"{vol:.2%}")
        perf_table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
        perf_table.add_row("Max Drawdown", f"{max_dd:.2%}")
        
        console.print(perf_table)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("create")
def create_portfolio(
    name: str = typer.Option(..., "--name", "-n", help="Portfolio name"),
    cash: float = typer.Option(1000000, "--cash", "-c", help="Initial cash"),
):
    """Create a new portfolio."""
    console.print(f"[bold green]Created portfolio: {name}[/bold green]")
    console.print(f"Initial cash: ${cash:,.2f}")


@app.command("add")
def add_position(
    ticker: str = typer.Argument(..., help="Ticker symbol"),
    quantity: float = typer.Option(..., "--quantity", "-q", help="Quantity"),
    avg_cost: float = typer.Option(..., "--avg-cost", help="Average cost"),
):
    """Add position to portfolio."""
    console.print(f"[bold green]Added position: {ticker}[/bold green]")
    console.print(f"Quantity: {quantity}, Avg Cost: ${avg_cost:.2f}")


@app.command("perf")
def portfolio_performance(
    symbols: list[str] = typer.Argument(..., help="Portfolio symbols (e.g., AAPL MSFT GOOG)"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Portfolio weights (e.g., 0.6 0.4)"),
    start: str = typer.Option("1y", "--start", "-s", help="Start date (1y, 6m, YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="End date"),
):
    """
    Calculate portfolio performance metrics.
    
    Example:
        quantterm portfolio perf AAPL MSFT --start 6m
        quantterm portfolio perf AAPL MSFT GOOG --weights 0.5 0.3 0.2
    """
    try:
        from quantterm.data.providers import yahoo as yahoo_provider
        import numpy as np
        
        start_date = parse_relative_date(start)
        end_date = parse_relative_date(end) if end else None
        
        # Fetch historical data
        with console.status(f"Fetching historical data for {len(symbols)} symbols..."):
            prices_dict = {}
            for symbol in symbols:
                df = yahoo_provider.get_history(symbol, start=start_date, end=end_date)
                if df is None or df.empty:
                    console.print(f"[red]Error: No data available for {symbol}[/red]")
                    raise typer.Exit(1)
                prices_dict[symbol] = df['Close'].values
            
            # Align prices by finding common dates
            min_len = min(len(p) for p in prices_dict.values())
            prices = np.column_stack([p[-min_len:] for p in prices_dict.values()])
        
        # Calculate returns
        returns = np.diff(prices, axis=0) / prices[:-1]
        
        # Parse weights
        if weights:
            weight_list = [float(w) for w in weights.split()]
            if len(weight_list) != len(symbols):
                console.print(f"[red]Error: Number of weights ({len(weight_list)}) must match symbols ({len(symbols)})[/red]")
                raise typer.Exit(1)
            w = np.array(weight_list)
        else:
            w = np.ones(len(symbols)) / len(symbols)
        
        # Calculate portfolio metrics
        port_returns = returns @ w
        
        # Annualized return and volatility
        annual_return = np.mean(port_returns) * 252
        annual_vol = np.std(port_returns) * np.sqrt(252)
        
        # Sharpe ratio (assuming 5% risk-free rate)
        rf = 0.05
        sharpe = (annual_return - rf) / annual_vol if annual_vol > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + port_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown)
        
        # Display results
        table = Table(title="Portfolio Performance Metrics")
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        
        table.add_row("Annualized Return", f"{annual_return*100:.2f}%")
        table.add_row("Annualized Volatility", f"{annual_vol*100:.2f}%")
        table.add_row("Sharpe Ratio", f"{sharpe:.2f}")
        table.add_row("Max Drawdown", f"{max_drawdown*100:.2f}%")
        
        console.print(table)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

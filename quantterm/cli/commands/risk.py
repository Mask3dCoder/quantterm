"""Risk management CLI commands."""
import typer
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from typing import Optional

app = typer.Typer(help="Risk management and analytics")
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
    
    if normalized in mapping:
        delta = mapping[normalized]
        return (datetime.now() - delta).strftime('%Y-%m-%d')
    
    return date_str


@app.command("var")
def calculate_var(
    symbols: list[str] = typer.Argument(..., help="Portfolio symbols (e.g., AAPL MSFT GOOG)"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Weights (e.g., 0.4 0.4 0.2)"),
    confidence: float = typer.Option(0.95, "--confidence", "-c", help="Confidence level"),
    horizon: int = typer.Option(1, "--horizon", help="Time horizon in days"),
    method: str = typer.Option("historical", "--method", "-m", help="Method: historical, parametric"),
    lookback: str = typer.Option("1y", "--lookback", "-l", help="Historical lookback"),
    notional: float = typer.Option(1000000, "--notional", "-n", help="Portfolio value"),
):
    """
    Calculate Value at Risk (VaR) for a portfolio.
    
    Example:
        quantterm risk var AAPL MSFT GOOG --weights 0.4 0.4 0.2 --confidence 0.95
        quantterm risk var SPY --method parametric --horizon 5
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
        
        # Fetch historical data
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
        
        # Calculate VaR based on method
        if method == "historical":
            var = np.percentile(port_returns, (1 - confidence) * 100)
        else:  # parametric
            mu = port_returns.mean()
            sigma = port_returns.std()
            from scipy.stats import norm
            z = norm.ppf(1 - confidence)
            var = mu + sigma * z
        
        # Calculate Expected Shortfall (CVaR)
        cvar = port_returns[port_returns <= var].mean()
        
        # Display results
        console.print(Panel(
            f"[bold cyan]Value at Risk Analysis[/bold cyan]\n"
            f"Portfolio: {', '.join(symbols)}\n"
            f"Notional: ${notional:,.0f}",
            border_style="purple"
        ))
        
        # Summary
        summary = Table(show_header=False, box=None)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")
        
        summary.add_row("Method", method.capitalize())
        summary.add_row("Confidence", f"{confidence:.1%}")
        summary.add_row("Horizon", f"{horizon} day(s)")
        
        var_dollar = var * notional * np.sqrt(horizon)
        cvar_dollar = cvar * notional * np.sqrt(horizon)
        
        summary.add_row("Daily VaR", f"{var:.2%}")
        summary.add_row("Daily VaR ($)", f"${var * notional:,.2f}")
        summary.add_row(f"{horizon}-Day VaR", f"${var_dollar:,.2f}")
        summary.add_row(f"{horizon}-Day CVaR", f"${cvar_dollar:,.2f}")
        
        console.print(summary)
        
        # Risk stats
        stats_table = Table(title="Risk Statistics", show_header=True, box=None)
        stats_table.add_column("Statistic", style="cyan")
        stats_table.add_column("Value", justify="right")
        
        stats_table.add_row("Mean Return", f"{port_returns.mean()*252:.2%}")
        stats_table.add_row("Volatility (Ann.)", f"{port_returns.std()*np.sqrt(252):.2%}")
        stats_table.add_row("Skewness", f"{port_returns.skew():.3f}")
        stats_table.add_row("Kurtosis", f"{port_returns.kurtosis():.3f}")
        
        console.print(stats_table)
        
        # Max Drawdown
        cumulative = (1 + port_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        console.print(f"\n[cyan]Max Drawdown:[/cyan] [red]{max_dd:.2%}[/red]")
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("vol")
def analyze_risk(
    symbol: str = typer.Argument(..., help="Symbol to analyze"),
    period: str = typer.Option("1y", "--period", "-p", help="Lookback period"),
):
    """
    Analyze risk metrics for a single security.
    
    Example:
        quantterm risk vol AAPL --period 1y
    """
    try:
        from quantterm.data.providers import yahoo as yahoo_provider
        
        with console.status(f"Fetching data for {symbol}..."):
            start_date = parse_relative_date(period)
            df = yahoo_provider.get_history(symbol, start=start_date)
            
            if df is None or df.empty:
                console.print(f"[red]No data for {symbol}[/red]")
                raise typer.Exit(1)
            
            returns = df['Close'].pct_change().dropna()
        
        console.print(Panel(
            f"[bold cyan]{symbol.upper()} Risk Analysis[/bold cyan]",
            border_style="purple"
        ))
        
        # Calculate risk metrics
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)
        sharpe = mean_return / volatility if volatility > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # VaR
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        stats = Table(show_header=False, box=None)
        stats.add_column("Metric", style="cyan")
        stats.add_column("Value", justify="right")
        
        stats.add_row("Annual Return", f"{mean_return:.2%}")
        stats.add_row("Annual Volatility", f"{volatility:.2%}")
        stats.add_row("Sharpe Ratio", f"{sharpe:.2f}")
        stats.add_row("VaR (95%)", f"{var_95:.2%}")
        stats.add_row("VaR (99%)", f"{var_99:.2%}")
        stats.add_row("Max Drawdown", f"{max_dd:.2%}")
        
        console.print(stats)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("stress")
def stress_test(
    symbols: list[str] = typer.Argument(..., help="Portfolio symbols (e.g., AAPL MSFT GOOG)"),
    weights: Optional[str] = typer.Option(None, "--weights", "-w", help="Weights (e.g., 0.4 0.4 0.2)"),
    scenario: str = typer.Option("2008", "--scenario", "-s", help="Stress scenario: 2008, 2020, covid, rate_hike, custom"),
    lookback: str = typer.Option("1y", "--lookback", "-l", help="Historical lookback"),
    notional: float = typer.Option(1000000, "--notional", "-n", help="Portfolio value"),
):
    """
    Run stress tests on portfolio.
    
    Example:
        quantterm risk stress AAPL MSFT GOOG --scenario 2008
        quantterm risk stress AAPL MSFT --weights 0.6 0.4 --scenario covid
    """
    try:
        from quantterm.data.providers import yahoo as yahoo_provider
        
        # Define stress scenarios
        scenarios = {
            '2008': {
                'name': '2008 Financial Crisis',
                'equity_shock': -50,
                'bond_shift': 5,
                'description': 'Simulates 2008 financial crisis market conditions'
            },
            '2020': {
                'name': 'March 2020 Crash',
                'equity_shock': -34,
                'bond_shift': 8,
                'description': 'Simulates March 2020 COVID crash'
            },
            'covid': {
                'name': 'COVID-19 Pandemic',
                'equity_shock': -30,
                'bond_shift': 10,
                'description': 'Simulates COVID-19 market impact'
            },
            'rate_hike': {
                'name': 'Rate Hike Shock',
                'equity_shock': -20,
                'bond_shift': -8,
                'description': 'Simulates aggressive Fed rate hiking cycle'
            },
            'black_monday': {
                'name': 'Black Monday 1987',
                'equity_shock': -22,
                'bond_shift': 0,
                'description': 'Simulates 1987 Black Monday crash'
            },
            'dotcom': {
                'name': 'Dot-com Bubble Burst',
                'equity_shock': -45,
                'bond_shift': 3,
                'description': 'Simulates 2000-2002 tech crash'
            },
        }
        
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
        
        # Get current prices
        with console.status("Fetching current prices..."):
            prices = []
            for symbol in symbols:
                quote = yahoo_provider.get_quote(symbol)
                price = quote.get('price', 0)
                if price == 0:
                    console.print(f"[red]Could not fetch price for {symbol}[/red]")
                    raise typer.Exit(1)
                prices.append(price)
        
        # Get historical data for correlation
        start_date = parse_relative_date(lookback)
        with console.status("Fetching historical data..."):
            returns_list = []
            for symbol in symbols:
                df = yahoo_provider.get_history(symbol, start=start_date)
                if df is None or df.empty:
                    console.print(f"[red]No data for {symbol}[/red]")
                    raise typer.Exit(1)
                rets = df['Close'].pct_change().dropna()
                returns_list.append(rets)
        
        returns_df = pd.concat(returns_list, axis=1).dropna()
        returns_df.columns = symbols
        
        # Calculate portfolio volatility
        port_returns = (returns_df * weights_arr).sum(axis=1)
        portfolio_vol = port_returns.std() * np.sqrt(252)
        
        # Get scenario
        if scenario.lower() not in scenarios:
            console.print(f"[yellow]Unknown scenario '{scenario}'. Available: {', '.join(scenarios.keys())}[/yellow]")
            console.print("[dim]Using default 2008 scenario[/dim]")
            scenario = '2008'
        
        sc = scenarios[scenario.lower()]
        
        # Calculate stress impact
        equity_shock = sc['equity_shock'] / 100
        bond_shift = sc['bond_shift'] / 100
        
        # Simplified: assume all equity-like assets affected equally
        # In reality, you'd want sector/asset class mapping
        portfolio_impact = equity_shock
        
        # Calculate dollar impact
        dollar_impact = notional * portfolio_impact
        new_value = notional + dollar_impact
        
        # Display results
        console.print(Panel(
            f"[bold cyan]Stress Test Results[/bold cyan]\n"
            f"Scenario: {sc['name']}\n"
            f"Portfolio: {', '.join(symbols)}",
            border_style="red"
        ))
        
        console.print(f"[dim]{sc['description']}[/dim]\n")
        
        # Summary table
        summary = Table(show_header=False, box=None)
        summary.add_column("Metric", style="cyan")
        summary.add_column("Value", justify="right")
        
        summary.add_row("Scenario", sc['name'])
        summary.add_row("Market Shock", f"{sc['equity_shock']}%")
        summary.add_row("Portfolio Vol (Ann.)", f"{portfolio_vol:.2%}")
        summary.add_row("Current Value", f"${notional:,.0f}")
        summary.add_row("[bold]Stressed Value[/bold]", f"[bold]${new_value:,.0f}[/bold]")
        summary.add_row("[bold]Loss[/bold]", f"[bold red]${dollar_impact:,.0f}[/bold red]")
        summary.add_row("[bold]Return[/bold]", f"[bold red]{portfolio_impact:.2%}[/bold red]")
        
        console.print(summary)
        
        # Additional context
        console.print(f"\n[dim]Note: This is a simplified stress test. Real analysis would include:\n"        f"• Sector/industry specific shocks\n"
        f"• Correlation breakdown in crisis\n"
        f"• Liquidity stress\n"
        f"• Counterparty risk[/dim]")
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

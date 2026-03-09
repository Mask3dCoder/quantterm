"""Technical analysis CLI commands."""
import typer
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np
import pandas as pd

from quantterm.analytics.technical.indicators import (
    sma, ema, wma, hma, macd, rsi, stochastic,
    bollinger_bands, atr, obv, vwap, cci, williams_r,
    roc, mfi, keltner_channels, donchian_channels, ichimoku
)
from quantterm.data.providers import yahoo as yahoo_provider
from quantterm.cli.utils import SuppressStderr

app = typer.Typer(help="Technical analysis commands")
console = Console()

INDICATOR_MAP = {
    # Aliases
    'bollinger': ('volatility', 'prices', None),
    'bb': ('volatility', 'prices', None),
    'macd': ('trend', 'prices', None),
    # Full names
    'sma': ('trend', 'prices', 'period'),
    'ema': ('trend', 'prices', 'period'),
    'wma': ('trend', 'prices', 'period'),
    'hma': ('trend', 'prices', 'period'),
    'rsi': ('momentum', 'prices', 'period'),
    'stochastic': ('momentum', 'ohlc', None),
    'bollinger_bands': ('volatility', 'prices', None),
    'atr': ('volatility', 'ohlc', 'period'),
    'obv': ('volume', 'prices_vol', None),
    'vwap': ('volume', 'ohlcv', None),
    'cci': ('momentum', 'ohlc', 'period'),
    'williams_r': ('momentum', 'ohlc', 'period'),
    'roc': ('momentum', 'prices', 'period'),
    'mfi': ('volume', 'ohlcv', 'period'),
    'keltner_channels': ('volatility', 'ohlc', None),
    'donchian_channels': ('volatility', 'ohlc', 'period'),
    'ichimoku': ('trend', 'ohlc', None),
}


def parse_relative_date(date_str: str) -> str:
    """Convert relative dates like 1y, 6m to YYYY-MM-DD."""
    from datetime import datetime, timedelta
    
    if date_str in ('today', 'now'):
        return datetime.now().strftime('%Y-%m-%d')
    
    mapping = {
        '1d': timedelta(days=1),
        '5d': timedelta(days=5),
        '1w': timedelta(weeks=1),
        '1mo': timedelta(days=30),
        '3mo': timedelta(days=90),
        '6mo': timedelta(days=180),
        '1y': timedelta(days=365),
        '2y': timedelta(days=730),
        '5y': timedelta(days=1825),
    }
    
    if date_str.lower() in mapping:
        delta = mapping[date_str.lower()]
        return (datetime.now() - delta).strftime('%Y-%m-%d')
    
    return date_str


@app.command("indicator")
def calculate_indicator(
    symbol: str = typer.Argument(..., help="Stock symbol (e.g., AAPL)"),
    indicator: str = typer.Argument(..., help="Indicator name (sma, ema, rsi, macd, etc.)"),
    period: int = typer.Option(14, "--period", "-p", help="Lookback period"),
    start: str = typer.Option("1y", "--start", "-s", help="Start date (1y, 6m, YYYY-MM-DD)"),
    end: Optional[str] = typer.Option(None, "--end", "-e", help="End date"),
):
    """
    Calculate technical indicators for a given symbol.
    
    Example:
        quantterm tech indicator AAPL rsi --period 14
        quantterm tech indicator TSLA macd
        quantterm tech indicator SPY bollinger_bands --period 20
    """
    indicator = indicator.lower()
    
    # Validate indicator
    if indicator not in INDICATOR_MAP:
        available = ", ".join(sorted(INDICATOR_MAP.keys()))
        console.print(f"[red]Error: Unknown indicator '{indicator}'[/red]")
        console.print(f"Available: {available}")
        raise typer.Exit(1)
    
    # Fetch data
    start_date = parse_relative_date(start)
    end_date = parse_relative_date(end) if end else None
    
    try:
        df = yahoo_provider.get_history(symbol, start=start_date, end=end_date)
        
        if df is None or df.empty:
            console.print(f"[red]No data returned for {symbol}[/red]")
            raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error fetching data: {e}[/red]")
        raise typer.Exit(1)
    
    # Prepare data arrays
    prices = df['Close'].values
    high = df['High'].values if 'High' in df.columns else prices
    low = df['Low'].values if 'Low' in df.columns else prices
    volume = df['Volume'].values if 'Volume' in df.columns else np.ones(len(prices))
    
    try:
        # Calculate indicator
        if indicator == 'sma':
            result = sma(prices, period)
        elif indicator == 'ema':
            result = ema(prices, period)
        elif indicator == 'wma':
            result = wma(prices, period)
        elif indicator == 'hma':
            result = hma(prices, period)
        elif indicator == 'macd':
            macd_line, signal, hist = macd(prices)
            result = (macd_line, signal, hist)
        elif indicator == 'rsi':
            result = rsi(prices, period)
        elif indicator == 'stochastic':
            k, d = stochastic(high, low, prices, period)
            result = (k, d)
        elif indicator in ('bollinger', 'bb', 'bollinger_bands'):
            upper, middle, lower = bollinger_bands(prices, period)
            result = (upper, middle, lower)
        elif indicator == 'atr':
            result = atr(high, low, prices, period)
        elif indicator == 'obv':
            result = obv(prices, volume)
        elif indicator == 'vwap':
            result = vwap(high, low, prices, volume)
        elif indicator == 'cci':
            result = cci(high, low, prices, period)
        elif indicator == 'williams_r':
            result = williams_r(high, low, prices, period)
        elif indicator == 'roc':
            result = roc(prices, period)
        elif indicator == 'mfi':
            result = mfi(high, low, prices, volume, period)
        elif indicator == 'keltner_channels':
            upper, middle, lower = keltner_channels(high, low, prices)
            result = (upper, middle, lower)
        elif indicator == 'donchian_channels':
            upper, lower = donchian_channels(high, low, period)
            result = (upper, lower)
        elif indicator == 'ichimoku':
            result = ichimoku(high, low, prices)
        else:
            result = sma(prices, period)  # Default fallback
    except Exception as e:
        console.print(f"[red]Error calculating {indicator}: {e}[/red]")
        raise typer.Exit(1)
    
    # Display results
    _display_indicator_results(df, result, indicator, symbol)


def _display_indicator_results(df: pd.DataFrame, result, indicator: str, symbol: str):
    """Format and display indicator results."""
    console.print(Panel(
        f"[bold cyan]{indicator.upper()} for {symbol.upper()}[/bold cyan]",
        border_style="purple"
    ))
    
    # Handle different return types
    if isinstance(result, tuple):
        # Multiple arrays returned
        if indicator == 'macd':
            labels = ['MACD', 'Signal', 'Histogram']
            latest = result[0][-1] if len(result[0]) > 0 else np.nan
            if not np.isnan(latest):
                color = "green" if latest > 0 else "red"
                console.print(f"Current MACD: [{color}]{latest:.4f}[/{color}]")
        elif indicator in ('bollinger', 'bb', 'bollinger_bands'):
            latest_upper = result[0][-1] if len(result[0]) > 0 else np.nan
            latest_middle = result[1][-1] if len(result[1]) > 0 else np.nan
            latest_lower = result[2][-1] if len(result[2]) > 0 else np.nan
            console.print(f"Upper: ${latest_upper:.2f} | Middle: ${latest_middle:.2f} | Lower: ${latest_lower:.2f}")
        elif indicator == 'stochastic':
            latest_k = result[0][-1] if len(result[0]) > 0 else np.nan
            latest_d = result[1][-1] if len(result[1]) > 0 else np.nan
            console.print(f"%K: {latest_k:.2f} | %D: {latest_d:.2f}")
            if latest_k > 80:
                console.print("[red]Overbought[/red]")
            elif latest_k < 20:
                console.print("[green]Oversold[/green]")
        elif indicator == 'ichimoku':
            console.print("Tenkan: N/A | Kijun: N/A | Senkou A/B: N/A")
        else:
            console.print("Multiple values returned")
        
        # Show table with last 10 rows
        _show_result_table(df, result, indicator)
        
    elif isinstance(result, dict):
        # Ichimoku returns dict
        for key, val in result.items():
            latest = val[-1] if len(val) > 0 else np.nan
            if not np.isnan(latest):
                console.print(f"{key}: {latest:.2f}")
        _show_result_table(df, result, indicator)
        
    else:
        # Single array result
        latest = result[-1] if len(result) > 0 else np.nan
        prev = result[-2] if len(result) > 1 else np.nan
        
        if not np.isnan(latest):
            color = "green" if latest > prev else "red" if latest < prev else "white"
            console.print(f"Current: [{color}]{latest:.4f}[/{color}]")
            console.print(f"Previous: {prev:.4f}")
            
            # Special signals
            if indicator == 'rsi':
                if latest > 70:
                    console.print("[bold red]OVERBOUGHT[/bold red]")
                elif latest < 30:
                    console.print("[bold green]OVERSOLD[/bold green]")
            elif indicator == 'atr':
                console.print(f"ATR as % of price: {(latest/df['Close'].iloc[-1])*100:.2f}%")
        
        _show_result_table(df, result, indicator)


def _show_result_table(df: pd.DataFrame, result, indicator: str):
    """Show recent indicator values in a table."""
    table = Table(title=f"Recent {indicator.upper()} Values", box=None)
    table.add_column("Date", style="cyan")
    
    if isinstance(result, tuple):
        labels = {
            'macd': ['MACD', 'Signal', 'Hist'],
            'bollinger_bands': ['Upper', 'Middle', 'Lower'],
            'stochastic': ['%K', '%D'],
            'keltner_channels': ['Upper', 'Middle', 'Lower'],
            'donchian_channels': ['Upper', 'Lower'],
        }.get(indicator, [f'Line{i+1}' for i in range(len(result))])
        
        for label in labels[:len(result)]:
            table.add_column(label, justify="right")
        
        for i in range(-10, 0):
            if abs(i) <= len(df):
                row = [df.index[i].strftime('%Y-%m-%d')]
                for arr in result:
                    if len(arr) > abs(i):
                        val = arr[i]
                        row.append(f"{val:.2f}" if not np.isnan(val) else "-")
                    else:
                        row.append("-")
                table.add_row(*row)
                
    elif isinstance(result, dict):
        keys = list(result.keys())
        for key in keys:
            table.add_column(key, justify="right")
        
        for i in range(-10, 0):
            if abs(i) <= len(df):
                row = [df.index[i].strftime('%Y-%m-%d')]
                for key in keys:
                    if len(result[key]) > abs(i):
                        val = result[key][i]
                        row.append(f"{val:.2f}" if not np.isnan(val) else "-")
                    else:
                        row.append("-")
                table.add_row(*row)
                
    else:
        table.add_column("Value", justify="right")
        table.add_column("Change", justify="right")
        
        for i in range(-10, 0):
            if abs(i) <= len(result):
                date = df.index[i].strftime('%Y-%m-%d')
                val = result[i]
                prev_val = result[i-1] if i > -len(result) else val
                change = val - prev_val if not np.isnan(val) and not np.isnan(prev_val) else 0
                
                val_str = f"{val:.4f}" if not np.isnan(val) else "-"
                change_str = f"{change:+.4f}" if change != 0 else "-"
                change_color = "green" if change > 0 else "red" if change < 0 else "white"
                
                table.add_row(date, val_str, f"[{change_color}]{change_str}[/{change_color}]")
    
    console.print(table)


@app.command("scan")
def scan_universe(
    indicator: str = typer.Option("rsi", "--indicator", "-i", help="Indicator to screen"),
    condition: str = typer.Option("oversold", "--condition", "-c", help="oversold, overbought"),
    max_results: int = typer.Option(10, "--max", "-n", help="Maximum results"),
):
    """
    Screen symbols for technical conditions.
    
    Example:
        quantterm tech scan --indicator rsi --condition oversold
    """
    console.print(f"[yellow]Scanning for {condition} {indicator}...[/yellow]")
    console.print("[dim]Note: Universe screening requires additional setup[/dim]")


@app.command("pattern")
def detect_patterns(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    pattern_type: str = typer.Option("harmonic", "--type", "-t", help="Pattern type"),
    timeframe: str = typer.Option("4h", "--timeframe", help="Timeframe"),
):
    """Detect chart patterns."""
    from datetime import datetime, timedelta
    
    try:
        # Get price data
        end_date = datetime.now()
        if timeframe == '1d':
            start_date = end_date - timedelta(days=30)
        elif timeframe == '4h':
            start_date = end_date - timedelta(days=60)
        elif timeframe == '1h':
            start_date = end_date - timedelta(days=30)
        else:
            start_date = end_date - timedelta(days=90)
        
        with console.status(f"Fetching data for {ticker}..."):
            df = yahoo_provider.get_history(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval=timeframe)
        
        if df is None or df.empty:
            console.print(f"[yellow]No data found for {ticker}[/yellow]")
            return
        
        # Simple price-based pattern detection
        prices = df['Close'].values
        recent_high = max(prices[-20:])
        recent_low = min(prices[-20:])
        current_price = prices[-1]
        
        console.print(Panel(
            f"[bold cyan]{ticker} Pattern Analysis ({timeframe})[/bold cyan]\n"
            f"Current Price: ${current_price:.2f}\n"
            f"20-Bar High: ${recent_high:.2f}\n"
            f"20-Bar Low: ${recent_low:.2f}\n\n"
            f"[yellow]Note:[/yellow] Advanced pattern detection requires additional\n"
            f"technical analysis libraries (e.g., ta-lib, pattern recognition ML models).",
            border_style="purple"
        ))
        
    except Exception as e:
        console.print(f"[red]Error analyzing patterns: {e}[/red]")


@app.command("levels")
def calculate_levels(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    level_type: str = typer.Option("fibonacci", "--type", "-t", help="Level type"),
    pivot: str = typer.Option("monthly", "--pivot", help="Pivot timeframe"),
):
    """Calculate support/resistance levels."""
    from datetime import datetime, timedelta
    
    try:
        # Get price data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)
        
        with console.status(f"Fetching data for {ticker}..."):
            df = yahoo_provider.get_history(ticker, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        
        if df is None or df.empty:
            console.print(f"[yellow]No data found for {ticker}[/yellow]")
            return
        
        # Calculate Fibonacci retracement levels
        prices = df['Close'].values
        high = max(prices)
        low = min(prices)
        current = prices[-1]
        range_size = high - low
        
        fib_levels = {
            '0.0% (Low)': low,
            '23.6%': low + range_size * 0.236,
            '38.2%': low + range_size * 0.382,
            '50.0%': low + range_size * 0.5,
            '61.8%': low + range_size * 0.618,
            '78.6%': low + range_size * 0.786,
            '100.0% (High)': high,
        }
        
        # Calculate pivot points
        pivot_high = df['High'].max()
        pivot_low = df['Low'].min()
        pivot_close = df['Close'].iloc[-1]
        pivot_s1 = (pivot_high + pivot_low + pivot_close) / 3 * 2 - pivot_high
        pivot_s2 = pivot_low + (pivot_high - pivot_low)
        pivot_r1 = (pivot_high + pivot_low + pivot_close) / 3 * 2 - pivot_low
        pivot_r2 = pivot_high + (pivot_high - pivot_low)
        
        console.print(Panel(
            f"[bold cyan]{ticker} Support/Resistance Levels[/bold cyan]\n"
            f"Current Price: ${current:.2f}\n\n"
            f"[bold]Fibonacci Retracements:[/bold]\n"
            + "\n".join([f"  {level}: ${price:.2f}" for level, price in fib_levels.items()]),
            border_style="purple",
            padding=(1, 1)
        ))
        
        console.print(Panel(
            f"[bold]Classic Pivot Points:[/bold]\n"
            f"  R2: ${pivot_r2:.2f}\n"
            f"  R1: ${pivot_r1:.2f}\n"
            f"  PP: ${pivot_close:.2f}\n"
            f"  S1: ${pivot_s1:.2f}\n"
            f"  S2: ${pivot_s2:.2f}",
            border_style="purple"
        ))
        
    except Exception as e:
        console.print(f"[red]Error calculating levels: {e}[/red]")


@app.command("list")
def list_indicators():
    """List all available technical indicators."""
    table = Table(title="Available Technical Indicators")
    table.add_column("Indicator", style="cyan", no_wrap=True)
    table.add_column("Category", style="magenta")
    table.add_column("Input Type", style="green")
    table.add_column("Parameters", style="yellow")
    
    indicators_info = {
        'sma': ('trend', 'prices', 'period'),
        'ema': ('trend', 'prices', 'period'),
        'wma': ('trend', 'prices', 'period'),
        'hma': ('trend', 'prices', 'period'),
        'macd': ('trend', 'prices', 'fast, slow, signal'),
        'rsi': ('momentum', 'prices', 'period'),
        'stochastic': ('momentum', 'ohlc', 'k_period, d_period'),
        'bollinger_bands': ('volatility', 'prices', 'period, std_dev'),
        'atr': ('volatility', 'ohlc', 'period'),
        'obv': ('volume', 'prices', 'none'),
        'vwap': ('volume', 'ohlcv', 'none'),
        'cci': ('momentum', 'ohlc', 'period'),
        'williams_r': ('momentum', 'ohlc', 'period'),
        'roc': ('momentum', 'prices', 'period'),
        'mfi': ('volume', 'ohlcv', 'period'),
        'keltner_channels': ('volatility', 'ohlc', 'period, multiplier'),
        'donchian_channels': ('volatility', 'ohlc', 'period'),
        'ichimoku': ('trend', 'ohlc', 'none'),
    }
    
    for name, (category, input_type, params) in sorted(indicators_info.items()):
        table.add_row(name, category, input_type, params)
    
    console.print(table)


if __name__ == "__main__":
    app()

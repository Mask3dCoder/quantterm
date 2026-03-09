"""Market data CLI commands."""
import sys
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
import typer

app = typer.Typer(name="market", help="Market data commands")

console = Console()

# Suppress yfinance stderr
class SuppressStderr:
    def __enter__(self):
        self._stderr = sys.stderr
        devnull = 'NUL' if sys.platform == 'win32' else '/dev/null'
        sys.stderr = open(devnull, 'w')
        return self
    def __exit__(self, *args):
        sys.stderr.close()
        sys.stderr = self._stderr


def format_value(key: str, value) -> str:
    """Format values based on their type and field name."""
    if value is None:
        return "—"
    
    # Large numbers
    if key in ('market_cap', 'volume', 'shares_outstanding'):
        if value >= 1e12:
            return f"{value/1e12:.2f}T"
        elif value >= 1e9:
            return f"{value/1e9:.2f}B"
        elif value >= 1e6:
            return f"{value/1e6:.2f}M"
        return f"{value:,.0f}"
    
    # Percentages
    if key in ('dividend_yield', 'yield', 'pe_ratio', 'eps'):
        return f"{value:.2f}"
    
    # Currency
    if key in ('price', 'bid', 'ask', 'target_mean_price', 'beta', 'fifty_day_average', 'two_hundred_day_average'):
        return f"${value:.2f}"
    
    # Dates/times
    if key in ('last_fink_date', 'ex_dividend_date', 'last_split_date'):
        if isinstance(value, datetime):
            return value.strftime('%Y-%m-%d')
        return str(value)
    
    # Booleans
    if isinstance(value, bool):
        return "✓" if value else "✗"
    
    # Default
    return str(value)


def format_field_name(key: str) -> str:
    """Convert snake_case to Title Case."""
    replacements = {
        'market_cap': 'Market Cap',
        'dividend_yield': 'Dividend Yield',
        '52w_high': '52W High',
        '52w_low': '52W Low',
        '52w_change': '52W Change',
        'avg_vol_10day': 'Avg Vol (10D)',
        'avg_vol_3month': 'Avg Vol (3M)',
        'pe_ratio': 'P/E Ratio',
        'eps': 'EPS',
        'beta': 'Beta',
        'last_fink_date': 'Last Fink Date',
        'ex_dividend_date': 'Ex-Dividend',
        'last_split_date': 'Last Split',
        'target_mean_price': 'Target Price',
        'recommendation': 'Analyst Rating',
        'recommendation_key': 'Rating',
    }
    return replacements.get(key, key.replace('_', ' ').title())


@app.command()
def quote(
    ticker: str = typer.Argument(..., help="Stock ticker symbol(s) - comma separated for multiple"),
):
    """Get real-time quote for a symbol."""
    from quantterm.data.providers.yahoo import get_quote
    
    # Split comma-separated tickers
    tickers = [t.strip().upper() for t in ticker.split(',')]
    
    for ticker_symbol in tickers:
        try:
            with SuppressStderr():
                data = get_quote(ticker_symbol)
            
            if not data:
                console.print(f"[yellow]No data found for {ticker_symbol}[/yellow]")
                continue
            
            # Create styled table
            table = Table(
                box=None,
                show_header=False,
                pad_edge=False,
            )
            table.add_column(style="cyan", width=20)
            table.add_column(style="white", width=15)
            
            # Key metrics to show at top
            priority_keys = ['price', 'change', 'change_percent', 'volume', 'market_cap']
            
            # Add priority fields first
            for key in priority_keys:
                if key in data and data[key] is not None and data[key] != 0:
                    val = data[key]
                    if key == 'change_percent':
                        val_str = f"{val:+.2f}%"
                        color = "green" if val >= 0 else "red"
                    elif key == 'change':
                        val_str = f"{val:+.2f}"
                        color = "green" if val >= 0 else "red"
                    else:
                        val_str = format_value(key, val)
                        color = "white"
                    table.add_row(format_field_name(key), Text(val_str, style=color))
            
            # Add separator
            table.add_row("", "")
            
            # Add remaining fields
            for key, value in data.items():
                if key not in priority_keys and value is not None and value != 0:
                    table.add_row(format_field_name(key), format_value(key, value))
            
            # Create panel with title
            title = f" {ticker_symbol} "
            subtitle = data.get('short_name', data.get('long_name', ''))
            if subtitle:
                title += f"• {subtitle}"
            
            console.print(Panel(
                table,
                title=title,
                border_style="purple",
                padding=(0, 1),
            ))
        except Exception as e:
            console.print(Panel(
                f"[red]Error:[/red] {e}",
                border_style="red",
            ))


@app.command()
def history(
    ticker: str = typer.Argument(..., help="Stock ticker symbol"),
    start: str = typer.Option("1y", "--start", "-s", help="Start date (1y, 6m, YYYY-MM-DD)"),
    end: str = typer.Option("today", "--end", "-e", help="End date"),
    interval: str = typer.Option("1d", "--interval", "-i", help="Data interval"),
):
    """Get historical price data."""
    from quantterm.data.providers.yahoo import get_history
    
    def parse_relative_date(date_str: str) -> str:
        """Convert relative dates like 1y, 6m to YYYY-MM-DD."""
        from datetime import datetime, timedelta
        now = datetime.now()
        
        if date_str in ('today', 'now'):
            return now.strftime('%Y-%m-%d')
        
        # Parse relative strings
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
            '10y': timedelta(days=3650),
        }
        
        if date_str.lower() in mapping:
            delta = mapping[date_str.lower()]
            return (now - delta).strftime('%Y-%m-%d')
        
        # Already a date
        return date_str
    
    try:
        # Parse relative dates
        start_date = parse_relative_date(start)
        end_date = None if end == "today" else (parse_relative_date(end) if end != 'today' else None)
        
        with SuppressStderr():
            df = get_history(ticker, start=start_date, end=end_date, interval=interval)
        
        if df is None or df.empty:
            console.print(Panel(
                f"[yellow]No data found for {ticker}[/yellow]",
                border_style="yellow",
            ))
            return
        
        # Format the dataframe for display
        from rich.console import Console
        from rich.table import Table
        
        table = Table(
            title=f"{ticker.upper()} History",
            box=None,
            header_style="bold purple",
        )
        
        # Add columns
        table.add_column("Date", style="cyan", width=12)
        table.add_column("Open", justify="right", width=10)
        table.add_column("High", justify="right", width=10)
        table.add_column("Low", justify="right", width=10)
        table.add_column("Close", justify="right", width=10)
        table.add_column("Volume", justify="right", width=12)
        
        # Show last 10 rows
        for idx, row in df.tail(10).iterrows():
            date_str = idx.strftime('%Y-%m-%d') if isinstance(idx, datetime) else str(idx)[:10]
            table.add_row(
                date_str,
                f"${row['Open']:.2f}",
                f"${row['High']:.2f}",
                f"${row['Low']:.2f}",
                f"${row['Close']:.2f}",
                f"{int(row['Volume']):,}",
            )
        
        console.print(Panel(
            table,
            border_style="purple",
            padding=(0, 0),
        ))
        
        # Show summary stats
        latest = df.iloc[-1]
        first = df.iloc[0]
        change = latest['Close'] - first['Close']
        change_pct = (change / first['Close']) * 100
        
        console.print(f"\n[cyan]Summary:[/cyan] {len(df)} rows | "
                     f"[green]${first['Close']:.2f}[/green] -> [green]${latest['Close']:.2f}[/green] "
                     f"([{'green' if change >= 0 else 'red'}]{change_pct:+.2f}%[/])")
    except Exception as e:
        console.print(Panel(
            f"[red]Error:[/red] {e}",
            border_style="red",
        ))


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
):
    """Search for securities."""
    from quantterm.data.providers.yahoo import search_ticker
    
    try:
        with SuppressStderr():
            results = search_ticker(query)
        
        if not results:
            console.print(f"[yellow]No results for '{query}'[/yellow]")
            return
        
        table = Table(
            title=f"Search: {query}",
            box=None,
            header_style="bold purple",
        )
        table.add_column("Symbol", style="cyan", width=8)
        table.add_column("Name", style="white")
        table.add_column("Type", style="dim", width=10)
        table.add_column("Exchange", style="dim", width=10)
        
        for r in results[:10]:
            table.add_row(
                r.get('symbol', ''),
                r.get('longname', r.get('shortname', '')),
                r.get('type', ''),
                r.get('exchange', ''),
            )
        
        console.print(Panel(table, border_style="purple"))
    except Exception as e:
        console.print(Panel(
            f"[red]Error:[/red] {e}",
            border_style="red",
        ))

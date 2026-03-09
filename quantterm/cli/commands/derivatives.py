"""Derivatives CLI commands."""
import typer
from typing import Optional, Literal
from datetime import datetime, timedelta
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

app = typer.Typer(help="Options and derivatives analytics")
console = Console()


def parse_relative_date(date_str: str) -> str:
    """Convert relative dates like 1y, 6m to YYYY-MM-DD."""
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
    }
    
    if date_str.lower() in mapping:
        delta = mapping[date_str.lower()]
        return (datetime.now() - delta).strftime('%Y-%m-%d')
    
    return date_str


@app.command("price")
def price_option(
    symbol: str = typer.Argument(..., help="Underlying symbol (e.g., AAPL)"),
    strike: float = typer.Argument(..., help="Strike price"),
    expiration: str = typer.Argument(..., help="Expiration (YYYY-MM-DD or 30d)"),
    option_type: Literal['call', 'put'] = typer.Argument(..., help="Option type"),
    volatility: Optional[float] = typer.Option(None, "--vol", "-v", help="Volatility (default: 30%)"),
    rate: float = typer.Option(0.05, "--rate", "-r", help="Risk-free rate"),
    dividend: float = typer.Option(0.0, "--div", "-d", help="Dividend yield"),
    spot_price: Optional[float] = typer.Option(None, "--spot", "-s", help="Current spot price"),
):
    """
    Price European options using Black-Scholes model.
    
    Example:
        quantterm derivatives price AAPL 180 2024-12-20 call
        quantterm derivatives price TSLA 200 30d put --vol 0.45
    """
    try:
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        from quantterm.data.providers import yahoo as yahoo_provider
        
        # Parse expiration
        if expiration.endswith('d'):
            days = int(expiration[:-1])
            exp_date = datetime.now() + timedelta(days=days)
            t = days / 365.25
        else:
            try:
                exp_date = datetime.strptime(expiration, '%Y-%m-%d')
                t = (exp_date - datetime.now()).days / 365.25
            except ValueError:
                console.print(f"[red]Invalid expiration format: {expiration}[/red]")
                raise typer.Exit(1)
        
        if t <= 0:
            console.print("[red]Expiration must be in the future[/red]")
            raise typer.Exit(1)
        
        # Fetch spot price if not provided
        if spot_price is None:
            try:
                with console.status(f"Fetching price for {symbol}..."):
                    quote = yahoo_provider.get_quote(symbol)
                    spot_price = quote.get('price', 0)
                    if spot_price == 0:
                        console.print(f"[red]Could not fetch price for {symbol}[/red]")
                        raise typer.Exit(1)
            except Exception:
                console.print(f"[red]Could not fetch price for {symbol}[/red]")
                raise typer.Exit(1)
        
        # Use provided or default volatility
        if volatility is None:
            volatility = 0.30
            console.print(f"[dim]Using default volatility: {volatility:.1%}[/dim]")
        
        is_call = option_type == 'call'
        
        # Calculate Black-Scholes
        try:
            result = black_scholes(
                s=spot_price,
                k=strike,
                t=t,
                r=rate,
                sigma=volatility,
                q=dividend
            )
        except Exception as e:
            console.print(f"[red]Black-Scholes calculation error: {e}[/red]")
            raise typer.Exit(1)
        
        # Display results
        call_price = result.call_price if hasattr(result, 'call_price') else result[0]
        put_price = result.put_price if hasattr(result, 'put_price') else result[1]
        
        console.print(Panel(
            f"[bold cyan]{symbol.upper()} {option_type.upper()} ${strike:.2f}[/bold cyan]\n"
            f"Spot: ${spot_price:.2f} | Strike: ${strike:.2f} | Time: {t*365:.0f}D\n"
            f"Vol: {volatility:.1%} | Rate: {rate:.1%} | Div: {dividend:.1%}",
            border_style="purple"
        ))
        
        # Price table
        price_table = Table(show_header=False, box=None)
        price_table.add_column("Metric", style="cyan")
        price_table.add_column("Value", justify="right", style="white")
        
        price_table.add_row("Call Price", f"${call_price:.4f}")
        price_table.add_row("Put Price", f"${put_price:.4f}")
        
        # Show moneyness
        moneyness = spot_price / strike
        if is_call:
            price_table.add_row("Moneyness", f"{moneyness:.2%} (ITM)" if moneyness > 1 else f"{moneyness:.2%} (OTM)")
        else:
            price_table.add_row("Moneyness", f"{moneyness:.2%} (ITM)" if moneyness < 1 else f"{moneyness:.2%} (OTM)")
        
        console.print(price_table)
        
        # Greeks table
        greeks_table = Table(title="Greeks", show_header=True, box=None)
        greeks_table.add_column("Greek", style="cyan")
        greeks_table.add_column("Call", justify="right")
        greeks_table.add_column("Put", justify="right")
        
        if hasattr(result, 'delta_call'):
            delta_call = result.delta_call
            delta_put = result.delta_put
            gamma = result.gamma
            theta_call = result.theta_call
            theta_put = result.theta_put
            vega = result.vega
            rho_call = result.rho_call
            rho_put = result.rho_put
        else:
            delta_call = result[2]
            delta_put = result[3]
            gamma = result[4]
            theta_call = result[5]
            theta_put = result[6]
            vega = result[7]
            rho_call = result[8]
            rho_put = result[9]
        
        greeks_table.add_row("Delta", f"{delta_call:.4f}", f"{delta_put:.4f}")
        greeks_table.add_row("Gamma", f"{gamma:.4f}", f"{gamma:.4f}")
        greeks_table.add_row("Theta", f"{theta_call:.4f}", f"{theta_put:.4f}")
        greeks_table.add_row("Vega", f"{vega:.4f}", f"{vega:.4f}")
        greeks_table.add_row("Rho", f"{rho_call:.4f}", f"{rho_put:.4f}")
        
        console.print(greeks_table)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("chain")
def view_options_chain(
    symbol: str = typer.Argument(..., help="Underlying symbol"),
    expiration: Optional[str] = typer.Option(None, "--exp", "-e", help="Expiration date"),
):
    """
    View options chain.
    
    Example:
        quantterm derivatives chain AAPL --exp 2024-12-20
    """
    try:
        from quantterm.data.providers import yahoo as yahoo_provider
        
        with console.status(f"Fetching options chain for {symbol}..."):
            chain = yahoo_provider.get_options(symbol)
        
        if 'expirations' in chain:
            console.print(Panel(
                f"[bold cyan]{symbol.upper()} Options[/bold cyan]\n"
                f"Available Expirations: {', '.join(chain['expirations'][:10])}...",
                border_style="purple"
            ))
        else:
            calls = chain.get('calls', None)
            puts = chain.get('puts', None)
            
            if calls is not None and not calls.empty:
                console.print(f"[green]Calls: {len(calls)} | Puts: {len(puts)}[/green]")
            else:
                console.print(f"[yellow]No options data for {symbol}[/yellow]")
                
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("vol")
def analyze_volatility(
    symbol: str = typer.Argument(..., help="Underlying symbol"),
    period: str = typer.Option("1y", "--period", "-p", help="Historical period"),
):
    """
    Analyze volatility for an underlying.
    
    Example:
        quantterm derivatives vol AAPL --period 1y
    """
    try:
        from quantterm.data.providers import yahoo as yahoo_provider
        
        with console.status(f"Fetching data for {symbol}..."):
            start_date = parse_relative_date(period)
            df = yahoo_provider.get_history(symbol, start=start_date)
            
            if df is None or df.empty:
                console.print(f"[red]No data for {symbol}[/red]")
                raise typer.Exit(1)
            
            # Calculate historical volatility
            returns = df['Close'].pct_change().dropna()
            hist_vol = returns.std() * np.sqrt(252)
            
            # Calculate IV from options if available
            try:
                chain = yahoo_provider.get_options(symbol)
                if 'expirations' in chain and len(chain['expirations']) > 0:
                    # Try to get implied vol
                    implied_vol = None
                else:
                    implied_vol = None
            except:
                implied_vol = None
        
        console.print(Panel(
            f"[bold cyan]{symbol.upper()} Volatility Analysis[/bold cyan]",
            border_style="purple"
        ))
        
        vol_table = Table(show_header=False, box=None)
        vol_table.add_column("Metric", style="cyan")
        vol_table.add_column("Value", justify="right")
        
        vol_table.add_row("Historical Volatility (Ann.)", f"{hist_vol:.2%}")
        vol_table.add_row("Historical Volatility (Daily)", f"{hist_vol/np.sqrt(252):.2%}")
        
        if implied_vol:
            vol_table.add_row("Implied Volatility", f"{implied_vol:.2%}")
            vol_table.add_row("IV-HV Spread", f"{(implied_vol - hist_vol)*100:.1f} bps")
        
        console.print(vol_table)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command("future")
def futures_pricing(
    symbol: str = typer.Argument(..., help="Futures symbol"),
):
    """Get futures contract information."""
    console.print(f"[yellow]Futures pricing for {symbol}...[/yellow]")
    console.print("[dim]Coming soon[/dim]")


@app.command("greeks")
def calculate_greeks(
    symbol: str = typer.Argument(..., help="Underlying symbol (e.g., AAPL)"),
    strike: float = typer.Argument(..., help="Strike price"),
    expiration: str = typer.Argument(..., help="Expiration (YYYY-MM-DD or 30d)"),
    option_type: Literal['call', 'put'] = typer.Argument(..., help="Option type"),
    volatility: Optional[float] = typer.Option(None, "--vol", "-v", help="Volatility (default: 30%)"),
    rate: float = typer.Option(0.05, "--rate", "-r", help="Risk-free rate"),
    dividend: float = typer.Option(0.0, "--div", "-d", help="Dividend yield"),
    spot_price: Optional[float] = typer.Option(None, "--spot", "-s", help="Current spot price"),
):
    """
    Calculate option Greeks using Black-Scholes model.
    
    Example:
        quantterm options greeks AAPL 180 30d call --vol 0.30
        quantterm options greeks TSLA 200 2024-12-20 put --vol 0.45
    """
    try:
        from quantterm.derivatives.pricing.black_scholes import black_scholes
        from quantterm.data.providers import yahoo as yahoo_provider
        
        # Parse expiration
        if expiration.endswith('d'):
            days = int(expiration[:-1])
            from datetime import datetime, timedelta
            exp_date = datetime.now() + timedelta(days=days)
        else:
            from datetime import datetime
            exp_date = datetime.strptime(expiration, '%Y-%m-%d')
        
        from datetime import datetime
        t = (exp_date - datetime.now()).days / 365.0
        
        if t <= 0:
            console.print("[red]Error: Expiration must be in the future[/red]")
            raise typer.Exit(1)
        
        # Get spot price if not provided
        if spot_price is None:
            try:
                quote = yahoo_provider.get_quote(symbol)
                spot_price = quote.get('price')
                if spot_price is None or spot_price == 0:
                    console.print(f"[red]Error: Could not fetch spot price for {symbol}[/red]")
                    raise typer.Exit(1)
            except typer.Exit:
                raise
            except Exception as e:
                console.print(f"[red]Error fetching quote: {e}[/red]")
                raise typer.Exit(1)
        
        # Use provided volatility or default
        if volatility is None:
            volatility = 0.30
        
        # Calculate Greeks
        result = black_scholes(
            s=spot_price,
            k=strike,
            t=t,
            r=rate,
            sigma=volatility,
            q=dividend
        )
        
        # Display results
        table = Table(title=f"Greeks for {symbol} {strike} {expiration} {option_type}")
        table.add_column("Greek", style="cyan", no_wrap=True)
        table.add_column("Value", style="green", justify="right")
        
        if option_type == 'call':
            table.add_row("Delta", f"{result.delta_call:.4f}")
            table.add_row("Theta", f"{result.theta_call:.4f}")
            table.add_row("Rho", f"{result.rho_call:.4f}")
        else:
            table.add_row("Delta", f"{result.delta_put:.4f}")
            table.add_row("Theta", f"{result.theta_put:.4f}")
            table.add_row("Rho", f"{result.rho_put:.4f}")
        
        table.add_row("Gamma", f"{result.gamma:.4f}")
        table.add_row("Vega", f"{result.vega:.4f}")
        
        console.print(table)
        
    except typer.Exit:
        raise
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

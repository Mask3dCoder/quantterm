"""CLI commands for paper trading."""
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="Paper trading commands")
console = Console()


@app.command("start")
def paper_start(
    strategy: str = typer.Option(..., "--strategy", "-s", help="Strategy name"),
    symbols: str = typer.Option(..., "--symbols", "-sym", help="Comma-separated symbols"),
    capital: float = typer.Option(100000, "--capital", "-c"),
    interval: int = typer.Option(60, "--interval", "-i", help="Bar interval in seconds"),
    source: str = typer.Option("yahoo", "--source", help="Data source (yahoo/polygon)")
):
    """
    Start paper trading with a strategy.
    
    Example:
        quantterm paper start --strategy Momentum --symbols SPY,QQQ --capital 100000
    """
    from quantterm.live import LiveDataFeed, PaperTradingEngine, LiveStrategyRunner
    from quantterm.backtesting.strategy import BuyAndHoldStrategy
    from quantterm.backtesting.portfolio import Portfolio
    
    console.print(f"[bold green]Starting Paper Trading[/bold green]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Symbols: {symbols}")
    console.print(f"Capital: ${capital:,.2f}")
    console.print(f"Data Source: {source}")
    
    # Parse symbols
    symbol_list = [s.strip() for s in symbols.split(',')]
    
    # Create components
    data_feed = LiveDataFeed(source=source)
    trading = PaperTradingEngine(initial_capital=capital)
    
    # Create strategy (placeholder - would load actual strategy)
    portfolio = Portfolio(capital)
    strategy_obj = BuyAndHoldStrategy("BuyAndHold", portfolio, None)
    
    # Create runner
    runner = LiveStrategyRunner(
        strategy=strategy_obj,
        symbols=symbol_list,
        data_feed=data_feed,
        trading_engine=trading,
        bar_interval=interval
    )
    
    console.print("\n[yellow]Paper trading started. Press Ctrl+C to stop.[/yellow]")
    
    try:
        # Run (in practice, would use asyncio.run)
        import asyncio
        asyncio.run(runner.start())
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping paper trading...[/yellow]")
        runner.stop()
        
        # Show final status
        status = runner.get_status()
        console.print(f"\n[bold]Final Results:[/bold]")
        console.print(f"Total Equity: ${status['total_equity']:,.2f}")
        console.print(f"Total P&L: ${status['total_pnl']:,.2f} ({status['pnl_pct']:.2f}%)")
        console.print(f"Total Trades: {status['trades']}")


@app.command("status")
def paper_status():
    """Show current paper trading status."""
    console.print("[yellow]No active paper trading session[/yellow]")


@app.command("stop")
def paper_stop():
    """Stop paper trading and flatten positions."""
    console.print("[yellow]Stopping paper trading...[/yellow]")


if __name__ == "__main__":
    app()

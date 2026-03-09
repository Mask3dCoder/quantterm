"""QuantTerm CLI main application."""
import warnings
import typer
from rich.console import Console
from rich.theme import Theme

# Import utilities
from quantterm.cli.utils import SuppressStderr

# Suppress all warnings for cleaner output
warnings.filterwarnings('ignore')

from quantterm import __version__

# Custom theme for the CLI
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "success": "bold green",
    "command": "bold blue",
})

console = Console(theme=custom_theme)

app = typer.Typer(
    name="quantterm",
    help="QuantTerm - Institutional-grade quantitative finance CLI",
    add_completion=False,
    invoke_without_command=True,
)


@app.callback(invoke_without_command=True)
def callback(ctx: typer.Context):
    """Default callback - show help when no command is provided."""
    if ctx.invoked_subcommand is None:
        console.print("[bold blue]QuantTerm[/bold blue] v0.1.0")
        console.print("[info]Institutional-grade quantitative finance CLI[/info]\n")
        console.print("[info]Run 'quantterm --help' for available commands.[/info]")
        console.print("[info]Quick start: quantterm quote AAPL[/info]")


@app.command()
def version():
    """Show QuantTerm version."""
    console.print(f"[bold blue]QuantTerm[/bold blue] v{__version__}")
    console.print("[info]Institutional-grade quantitative finance CLI[/info]")


@app.command()
def info():
    """Show system information."""
    import platform
    import sys

    console.print("[bold]System Information[/bold]")
    console.print(f"Python: {sys.version}")
    console.print(f"Platform: {platform.platform()}")
    console.print(f"Processor: {platform.processor()}")


# Import command groups
from quantterm.cli.commands import market_data  # noqa: E402
from quantterm.cli.commands import technical  # noqa: E402
from quantterm.cli.commands import derivatives  # noqa: E402
from quantterm.cli.commands import portfolio  # noqa: E402
from quantterm.cli.commands import risk  # noqa: E402
from quantterm.cli.commands import backtest  # noqa: E402
from quantterm.cli.commands import ml  # noqa: E402
from quantterm.cli.commands import paper  # noqa: E402
from quantterm.cli.commands import optimize  # noqa: E402
from quantterm.cli.commands import fixed_income  # noqa: E402
from quantterm.cli.commands import config  # noqa: E402

# Cache subcommand
cache_app = typer.Typer(name="cache", help="Cache management commands")


@cache_app.command()
def stats():
    """Show cache statistics (hit rate, size, entries)."""
    from quantterm.data.cache import get_cache
    
    cache = get_cache()
    stats = cache.get_stats()
    
    console.print("\n[bold]Cache Statistics[/bold]")
    console.print(f"  Hits:     {stats.hits}")
    console.print(f"  Misses:   {stats.misses}")
    console.print(f"  Hit Rate: {stats.hit_rate:.1f}%")
    console.print(f"  Entries:  {stats.entries}")
    console.print(f"  Size:     {stats.size_bytes / (1024*1024):.2f} MB")


@cache_app.command()
def clear():
    """Clear all cached data."""
    from quantterm.data.cache import get_cache
    
    cache = get_cache()
    count = cache.clear()
    console.print(f"[success]Cleared {count} entries from cache.[/success]")


@app.command()
def cache():
    """Cache management commands."""
    """Placeholder - use 'quantterm cache stats' or 'quantterm cache clear'."""
    pass


# Register command groups - add market_data directly so quote, history, search work directly
app.add_typer(market_data.app, name=None)  # Commands become top-level: quote, history, search
app.add_typer(technical.app, name="tech")
app.add_typer(technical.app, name="pattern")
app.add_typer(technical.app, name="levels")
app.add_typer(derivatives.app, name="options")
app.add_typer(derivatives.app, name="vol")
app.add_typer(derivatives.app, name="future")
app.add_typer(derivatives.app, name="price")
app.add_typer(portfolio.app, name="portfolio")
app.add_typer(optimize.app, name="optimize")
app.add_typer(risk.app, name="risk")
app.add_typer(backtest.app, name="backtest")
app.add_typer(ml.app, name="ml")
app.add_typer(paper.app, name="paper")
app.add_typer(fixed_income.app, name="fixed-income")
app.add_typer(cache_app, name="cache")
app.add_typer(config.app, name="config")


def main():
    """Main entry point."""
    # Suppress yfinance warnings during command execution
    with SuppressStderr():
        app()


if __name__ == "__main__":
    main()

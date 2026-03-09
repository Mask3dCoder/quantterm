"""CLI commands for Fixed Income analytics - bonds, yield curves, and portfolios."""
import typer
from rich.console import Console
from rich.table import Table
from datetime import date, timedelta

app = typer.Typer(help="Fixed income analytics commands")
console = Console()


@app.command("price")
def price_bond(
    coupon: float = typer.Option(..., "--coupon", "-c", help="Annual coupon rate (e.g., 0.05 for 5%)"),
    maturity: str = typer.Option(..., "--maturity", "-m", help="Maturity date (YYYY-MM-DD)"),
    face_value: float = typer.Option(1000.0, "--face", "-f", help="Face value"),
    yield_rate: float = typer.Option(..., "--yield", "-y", help="Yield to maturity (e.g., 0.04 for 4%)"),
    frequency: int = typer.Option(2, "--frequency", "--freq", help="Coupon frequency (1=annual, 2=semi-annual)"),
    settlement: str = typer.Option(None, "--settlement", "-s", help="Settlement date (YYYY-MM-DD, default: today)")
):
    """Price a bond given coupon, maturity, and yield.
    
    Calculates both clean and dirty price of a fixed-rate bond.
    
    Example:
        quantterm fixed-income price --coupon 0.05 --maturity 2030-01-01 --yield 0.04
    """
    from quantterm.fixed_income.bonds import Bond, BondAnalytics
    
    # Parse dates
    maturity_date = date.fromisoformat(maturity)
    if settlement:
        settlement_date = date.fromisoformat(settlement)
    else:
        settlement_date = date.today()
    
    # Create bond
    bond = Bond(
        cusip="CLI_BOND",
        coupon=coupon,
        maturity=maturity_date,
        face_value=face_value,
        frequency=frequency
    )
    
    # Calculate prices
    dirty_price = BondAnalytics.price(bond, yield_rate, settlement_date)
    clean_price = BondAnalytics.clean_price(bond, yield_rate, settlement_date)
    accrued = BondAnalytics.accrued_interest(bond, settlement_date)
    
    # Display results
    console.print(f"[bold]Bond Pricing[/bold]")
    console.print(f"Maturity: {maturity_date}")
    console.print(f"Coupon: {coupon*100:.2f}%")
    console.print(f"Yield: {yield_rate*100:.2f}%")
    console.print("-" * 40)
    
    price_table = Table(title="Bond Prices")
    price_table.add_column("Price Type", style="cyan")
    price_table.add_column("Amount", style="green")
    
    price_table.add_row("Dirty Price", f"${dirty_price:,.2f}")
    price_table.add_row("Clean Price", f"${clean_price:,.2f}")
    price_table.add_row("Accrued Interest", f"${accrued:,.2f}")
    price_table.add_row("Face Value", f"${face_value:,.2f}")
    
    console.print(price_table)


@app.command("yield")
def calculate_ytm(
    coupon: float = typer.Option(..., "--coupon", "-c", help="Annual coupon rate"),
    maturity: str = typer.Option(..., "--maturity", "-m", help="Maturity date (YYYY-MM-DD)"),
    price: float = typer.Option(..., "--price", "-p", help="Market price"),
    face_value: float = typer.Option(1000.0, "--face", "-f", help="Face value"),
    frequency: int = typer.Option(2, "--frequency", help="Coupon frequency"),
    settlement: str = typer.Option(None, "--settlement", "-s", help="Settlement date")
):
    """Calculate yield to maturity given bond price.
    
    Solves for YTM using Newton-Raphson iteration.
    
    Example:
        quantterm fixed-income yield --coupon 0.05 --maturity 2030-01-01 --price 1050
    """
    from quantterm.fixed_income.bonds import Bond, BondAnalytics
    
    # Parse dates
    maturity_date = date.fromisoformat(maturity)
    if settlement:
        settlement_date = date.fromisoformat(settlement)
    else:
        settlement_date = date.today()
    
    # Create bond
    bond = Bond(
        cusip="CLI_BOND",
        coupon=coupon,
        maturity=maturity_date,
        face_value=face_value,
        frequency=frequency
    )
    
    # Calculate YTM
    try:
        ytm = BondAnalytics.yield_to_maturity(bond, price, settlement_date)
        
        console.print(f"[bold]Yield to Maturity Calculation[/bold]")
        console.print(f"Coupon: {coupon*100:.2f}%")
        console.print(f"Maturity: {maturity_date}")
        console.print(f"Price: ${price:,.2f}")
        console.print("-" * 40)
        console.print(f"[green]Yield to Maturity: {ytm*100:.4f}%[/green]")
        
        # Also show duration and convexity
        duration = BondAnalytics.modified_duration(bond, ytm, settlement_date)
        convexity = BondAnalytics.convexity(bond, ytm, settlement_date)
        
        console.print(f"\n[bold]Risk Metrics:[/bold]")
        console.print(f"Modified Duration: {duration:.4f}")
        console.print(f"Convexity: {convexity:.4f}")
        
    except Exception as e:
        console.print(f"[red]Error calculating YTM: {e}[/red]")


@app.command("duration")
def calculate_duration(
    coupon: float = typer.Option(..., "--coupon", "-c", help="Annual coupon rate"),
    maturity: str = typer.Option(..., "--maturity", "-m", help="Maturity date (YYYY-MM-DD)"),
    yield_rate: float = typer.Option(..., "--yield", "-y", help="Yield to maturity"),
    face_value: float = typer.Option(1000.0, "--face", "-f", help="Face value"),
    frequency: int = typer.Option(2, "--frequency", help="Coupon frequency"),
    settlement: str = typer.Option(None, "--settlement", "-s", help="Settlement date")
):
    """Calculate bond duration and risk metrics.
    
    Computes both Macaulay and Modified duration, plus convexity.
    
    Example:
        quantterm fixed-income duration --coupon 0.05 --maturity 2030-01-01 --yield 0.04
    """
    from quantterm.fixed_income.bonds import Bond, BondAnalytics
    
    # Parse dates
    maturity_date = date.fromisoformat(maturity)
    if settlement:
        settlement_date = date.fromisoformat(settlement)
    else:
        settlement_date = date.today()
    
    # Create bond
    bond = Bond(
        cusip="CLI_BOND",
        coupon=coupon,
        maturity=maturity_date,
        face_value=face_value,
        frequency=frequency
    )
    
    # Calculate metrics
    macaulay = BondAnalytics.macaulay_duration(bond, yield_rate, settlement_date)
    modified = BondAnalytics.modified_duration(bond, yield_rate, settlement_date)
    convexity = BondAnalytics.convexity(bond, yield_rate, settlement_date)
    dv01 = BondAnalytics.dv01(bond, yield_rate, settlement_date)
    
    # Display
    console.print(f"[bold]Bond Risk Metrics[/bold]")
    console.print(f"Coupon: {coupon*100:.2f}% | Yield: {yield_rate*100:.2f}%")
    console.print(f"Maturity: {maturity_date}")
    console.print("-" * 40)
    
    metrics_table = Table()
    metrics_table.add_column("Metric", style="cyan")
    metrics_table.add_column("Value", style="green")
    metrics_table.add_column("Interpretation", style="dim")
    
    metrics_table.add_row("Macaulay Duration", f"{macaulay:.4f} years", "Weighted avg time to cash flows")
    metrics_table.add_row("Modified Duration", f"{modified:.4f}", "% change per 1% yield change")
    metrics_table.add_row("Convexity", f"{convexity:.4f}", "Curvature adjustment")
    metrics_table.add_row("DV01", f"${dv01:.4f}", "Price change per 1bp yield change")
    
    console.print(metrics_table)
    
    # Estimate price change
    console.print("\n[bold]Price Sensitivity:[/bold]")
    for shift in [-0.01, -0.005, 0.005, 0.01]:
        new_yield = yield_rate + shift
        new_price = BondAnalytics.price(bond, new_yield, settlement_date)
        original_price = BondAnalytics.price(bond, yield_rate, settlement_date)
        pct_change = (new_price - original_price) / original_price * 100
        
        console.print(f"  Yield {shift:+.1%}: Price {pct_change:+.2f}%")


@app.command("curve")
def analyze_yield_curve(
    tenor: str = typer.Option(..., "--tenor", "-t", help="Tenor in years (comma-separated, e.g., 0.25,0.5,1,2,5,10,30)"),
    rates: str = typer.Option(..., "--rates", "-r", help="Yield rates (comma-separated, e.g., 5.2,5.5,5.8,6.1,6.5,6.8,7.0)"),
    interpolate_tenor: float = typer.Option(None, "--interp", help="Tenor to interpolate")
):
    """Analyze and interpolate yield curve.
    
    Builds a yield curve from points and can interpolate
    any maturity using cubic splines.
    
    Example:
        quantterm fixed-income curve --tenor 1,2,5,10,30 --rates 5.5,5.8,6.2,6.5,6.8 --interp 7
    """
    from quantterm.fixed_income.yield_curve import YieldCurve
    
    # Parse tenors and rates
    try:
        tenors = [float(x.strip()) for x in tenor.split(",")]
        rate_values = [float(x.strip()) / 100 for x in rates.split(",")]  # Convert from %
    except ValueError as e:
        console.print(f"[red]Error parsing input: {e}[/red]")
        raise typer.Exit(1)
    
    if len(tenors) != len(rate_values):
        console.print("[red]Number of tenors must match number of rates[/red]")
        raise typer.Exit(1)
    
    # Build curve
    curve = YieldCurve(valuation_date=date.today())
    
    for t, r in zip(tenors, rate_values):
        curve.add_point(t, r)
    
    console.print(f"[bold]Yield Curve Analysis[/bold]")
    console.print("-" * 40)
    
    # Show input points
    points_table = Table(title="Input Yield Curve Points")
    points_table.add_column("Tenor", style="cyan")
    points_table.add_column("Rate", style="green")
    
    for t, r in zip(tenors, rate_values):
        points_table.add_row(f"{t} years", f"{r*100:.2f}%")
    
    console.print(points_table)
    
    # Interpolate if requested
    if interpolate_tenor:
        interp_rate = curve.zero_rate(interpolate_tenor)
        console.print(f"\n[green]Interpolated rate at {interpolate_tenor} years: {interp_rate*100:.2f}%[/green]")
        
        # Show nearby points for context
        for t in [interpolate_tenor - 1, interpolate_tenor, interpolate_tenor + 1]:
            if t > 0:
                r = curve.zero_rate(t)
                console.print(f"  {t:.1f}y: {r*100:.2f}%")
    
    # Show forward rates
    console.print("\n[bold]Implied Forward Rates:[/bold]")
    if len(tenors) >= 2:
        for i in range(len(tenors) - 1):
            fwd_rate = curve.forward_rate(tenors[i], tenors[i+1])
            console.print(f"  {tenors[i]}y → {tenors[i+1]}y: {fwd_rate*100:.2f}%")


@app.command("fred")
def fetch_fred_data(
    series: str = typer.Option(..., "--series", "-s", help="FRED series ID (e.g., DGS10, DGS2, T10Y2Y)"),
    start: str = typer.Option("1y", "--start", help="Start date or period (1y, 6m, etc.)"),
    end: str = typer.Option("today", "--end", help="End date or period")
):
    """Fetch economic data from FRED (Federal Reserve Economic Data).
    
    Downloads Treasury yields, spreads, and other macro indicators.
    
    Common series:
    - DGS10: 10-Year Treasury Rate
    - DGS2: 2-Year Treasury Rate
    - DGS30: 30-Year Treasury Rate
    - T10Y2Y: 10-Year minus 2-Year Spread
    - T10Y3M: 10-Year minus 3-Month Spread
    
    Example:
        quantterm fixed-income fred --series DGS10 --start 2020-01-01 --end 2023-12-31
    """
    from quantterm.fixed_income.fred_data import FREDDataProvider
    
    console.print(f"[bold]Fetching FRED Data[/bold]")
    console.print(f"Series: {series}")
    console.print("-" * 40)
    
    # Create provider
    provider = FREDDataProvider()
    
    try:
        # Fetch data
        data = provider.get_series(series, start, end)
        
        if data.empty:
            console.print("[yellow]No data returned. Check the series ID.[/yellow]")
            return
        
        console.print(f"[green]Retrieved {len(data)} observations[/green]")
        console.print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Show latest values
        console.print(f"\n[bold]Latest Values:[/bold]")
        console.print(f"  Latest: {data.iloc[-1]:.4f}")
        console.print(f"  Mean: {data.mean():.4f}")
        console.print(f"  Std Dev: {data.std():.4f}")
        console.print(f"  Min: {data.min():.4f}")
        console.print(f"  Max: {data.max():.4f}")
        
        # Show recent values
        console.print(f"\n[bold]Recent History:[/bold]")
        recent = data.tail(10)
        for idx, val in recent.items():
            console.print(f"  {idx.date()}: {val:.4f}")
            
    except Exception as e:
        console.print(f"[red]Error fetching FRED data: {e}[/red]")
        console.print("[yellow]Note: FRED API key may be required. Set FRED_API_KEY environment variable.[/yellow]")


@app.command("portfolio")
def analyze_bond_portfolio(
    bonds: str = typer.Option(..., "--bonds", "-b", help="Bond specs (format: coupon,maturity,face,yield separated by semicolons)"),
    settlement: str = typer.Option(None, "--settlement", "-s", help="Settlement date")
):
    """Analyze a bond portfolio.
    
    Calculates portfolio-level duration, convexity, and yield sensitivity.
    
    Format for --bonds:
        coupon,maturity,face_value,yield
    
    Example:
        quantterm fixed-income portfolio --bonds "0.05,2030-01-01,1000,0.04;0.04,2028-01-01,2000,0.035"
    """
    from quantterm.fixed_income.bonds import Bond, BondAnalytics
    from quantterm.fixed_income.portfolio import FixedIncomePortfolio
    
    # Parse settlement date
    if settlement:
        settlement_date = date.fromisoformat(settlement)
    else:
        settlement_date = date.today()
    
    # Parse bond specifications
    try:
        bond_specs = []
        for spec in bonds.split(";"):
            parts = spec.split(",")
            if len(parts) != 4:
                continue
            coupon = float(parts[0])
            maturity = date.fromisoformat(parts[1].strip())
            face = float(parts[2])
            yld = float(parts[3])
            bond_specs.append((coupon, maturity, face, yld))
    except Exception as e:
        console.print(f"[red]Error parsing bonds: {e}[/red]")
        raise typer.Exit(1)
    
    if not bond_specs:
        console.print("[red]No valid bonds specified[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]Bond Portfolio Analysis[/bold]")
    console.print(f"Number of bonds: {len(bond_specs)}")
    console.print("-" * 40)
    
    # Create portfolio
    portfolio = FixedIncomePortfolio(name="CLI Portfolio")
    
    for i, (coupon, maturity, face, yld) in enumerate(bond_specs):
        bond = Bond(
            cusip=f"BOND_{i+1}",
            coupon=coupon,
            maturity=maturity,
            face_value=face,
            frequency=2
        )
        
        # Calculate metrics for this bond
        price = BondAnalytics.price(bond, yld, settlement_date)
        mod_dur = BondAnalytics.modified_duration(bond, yld, settlement_date)
        convex = BondAnalytics.convexity(bond, yld, settlement_date)
        
        portfolio.add_bond(bond, yld, settlement_date)
        
        # Show individual bond details
        console.print(f"\nBond {i+1}: {coupon*100:.2f}% {maturity}")
        console.print(f"  Face: ${face:,.2f} | Price: ${price:,.2f} | YTM: {yld*100:.2f}%")
        console.print(f"  Duration: {mod_dur:.3f} | Convexity: {convex:.3f}")
    
    # Portfolio-level metrics
    port_metrics = portfolio.get_metrics()
    
    console.print(f"\n[bold]Portfolio Summary:[/bold]")
    console.print(f"Total Face Value: ${port_metrics['total_face']:,.2f}")
    console.print(f"Total Market Value: ${port_metrics['total_market_value']:,.2f}")
    console.print(f"Weighted Duration: {port_metrics['weighted_duration']:.4f}")
    console.print(f"Weighted Convexity: {port_metrics['weighted_convexity']:.4f}")
    console.print(f"Portfolio Yield: {port_metrics['portfolio_yield']*100:.4f}%")


if __name__ == "__main__":
    app()

"""CLI commands for optimization - Bayesian and Walk-Forward."""
import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(help="Strategy optimization commands")
console = Console()

# Strategy class registry
STRATEGY_REGISTRY = {
    "BuyAndHold": "quantterm.backtesting.strategy.base.BuyAndHoldStrategy",
    "Rebalancing": "quantterm.backtesting.strategy.rebalancing.RebalancingStrategy",
}


def _load_strategy_class(name: str):
    """Dynamically load strategy class."""
    if name not in STRATEGY_REGISTRY:
        available = ", ".join(STRATEGY_REGISTRY.keys())
        raise ImportError(f"Unknown strategy '{name}'. Available: {available}")
    
    module_path, class_name = STRATEGY_REGISTRY[name].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


@app.command("bayesian")
def optimize_bayesian(
    strategy: str = typer.Argument(..., help="Strategy name (BuyAndHold, Rebalancing)"),
    symbols: list[str] = typer.Option(..., "--symbols", "-s", help="Trading symbols"),
    start: str = typer.Option("2020-01-01", "--start", help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", "--end", help="Backtest end date (YYYY-MM-DD)"),
    objective: str = typer.Option("sharpe_ratio", "--objective", "-o", 
                                  help="Objective: sharpe_ratio, total_return, calmar_ratio"),
    n_iterations: int = typer.Option(30, "--iterations", "-n", help="Number of optimization iterations"),
    initial_points: int = typer.Option(10, "--initial", "-i", help="Initial random sampling points"),
    capital: float = typer.Option(1000000.0, "--capital", "-c", help="Initial capital"),
    lookback: str = typer.Option("10,20,30", "--lookback", help="Lookback periods (comma-separated)"),
    threshold: str = typer.Option("0.01,0.05,0.10", "--threshold", help="Signal thresholds (comma-separated)"),
):
    """
    Optimize strategy parameters using Bayesian optimization.
    
    Finds optimal parameters by building a surrogate model of the
    parameter to performance landscape, then intelligently exploring
    high-potential regions.
    
    Example:
        quantterm optimize bayesian Rebalancing --symbols SPY QQQ --start 2020-01-01 --end 2023-12-31
    """
    from quantterm.optimization.bayesian import BayesianOptimizer
    from quantterm.backtesting.engine import BacktestEngine, MultiSymbolEngine
    from quantterm.backtesting.data_handler import DataHandler, MultiSymbolDataHandler
    from quantterm.backtesting.execution import Execution
    
    # Parse parameter space
    try:
        lookback_values = [int(x.strip()) for x in lookback.split(",")]
        threshold_values = [float(x.strip()) for x in threshold.split(",")]
    except ValueError as e:
        console.print(f"[red]Error parsing parameters: {e}[/red]")
        raise typer.Exit(1)
    
    # Build parameter space based on strategy type
    if strategy == "Rebalancing":
        param_space = {
            "lookback": lookback_values,
            "threshold": threshold_values,
        }
    else:
        # Default parameter space for other strategies
        param_space = {
            "lookback": lookback_values,
        }
    
    # Setup
    console.print(f"[bold green]Bayesian Optimization: {strategy}[/bold green]")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Period: {start} to {end}")
    console.print(f"Objective: {objective}")
    console.print(f"Parameter space: {param_space}")
    console.print("-" * 50)
    
    # Load strategy class
    try:
        strategy_class = _load_strategy_class(strategy)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Setup backtest engine
    data_handler = MultiSymbolDataHandler() if len(symbols) > 1 else DataHandler()
    execution = Execution()
    
    if len(symbols) > 1:
        engine = MultiSymbolEngine(
            strategy_class=strategy_class,
            symbols=symbols,
            initial_capital=capital,
            data_handler=data_handler,
            execution=execution
        )
        run_method = "multi"
    else:
        engine = BacktestEngine(
            strategy_class=strategy_class,
            initial_capital=capital,
            data_handler=data_handler,
            execution=execution
        )
        run_method = "single"
    
    # Create optimizer
    optimizer = BayesianOptimizer(
        param_space=param_space,
        strategy_class=strategy_class,
        backtest_engine=engine,
        objective=objective,
        n_initial=initial_points,
        n_iterations=n_iterations,
        random_state=42
    )
    
    # Run optimization with progress
    console.print("\n[bold]Phase 1: Initial random sampling...[/bold]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Running optimization...", total=n_iterations + initial_points)
        
        # Run optimization
        try:
            if run_method == "multi":
                result = optimizer.optimize(
                    start=start,
                    end=end,
                    target_weights={s: 1.0/len(symbols) for s in symbols}
                )
            else:
                result = optimizer.optimize(
                    symbol=symbols[0],
                    start=start,
                    end=end
                )
        except Exception as e:
            console.print(f"[red]Optimization error: {e}[/red]")
            raise typer.Exit(1)
        
        progress.update(task, completed=n_iterations + initial_points)
    
    # Display results
    console.print("\n[bold green]Optimization Complete![/bold green]\n")
    
    # Best parameters table
    params_table = Table(title=f"Best Parameters (Objective: {objective})")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Optimal Value", style="green")
    params_table.add_column("Search Range", style="dim")
    
    for param, value in result.best_params.items():
        range_str = str(param_space.get(param, "N/A"))
        params_table.add_row(
            param,
            str(value),
            range_str
        )
    
    console.print(params_table)
    
    # Performance summary
    perf_table = Table(title="Optimization Results")
    perf_table.add_column("Metric", style="cyan")
    perf_table.add_column("Value", style="green")
    
    perf_table.add_row("Best Objective", f"{result.best_objective:.4f}")
    perf_table.add_row("Total Evaluations", str(result.n_evaluations))
    perf_table.add_row("Initial Random Points", str(initial_points))
    perf_table.add_row("Bayesian Iterations", str(n_iterations))
    
    console.print(perf_table)
    
    # Parameter importance
    if result.parameter_importance:
        console.print("\n[bold]Parameter Importance:[/bold]")
        for param, importance in sorted(result.parameter_importance.items(), key=lambda x: -x[1]):
            bar = "█" * int(importance * 20)
            console.print(f"  {param:15s}: {bar} {importance:.1%}")


@app.command("walk-forward")
def run_walk_forward(
    strategy: str = typer.Argument(..., help="Strategy name"),
    symbols: list[str] = typer.Option(..., "--symbols", "-s", help="Trading symbols"),
    start: str = typer.Option("2015-01-01", "--start", help="Analysis start date"),
    end: str = typer.Option("2023-12-31", "--end", help="Analysis end date"),
    train_months: int = typer.Option(12, "--train-months", help="Training window in months"),
    test_months: int = typer.Option(3, "--test-months", help="Testing window in months"),
    objective: str = typer.Option("sharpe_ratio", "--objective", "-o"),
    capital: float = typer.Option(1000000.0, "--capital", "-c"),
):
    """
    Run walk-forward analysis to validate strategy robustness.
    
    Walks forward through time, training on historical data and
    testing on forward periods to detect overfitting.
    
    Example:
        quantterm optimize walk-forward Rebalancing --symbols SPY QQQ --train-months 12 --test-months 3
    """
    from quantterm.optimization.walk_forward import WalkForwardAnalyzer
    from quantterm.backtesting.engine import MultiSymbolEngine
    from quantterm.backtesting.data_handler import MultiSymbolDataHandler
    from quantterm.backtesting.execution import Execution
    
    console.print(f"[bold green]Walk-Forward Analysis: {strategy}[/bold green]")
    console.print(f"Symbols: {', '.join(symbols)}")
    console.print(f"Period: {start} to {end}")
    console.print(f"Train window: {train_months} months | Test window: {test_months} months")
    console.print("-" * 50)
    
    # Load strategy class
    try:
        strategy_class = _load_strategy_class(strategy)
    except ImportError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)
    
    # Setup
    data_handler = MultiSymbolDataHandler()
    execution = Execution()
    engine = MultiSymbolEngine(
        strategy_class=strategy_class,
        symbols=symbols,
        initial_capital=capital,
        data_handler=data_handler,
        execution=execution
    )
    
    # Simple parameter space (for demo)
    param_space = {
        "lookback": [10, 20, 30],
    }
    
    # Run walk-forward
    analyzer = WalkForwardAnalyzer(
        strategy_class=strategy_class,
        param_space=param_space,
        backtest_engine=engine,
        train_months=train_months,
        test_months=test_months,
        objective=objective
    )
    
    console.print("\n[bold]Running walk-forward analysis...[/bold]")
    
    try:
        result = analyzer.analyze(
            start_date=start,
            end_date=end,
            symbols=symbols
        )
    except Exception as e:
        console.print(f"[red]Walk-forward error: {e}[/red]")
        raise typer.Exit(1)
    
    # Display results
    summary = result.summary()
    
    console.print("\n[bold green]Walk-Forward Analysis Complete![/bold green]\n")
    
    wf_table = Table(title="Walk-Forward Results")
    wf_table.add_column("Metric", style="cyan")
    wf_table.add_column("Value", style="green")
    
    wf_table.add_row("Number of Windows", str(summary['n_windows']))
    wf_table.add_row("Mean Test Sharpe", f"{summary['mean_test_sharpe']:.3f}")
    wf_table.add_row("Std Test Sharpe", f"{summary['std_test_sharpe']:.3f}")
    wf_table.add_row("Mean Degradation", f"{summary['mean_degradation']:.1%}")
    wf_table.add_row("Stability Score", f"{summary['stability']:.1%}")
    
    console.print(wf_table)
    
    # Show per-window results
    if result.test_results:
        console.print("\n[bold]Per-Window Performance:[/bold]")
        window_table = Table()
        window_table.add_column("Window", style="cyan")
        window_table.add_column("Train Sharpe", style="dim")
        window_table.add_column("Test Sharpe", style="green")
        window_table.add_column("Degradation", style="yellow")
        
        for i, (train_r, test_r, deg) in enumerate(zip(result.train_results, result.test_results, result.degradation)):
            train_sharpe = train_r.get('sharpe_ratio', 0)
            test_sharpe = test_r.get('sharpe_ratio', 0)
            window_table.add_row(
                f"Window {i+1}",
                f"{train_sharpe:.3f}",
                f"{test_sharpe:.3f}",
                f"{deg:.1%}"
            )
        
        console.print(window_table)


if __name__ == "__main__":
    app()

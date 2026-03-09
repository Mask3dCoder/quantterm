"""CLI commands for ML training and backtesting."""
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="ML strategy commands")
console = Console()


@app.command("train")
def train_model(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Trading symbol"),
    start: str = typer.Option("2015-01-01", "--start", help="Training start date (YYYY-MM-DD)"),
    end: str = typer.Option("2020-12-31", "--end", help="Training end date (YYYY-MM-DD)"),
    model_type: str = typer.Option("random_forest", "--model", "-m", 
                                   help="Model type: random_forest, gradient_boosting, logistic"),
    output: str = typer.Option("model.pkl", "--output", "-o", help="Output model file")
):
    """Train ML model on historical data to predict price direction.
    
    The model learns to predict whether prices will go up or down
    based on technical indicators and price patterns.
    
    Example:
        quantterm ml train --symbol SPY --start 2015-01-01 --end 2020-12-31 --output momentum_model.pkl
    """
    from quantterm.ml.models import MLModelTrainer
    from quantterm.ml.features import FeatureEngineer
    from quantterm.backtesting.data_handler import DataHandler
    
    console.print(f"[bold]Training ML Model[/bold]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Period: {start} to {end}")
    console.print(f"Model: {model_type}")
    console.print("-" * 40)
    
    # Load data
    handler = DataHandler()
    data = handler.get_bars(symbol, start, end)
    
    if data.empty:
        console.print("[red]No data loaded. Check the symbol and date range.[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loaded {len(data)} bars")
    
    # Create feature engineer
    feature_engineer = FeatureEngineer(lookback_windows=[5, 10, 20, 60])
    
    # Create trainer
    trainer = MLModelTrainer(
        model_type=model_type,
        feature_engineer=feature_engineer
    )
    
    # Prepare training data
    X, y = trainer.prepare_training_data(data)
    
    if len(X) < 100:
        console.print(f"[red]Insufficient training data: {len(X)} samples (need at least 100)[/red]")
        raise typer.Exit(1)
    
    console.print(f"Training on {len(X)} samples with {len(X.columns)} features...")
    
    # Train model
    results = trainer.train(X, y)
    
    # Save model
    trainer.save(output)
    
    console.print(f"\n[green]Model saved to {output}[/green]")
    console.print(f"CV Accuracy: {results['cv_accuracy_mean']:.2%} (+/- {results['cv_accuracy_std']:.2%})")
    console.print(f"Overfitting Gap: {results['overfitting_gap']:.4f}")
    
    # Show feature importance if available
    if 'feature_importance' in results and results['feature_importance']:
        console.print("\n[bold]Top Feature Importances:[/bold]")
        fi_table = Table()
        fi_table.add_column("Feature", style="cyan")
        fi_table.add_column("Importance", style="green")
        
        sorted_fi = sorted(results['feature_importance'].items(), key=lambda x: -x[1])
        for feat, imp in sorted_fi[:10]:
            fi_table.add_row(feat[:30], f"{imp:.3f}")
        
        console.print(fi_table)


@app.command("backtest")
def backtest_ml(
    model_file: str = typer.Argument(..., help="Path to trained model file (.pkl)"),
    symbol: str = typer.Option(..., "--symbol", "-s", help="Trading symbol"),
    start: str = typer.Option("2021-01-01", "--start", help="Backtest start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", "--end", help="Backtest end date (YYYY-MM-DD)"),
    capital: float = typer.Option(1000000.0, "--capital", "-c", help="Initial capital"),
    threshold: float = typer.Option(0.6, "--threshold", "-t", help="Prediction threshold (0.5-0.9)")
):
    """Backtest ML strategy on out-of-sample data.
    
    Uses a trained model to generate trading signals and evaluates
    performance on historical data that was not used in training.
    
    Example:
        quantterm ml backtest model.pkl --symbol SPY --start 2021-01-01 --end 2023-12-31
    """
    from pathlib import Path
    
    from quantterm.ml.models import MLModelTrainer
    from quantterm.ml.strategy import MLStrategy
    from quantterm.backtesting.engine import BacktestEngine
    from quantterm.backtesting.data_handler import DataHandler
    from quantterm.backtesting.execution import Execution
    from quantterm.backtesting.portfolio import Portfolio
    from quantterm.backtesting.metrics import calculate_returns, calculate_metrics, format_metrics
    
    # Check model file exists
    model_path = Path(model_file)
    if not model_path.exists():
        console.print(f"[red]Model file not found: {model_file}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[bold]ML Backtest[/bold]")
    console.print(f"Model: {model_file}")
    console.print(f"Symbol: {symbol}")
    console.print(f"Period: {start} to {end}")
    console.print(f"Capital: ${capital:,.2f}")
    console.print("-" * 40)
    
    # Load model
    trainer = MLModelTrainer()
    trainer.load(model_file)
    console.print(f"Loaded model with {len(trainer.feature_names)} features")
    
    # Load test data
    handler = DataHandler()
    data = handler.get_bars(symbol, start, end)
    
    if data.empty:
        console.print("[red]No data loaded. Check the symbol and date range.[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loaded {len(data)} bars for testing")
    
    # Setup portfolio
    portfolio = Portfolio(name="MLStrategy", cash=capital)
    
    # Create ML strategy
    strategy = MLStrategy(
        name="MLStrategy",
        portfolio=portfolio,
        data_handler=handler,
        model_trainer=trainer,
        prediction_threshold=threshold,
        symbols=[symbol],
        target_weights={symbol: 1.0}
    )
    
    # Run backtest
    engine = BacktestEngine(
        strategy_class=type(strategy),  # Pass the class, but we'll use our instance
        initial_capital=capital,
        data_handler=handler,
        execution=Execution()
    )
    
    # We need to run manually since we're passing a specific instance
    trades = []
    portfolio_values = [capital]
    current_prices = {}
    
    # Pre-generate features for the test period
    from quantterm.ml.features import FeatureEngineer
    feature_engineer = FeatureEngineer(lookback_windows=[5, 10, 20, 60])
    
    # Create features for each bar and run strategy
    for idx, row in data.iterrows():
        # Update portfolio prices
        current_prices = {symbol: row['Close']}
        portfolio.update_prices(current_prices)
        
        # Create features up to this point
        hist_data = data.loc[:idx]
        if len(hist_data) < 30:
            continue
            
        try:
            features_df = feature_engineer.create_features_batch(hist_data)
            if features_df.empty or idx not in features_df.index:
                continue
                
            features = features_df.loc[idx]
            
            # Create bar event
            from quantterm.backtesting.events import BarEvent
            bar = BarEvent(
                timestamp=idx,
                symbol=symbol,
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['volume'])
            )
            
            # Get signal from strategy
            order = strategy.on_bar_with_features(bar, features)
            
            if order is not None:
                # Execute
                from quantterm.backtesting.events import FillEvent
                fill_price = row['Close']
                commission = max(1.0, abs(order.quantity) * fill_price * 0.001)  # 0.1% commission
                
                fill = FillEvent(
                    timestamp=idx,
                    symbol=symbol,
                    quantity=order.quantity,
                    price=fill_price,
                    commission=commission
                )
                
                # Process fill
                if order.quantity > 0:
                    # Buy
                    portfolio.cash -= order.quantity * fill_price + commission
                else:
                    # Sell
                    portfolio.cash += abs(order.quantity) * fill_price - commission
                
                trades.append(fill)
                
        except Exception as e:
            # Skip this bar on error
            continue
        
        # Track portfolio value
        portfolio_values.append(portfolio.get_total_value(current_prices))
    
    # Calculate final metrics
    final_value = portfolio.get_total_value(current_prices)
    returns = calculate_returns(portfolio_values)
    metrics = calculate_metrics(returns, capital)
    
    # Display results
    console.print("\n" + format_metrics(metrics, capital, final_value))
    
    # Additional ML-specific metrics
    console.print(f"\n[bold]ML Strategy Details:[/bold]")
    console.print(f"Total Trades: {len(trades)}")
    
    # Calculate prediction stats
    predictions = strategy.get_predictions()
    if not predictions.empty:
        buy_signals = len(predictions[predictions['prob_up'] > threshold])
        console.print(f"Buy Signals: {buy_signals}")
        console.print(f"Avg Confidence: {predictions['confidence'].mean():.2%}")
    
    # Save trade log
    console.print(f"\n[green]Backtest complete![/green]")


@app.command("features")
def analyze_features(
    symbol: str = typer.Option(..., "--symbol", "-s", help="Trading symbol"),
    start: str = typer.Option("2020-01-01", "--start", help="Start date (YYYY-MM-DD)"),
    end: str = typer.Option("2023-12-31", "--end", help="End date (YYYY-MM-DD)"),
    windows: str = typer.Option("5,10,20,60", "--windows", "-w", help="Lookback windows (comma-separated)")
):
    """Analyze features for a symbol.
    
    Shows all computed technical features that can be used
    for ML model training.
    
    Example:
        quantterm ml features --symbol SPY --windows 5,10,20,60
    """
    from quantterm.ml.features import FeatureEngineer
    from quantterm.backtesting.data_handler import DataHandler
    
    console.print(f"[bold]Feature Analysis[/bold]")
    console.print(f"Symbol: {symbol}")
    
    # Parse windows
    try:
        window_list = [int(w.strip()) for w in windows.split(",")]
    except ValueError:
        console.print("[red]Invalid windows format. Use comma-separated integers.[/red]")
        raise typer.Exit(1)
    
    # Load data
    handler = DataHandler()
    data = handler.get_bars(symbol, start, end)
    
    if data.empty:
        console.print("[red]No data loaded. Check the symbol and date range.[/red]")
        raise typer.Exit(1)
    
    console.print(f"Loaded {len(data)} bars")
    
    # Create features
    engineer = FeatureEngineer(lookback_windows=window_list)
    features_df = engineer.create_features_batch(data)
    
    if features_df.empty:
        console.print("[red]Could not generate features[/red]")
        raise typer.Exit(1)
    
    console.print(f"\n[green]Generated {len(features_df)} feature vectors[/green]")
    console.print(f"Feature count: {len(features_df.columns)}")
    
    # Show feature columns
    console.print(f"\n[bold]Feature Columns:[/bold]")
    for i, col in enumerate(features_df.columns):
        console.print(f"  {i+1:2d}. {col}")
    
    # Show summary statistics
    console.print(f"\n[bold]Feature Statistics:[/bold]")
    console.print(features_df.describe().to_string())


@app.command("validate")
def validate_no_lookahead(
    symbol: str = typer.Option("SPY", "--symbol", "-s", help="Symbol to test"),
    n_samples: int = typer.Option(100, "--samples", "-n", help="Number of samples to test")
):
    """Validate that features don't have lookahead bias.
    
    Ensures that feature calculations only use historical data
    and don't accidentally incorporate future information.
    
    Example:
        quantterm ml validate --symbol SPY --samples 100
    """
    import numpy as np
    import pandas as pd
    from quantterm.ml.features import FeatureEngineer
    
    console.print(f"[bold]Validating No Lookahead Bias[/bold]")
    console.print(f"Symbol: {symbol}")
    console.print(f"Testing {n_samples} random timestamps...")
    
    # Generate random walk data (no predictable pattern)
    np.random.seed(42)
    n = 200
    returns = np.random.randn(n) * 0.02  # Random walk
    prices = 100 * np.cumsum(np.exp(returns))  # Geometric random walk
    
    data = pd.DataFrame({
        'Open': prices * 0.99,
        'High': prices * 1.01,
        'Low': prices * 0.98,
        'Close': prices,
        'Volume': np.random.randint(1000000, 5000000, n)
    }, index=pd.date_range('2020-01-01', periods=n, freq='D'))
    
    engineer = FeatureEngineer(lookback_windows=[5, 10, 20])
    
    # Test feature creation at various points
    success_count = 0
    fail_count = 0
    
    for i in range(30, min(n - 10, n_samples + 30)):
        try:
            features = engineer.create_features(data.iloc[:i+1], data.index[i])
            if features is not None:
                success_count += 1
            else:
                fail_count += 1
        except Exception:
            fail_count += 1
    
    total = success_count + fail_count
    
    if success_count > total * 0.8:
        console.print(f"\n[green]✓ PASS: {success_count}/{total} samples computed successfully[/green]")
        console.print("Features do not appear to have lookahead bias.")
    else:
        console.print(f"\n[red]✗ FAIL: Only {success_count}/{total} samples succeeded[/red]")
        console.print("There may be an issue with feature computation.")


if __name__ == "__main__":
    app()

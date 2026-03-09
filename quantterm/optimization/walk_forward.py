"""
Walk-Forward Analysis for Strategy Validation.

Implements rolling window walk-forward analysis to validate
strategy robustness and detect overfitting.
"""
from dataclasses import dataclass, field
from typing import Any, Optional
import numpy as np
from datetime import datetime, timedelta


@dataclass
class WalkForwardResult:
    """Results from walk-forward analysis."""
    train_results: list[dict]
    test_results: list[dict]
    rolling_sharpe: list[float]
    rolling_returns: list[float]
    degradation: list[float]  # Train vs test performance degradation
    best_params_per_window: list[dict]
    overall_stability: float
    
    def summary(self) -> dict:
        """Generate summary statistics."""
        return {
            'mean_test_sharpe': np.mean([r.get('sharpe_ratio', 0) for r in self.test_results]),
            'std_test_sharpe': np.std([r.get('sharpe_ratio', 0) for r in self.test_results]),
            'mean_degradation': np.mean(self.degradation),
            'stability': self.overall_stability,
            'n_windows': len(self.test_results)
        }


class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.
    
    Features:
    - Rolling train/test window splitting
    - Parameter re-optimization on each training window
    - Performance degradation tracking
    - Robustness scoring
    """
    
    def __init__(
        self,
        strategy_class: type,
        param_space: dict,
        backtest_engine: Any,
        train_months: int = 12,
        test_months: int = 3,
        n_steps: Optional[int] = None,  # If None, roll until end of data
        objective: str = 'sharpe_ratio',
        min_train_months: int = 6,
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            strategy_class: Strategy class to optimize
            param_space: Parameter space for optimization
            backtest_engine: Backtest engine instance
            train_months: Length of training window in months
            test_months: Length of test window in months
            n_steps: Number of walk-forward steps (None = auto)
            objective: Objective metric to optimize
            min_train_months: Minimum training window size
        """
        self.strategy_class = strategy_class
        self.param_space = param_space
        self.backtest_engine = backtest_engine
        self.train_months = train_months
        self.test_months = test_months
        self.n_steps = n_steps
        self.objective = objective
        self.min_train_months = min_train_months
    
    def analyze(
        self,
        start_date: str,
        end_date: str,
        symbols: list[str],
        **optimization_kwargs
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.
        
        Args:
            start_date: Start date for analysis
            end_date: End date for analysis
            symbols: Trading symbols
            **optimization_kwargs: Args for Bayesian optimizer
            
        Returns:
            WalkForwardResult with analysis results
        """
        from quantterm.optimization.bayesian import BayesianOptimizer
        
        # Parse dates
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        
        # Calculate windows
        windows = self._generate_windows(start, end)
        
        train_results = []
        test_results = []
        rolling_sharpe = []
        rolling_returns = []
        degradation = []
        best_params = []
        
        print(f"Walk-Forward Analysis: {len(windows)} windows")
        
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\nWindow {i+1}/{len(windows)}:")
            print(f"  Train: {train_start.date()} to {train_end.date()}")
            print(f"  Test:  {test_start.date()} to {test_end.date()}")
            
            # Optimize on training window
            optimizer = BayesianOptimizer(
                param_space=self.param_space,
                strategy_class=self.strategy_class,
                backtest_engine=self.backtest_engine,
                objective=self.objective,
                **optimization_kwargs
            )
            
            train_result = optimizer.optimize(
                symbols=symbols,
                start=train_start.isoformat(),
                end=train_end.isoformat(),
            )
            
            # Evaluate best params on test window
            test_optimizer = BayesianOptimizer(
                param_space=self.param_space,
                strategy_class=self.strategy_class,
                backtest_engine=self.backtest_engine,
                objective=self.objective,
                n_initial=1,
                n_iterations=0,  # No optimization, just evaluate
            )
            
            test_obj = test_optimizer._evaluate_with_dates(
                train_result.best_params,
                symbols=symbols,
                start=test_start.isoformat(),
                end=test_end.isoformat(),
            )
            
            # Store results
            train_results.append({
                'window': i,
                'train_start': train_start.isoformat(),
                'train_end': train_end.isoformat(),
                'best_params': train_result.best_params,
                'best_objective': train_result.best_objective,
            })
            
            test_results.append({
                'window': i,
                'test_start': test_start.isoformat(),
                'test_end': test_end.isoformat(),
                'params': train_result.best_params,
                'objective': test_obj,
            })
            
            # Track metrics
            rolling_sharpe.append(test_obj)
            degradation.append(train_result.best_objective - test_obj)
            best_params.append(train_result.best_params)
            
            print(f"  Train: {train_result.best_objective:.4f} → Test: {test_obj:.4f}")
        
        # Calculate overall stability
        stability = self._calculate_stability(rolling_sharpe, degradation)
        
        return WalkForwardResult(
            train_results=train_results,
            test_results=test_results,
            rolling_sharpe=rolling_sharpe,
            rolling_returns=rolling_returns,
            degradation=degradation,
            best_params_per_window=best_params,
            overall_stability=stability
        )
    
    def _generate_windows(self, start: datetime, end: datetime) -> list:
        """Generate rolling train/test windows."""
        windows = []
        
        current_train_start = start
        
        while True:
            # Training window
            train_end = current_train_start + timedelta(days=30 * self.train_months)
            
            # Test window
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=30 * self.test_months)
            
            # Check if we exceed end date
            if test_start > end:
                break
            
            # Ensure minimum training size
            if (train_end - current_train_start).days < 30 * self.min_train_months:
                current_train_start += timedelta(days=30)
                continue
            
            windows.append((
                current_train_start,
                train_end,
                test_start,
                min(test_end, end)
            ))
            
            # Check if we've reached max steps
            if self.n_steps and len(windows) >= self.n_steps:
                break
            
            # Move to next window (rolling by test period)
            current_train_start = test_start
        
        return windows
    
    def _calculate_stability(
        self,
        rolling_sharpe: list[float],
        degradation: list[float]
    ) -> float:
        """Calculate strategy stability score (0-1)."""
        if not rolling_sharpe:
            return 0.0
        
        # Factors:
        # 1. Consistency of test Sharpe ratios (higher = better)
        # 2. Low degradation (lower = better)
        # 3. Positive returns ratio
        
        # Sharpe consistency (normalized)
        sharpe_std = np.std(rolling_sharpe) if len(rolling_sharpe) > 1 else 1.0
        sharpe_mean = np.mean(rolling_sharpe)
        
        sharpe_score = 1.0 / (1.0 + sharpe_std)  # Lower variance = higher score
        
        # Degradation score (lower degradation is better)
        avg_degradation = np.mean(degradation) if degradation else 0.0
        degradation_score = 1.0 / (1.0 + max(0, avg_degradation))
        
        # Positive returns ratio
        positive_ratio = sum(1 for s in rolling_sharpe if s > 0) / len(rolling_sharpe)
        
        # Combined score
        stability = (
            0.4 * sharpe_score +
            0.4 * degradation_score +
            0.2 * positive_ratio
        )
        
        return stability

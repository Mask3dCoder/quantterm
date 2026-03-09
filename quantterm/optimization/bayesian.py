"""
Bayesian Optimization for Strategy Parameter Tuning.

Uses Gaussian Processes to efficiently find optimal strategy parameters
without exhaustive grid search.
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
import math
import numpy as np


@dataclass
class OptimizationResult:
    """Results from Bayesian optimization."""
    best_params: dict[str, Any]
    best_objective: float
    n_evaluations: int
    convergence_history: list[dict] = field(default_factory=list)
    parameter_importance: dict[str, float] = field(default_factory=dict)


class BayesianOptimizer:
    """
    Bayesian optimization for strategy parameter tuning.
    
    Features:
    - Gaussian Process surrogate model
    - Expected Improvement (EI) acquisition function
    - Support for continuous, discrete, and categorical parameters
    - Constraint handling
    - Parallel evaluation support
    """
    
    def __init__(
        self,
        param_space: dict[str, tuple | list],
        objective_func: Optional[Callable] = None,
        strategy_class: Optional[type] = None,
        backtest_engine: Optional[Any] = None,
        objective: str = 'sharpe_ratio',
        acquisition: str = 'ei',
        noise_level: float = 0.1,
        n_initial: int = 10,
        n_iterations: int = 50,
        n_parallel: int = 1,
        constraints: Optional[list[Callable]] = None,
        random_state: int = 42,
    ):
        """
        Initialize Bayesian optimizer.
        
        Args:
            param_space: Dict mapping param name to (min, max) for continuous,
                        or list of values for discrete/categorical
            objective_func: Function that takes params and returns objective value
            strategy_class: Strategy class for backtesting
            backtest_engine: Engine to run backtests
            objective: Metric to optimize ('sharpe_ratio', 'total_return', etc.)
            acquisition: Acquisition function ('ei', 'ucb', 'poi')
            noise_level: Expected noise in objective (0.0-1.0)
            n_initial: Number of random initial samples
            n_iterations: Number of optimization iterations
            n_parallel: Number of parallel evaluations
            constraints: List of constraint functions (must return True if valid)
            random_state: Random seed for reproducibility
        """
        self.param_space = param_space
        self.objective_func = objective_func
        self.strategy_class = strategy_class
        self.backtest_engine = backtest_engine
        self.objective_name = objective
        self.acquisition = acquisition
        self.noise_level = noise_level
        self.n_initial = n_initial
        self.n_iterations = n_iterations
        self.n_parallel = n_parallel
        self.constraints = constraints or []
        self.random_state = random_state
        
        # Initialize random state
        np.random.seed(random_state)
        
        # State
        self.X: list[list[float]] = []  # Normalized params (0-1)
        self.y: list[float] = []  # Objective values
        self.param_names = list(param_space.keys())
        self.param_bounds = self._build_param_bounds()
        
        # Results
        self.best_params: Optional[dict] = None
        self.best_objective: float = -float('inf')
        self.history: list[dict] = []
        self.backtest_kwargs: dict = {}
    
    def _build_param_bounds(self) -> list[tuple]:
        """Convert param_space to normalized bounds."""
        bounds = []
        for name in self.param_names:
            values = self.param_space[name]
            if isinstance(values, (list, tuple)) and len(values) == 2:
                # Continuous: (min, max)
                bounds.append((0.0, 1.0))
            else:
                # Discrete/categorical: number of options
                bounds.append((0.0, 1.0))
        return bounds
    
    def _normalize_params(self, params: dict) -> list[float]:
        """Convert params to normalized 0-1 range."""
        normalized = []
        for name in self.param_names:
            value = params[name]
            bounds = self.param_space[name]
            
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                # Continuous: scale to [0, 1]
                min_val, max_val = bounds
                if max_val > min_val:
                    normalized.append((value - min_val) / (max_val - min_val))
                else:
                    normalized.append(0.5)
            else:
                # Discrete: find index
                try:
                    idx = bounds.index(value) if isinstance(bounds, list) else value
                    normalized.append(idx / len(bounds))
                except (ValueError, IndexError):
                    normalized.append(0.5)
        
        return normalized
    
    def _denormalize_params(self, normalized: list[float]) -> dict:
        """Convert normalized params back to original scale."""
        params = {}
        for i, name in enumerate(self.param_names):
            bounds = self.param_space[name]
            val = normalized[i]
            
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                # Continuous: scale from [0, 1]
                min_val, max_val = bounds
                params[name] = min_val + val * (max_val - min_val)
                
                # Round to int if bounds are integers
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = int(round(params[name]))
            else:
                # Discrete: find value at index
                if isinstance(bounds, list):
                    idx = int(round(val * (len(bounds) - 1)))
                    idx = max(0, min(idx, len(bounds) - 1))
                    params[name] = bounds[idx]
                else:
                    params[name] = val
        
        return params
    
    def _check_constraints(self, params: dict) -> bool:
        """Check if params satisfy all constraints."""
        for constraint in self.constraints:
            if not constraint(params):
                return False
        return True
    
    def _acquisition_expected_improvement(
        self, 
        x: np.ndarray, 
        y_best: float,
        noise: float
    ) -> np.ndarray:
        """
        Expected Improvement acquisition function.
        
        EI = E[max(0, f(x) - f_best - xi)]
        
        where xi is the exploration-exploitation tradeoff parameter.
        """
        mu, sigma = self._gp_predict(x)
        
        # Ensure sigma is not too small
        sigma = np.maximum(sigma, 1e-10)
        
        # Calculate z (standardized improvement)
        z = (mu - y_best - 0.01) / sigma
        
        # EI formula
        ei = (mu - y_best) * self._norm_cdf(z) + sigma * self._norm_pdf(z)
        
        return ei
    
    def _acquisition_ucb(
        self, 
        x: np.ndarray, 
        y_best: float,
        kappa: float = 2.0
    ) -> np.ndarray:
        """
        Upper Confidence Bound acquisition function.
        
        UCB = mu + kappa * sigma
        """
        mu, sigma = self._gp_predict(x)
        return mu + kappa * sigma
    
    @staticmethod
    def _norm_cdf(x):
        """Standard normal CDF."""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    @staticmethod
    def _norm_pdf(x):
        """Standard normal PDF."""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def _gp_predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Predict mean and variance using Gaussian Process.
        
        Uses constant mean and RBF kernel (simplified GP).
        """
        if len(self.X) == 0:
            # No data yet, return prior
            return np.zeros(len(x)), np.ones(len(x))
        
        X_obs = np.array(self.X)
        y_obs = np.array(self.y)
        
        # Compute kernel matrix
        K = self._rbf_kernel(X_obs, X_obs)
        
        # Add noise for numerical stability
        K += (self.noise_level ** 2) * np.eye(len(X_obs))
        
        try:
            K_inv = np.linalg.inv(K)
        except np.linalg.LinAlgError:
            # Fallback if singular
            K_inv = np.linalg.pinv(K)
        
        # Compute kernel between query points and observations
        K_star = self._rbf_kernel(x, X_obs)
        
        # Predictive mean
        mu = K_star @ K_inv @ y_obs
        
        # Predictive variance
        K_ss = self._rbf_kernel(x, x)
        var = np.diag(K_ss - K_star @ K_inv @ K_star.T)
        var = np.maximum(var, 1e-10)  # Ensure positive
        
        return mu, np.sqrt(var)
    
    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
        """RBF (Radial Basis Function) kernel."""
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        
        # Compute pairwise squared Euclidean distances
        # X1: (n1, d), X2: (n2, d)
        # Result: (n1, n2)
        dist_sq = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
        
        return np.exp(-0.5 * dist_sq / (length_scale ** 2))
    
    def _suggest_next_point(self) -> dict:
        """Suggest next point to evaluate using acquisition function."""
        if len(self.X) < self.n_initial:
            # Random sampling for initial points
            return self._random_sample()
        
        # Optimize acquisition function
        best_acq = -float('inf')
        best_x = None
        
        # Try multiple random starting points
        n_candidates = 1000
        candidates = np.random.uniform(0, 1, (n_candidates, len(self.param_names)))
        
        for x_cand in candidates:
            # Evaluate acquisition
            if self.acquisition == 'ei':
                acq = self._acquisition_expected_improvement(
                    x_cand.reshape(1, -1), 
                    self.best_objective,
                    self.noise_level
                )[0]
            elif self.acquisition == 'ucb':
                acq = self._acquisition_ucb(
                    x_cand.reshape(1, -1),
                    self.best_objective
                )[0]
            else:
                acq = self._acquisition_expected_improvement(
                    x_cand.reshape(1, -1),
                    self.best_objective,
                    self.noise_level
                )[0]
            
            if acq > best_acq:
                best_acq = acq
                best_x = x_cand
        
        return self._denormalize_params(best_x)
    
    def _random_sample(self) -> dict:
        """Generate random parameter sample."""
        params = {}
        for name in self.param_names:
            bounds = self.param_space[name]
            
            if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
                # Continuous
                min_val, max_val = bounds
                if isinstance(min_val, int) and isinstance(max_val, int):
                    params[name] = np.random.randint(min_val, max_val + 1)
                else:
                    params[name] = np.random.uniform(min_val, max_val)
            else:
                # Discrete/categorical
                idx = np.random.randint(0, len(bounds))
                params[name] = bounds[idx]
        
        return params
    
    def _evaluate(self, params: dict) -> float:
        """Evaluate objective function with given params."""
        if not self._check_constraints(params):
            # Return worst score for invalid params
            return -float('inf')
        
        if self.objective_func is not None:
            return self.objective_func(params)
        
        # Use backtest engine
        if self.strategy_class is not None and self.backtest_engine is not None:
            try:
                # Create strategy instance with params
                strategy = self.strategy_class(**params)
                
                # Run backtest
                result = self.backtest_engine.run(
                    strategy=strategy,
                    **self.backtest_kwargs
                )
                
                # Get objective metric
                if self.objective_name == 'sharpe_ratio':
                    return result.get('sharpe_ratio', -float('inf'))
                elif self.objective_name == 'total_return':
                    return result.get('total_return', -float('inf'))
                elif self.objective_name == 'calmar_ratio':
                    return result.get('calmar_ratio', -float('inf'))
                else:
                    return result.get(self.objective_name, -float('inf'))
                    
            except Exception as e:
                # Invalid parameter combination
                print(f"Backtest failed: {e}")
                return -float('inf')
        
        return -float('inf')
    
    def optimize(self, **backtest_kwargs) -> OptimizationResult:
        """
        Run Bayesian optimization.
        
        Args:
            **backtest_kwargs: Arguments passed to backtest_engine.run()
            
        Returns:
            OptimizationResult with best parameters and history
        """
        self.backtest_kwargs = backtest_kwargs
        
        # Phase 1: Random initial sampling
        print(f"Phase 1: Initial random sampling ({self.n_initial} points)...")
        
        for i in range(self.n_initial):
            params = self._random_sample()
            
            # Check constraints
            if not self._check_constraints(params):
                continue
            
            # Evaluate
            obj = self._evaluate(params)
            
            # Store
            self.X.append(self._normalize_params(params))
            self.y.append(obj)
            
            # Update best
            if obj > self.best_objective:
                self.best_objective = obj
                self.best_params = params.copy()
            
            self.history.append({
                'iteration': i,
                'params': params.copy(),
                'objective': obj,
                'is_best': obj == self.best_objective
            })
            
            print(f"  [{i+1}/{self.n_initial}] {params} -> {obj:.4f}")
        
        # Phase 2: Bayesian optimization
        print(f"Phase 2: Bayesian optimization ({self.n_iterations} iterations)...")
        
        for i in range(self.n_iterations):
            # Suggest next point
            params = self._suggest_next_point()
            
            # Evaluate
            obj = self._evaluate(params)
            
            # Store
            self.X.append(self._normalize_params(params))
            self.y.append(obj)
            
            # Update best
            if obj > self.best_objective:
                self.best_objective = obj
                self.best_params = params.copy()
            
            self.history.append({
                'iteration': self.n_initial + i,
                'params': params.copy(),
                'objective': obj,
                'is_best': obj == self.best_objective
            })
            
            print(f"  [{i+1}/{self.n_iterations}] {params} -> {obj:.4f}" + 
                  (" *BEST*" if obj == self.best_objective else ""))
        
        # Calculate parameter importance (simplified: correlation with objective)
        importance = self._calculate_importance()
        
        return OptimizationResult(
            best_params=self.best_params,
            best_objective=self.best_objective,
            n_evaluations=len(self.y),
            convergence_history=self.history,
            parameter_importance=importance
        )
    
    def _calculate_importance(self) -> dict[str, float]:
        """Calculate parameter importance based on variance of best samples."""
        if len(self.X) < 10:
            return {name: 1.0 / len(self.param_names) for name in self.param_names}
        
        # Simple importance: standard deviation of normalized values in top 20% samples
        y_arr = np.array(self.y)
        top_idx = y_arr >= np.percentile(y_arr, 80)
        
        if not any(top_idx):
            return {name: 1.0 / len(self.param_names) for name in self.param_names}
        
        X_top = np.array([self.X[i] for i in range(len(self.X)) if top_idx[i]])
        
        importance = {}
        for i, name in enumerate(self.param_names):
            importance[name] = float(np.std(X_top[:, i]))
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v / total for k, v in importance.items()}
        
        return importance

    def _evaluate_with_dates(
        self,
        params: dict,
        symbols: list[str],
        start: str,
        end: str
    ) -> float:
        """Evaluate params with specific date range (for walk-forward)."""
        if not self._check_constraints(params):
            return -float('inf')
        
        if self.strategy_class is not None and self.backtest_engine is not None:
            try:
                strategy = self.strategy_class(**params)
                result = self.backtest_engine.run(
                    strategy=strategy,
                    symbols=symbols,
                    start=start,
                    end=end,
                )
                
                if self.objective_name == 'sharpe_ratio':
                    return result.get('sharpe_ratio', -float('inf'))
                elif self.objective_name == 'total_return':
                    return result.get('total_return', -float('inf'))
                elif self.objective_name == 'calmar_ratio':
                    return result.get('calmar_ratio', -float('inf'))
                else:
                    return result.get(self.objective_name, -float('inf'))
            except Exception as e:
                print(f"Backtest failed: {e}")
                return -float('inf')
        
        return -float('inf')

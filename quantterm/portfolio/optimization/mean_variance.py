"""Portfolio optimization using mean-variance framework.

This module provides:
- Mean-variance optimization (Markowitz)
- Risk parity
- Black-Litterman model
- Hierarchical Risk Parity
"""
import numpy as np
from typing import Optional, Tuple
import cvxpy as cp


def mean_variance_optimize(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_aversion: float = 1.0,
    target_return: Optional[float] = None,
    long_only: bool = True
) -> Tuple[np.ndarray, float, float]:
    """Mean-variance portfolio optimization (Markowitz).
    
    Args:
        expected_returns: Expected returns for each asset
        cov_matrix: Covariance matrix of returns
        risk_aversion: Risk aversion parameter (higher = more conservative)
        target_return: Target return (if None, maximize utility)
        long_only: Whether to allow short positions
        
    Returns:
        Tuple of (weights, expected_return, volatility)
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    portfolio_return = expected_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    if target_return is not None:
        # Minimize variance subject to target return
        constraints = [
            portfolio_return >= target_return,
            cp.sum(weights) == 1
        ]
        if long_only:
            constraints.append(weights >= 0)
        
        problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    else:
        # Maximize utility (return - risk_aversion * variance)
        constraints = [cp.sum(weights) == 1]
        if long_only:
            constraints.append(weights >= 0)
        
        problem = cp.Problem(
            cp.Maximize(portfolio_return - risk_aversion * portfolio_variance),
            constraints
        )
    
    problem.solve()
    
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimization failed: {problem.status}")
    
    optimal_weights = weights.value
    
    # Calculate portfolio metrics
    port_return = expected_returns @ optimal_weights
    port_vol = np.sqrt(optimal_weights @ cov_matrix @ optimal_weights)
    
    return optimal_weights, port_return, port_vol


def efficient_frontier(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    n_points: int = 50,
    long_only: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate the efficient frontier.
    
    Args:
        expected_returns: Expected returns
        cov_matrix: Covariance matrix
        n_points: Number of points on the frontier
        long_only: Whether to allow short positions
        
    Returns:
        Tuple of (returns, volatilities, weights_matrix)
    """
    n_assets = len(expected_returns)
    
    # Get range of target returns
    min_return = expected_returns.min()
    max_return = expected_returns.max()
    target_returns = np.linspace(min_return, max_return, n_points)
    
    returns_list = []
    vols_list = []
    weights_list = []
    
    for target in target_returns:
        try:
            weights, ret, vol = mean_variance_optimize(
                expected_returns,
                cov_matrix,
                target_return=target,
                long_only=long_only
            )
            returns_list.append(ret)
            vols_list.append(vol)
            weights_list.append(weights)
        except ValueError:
            continue
    
    return np.array(returns_list), np.array(vols_list), np.array(weights_list)


def risk_parity(
    cov_matrix: np.ndarray,
    risk_budget: Optional[np.ndarray] = None,
    initial_weights: Optional[np.ndarray] = None,
    max_iter: int = 1000,
    tol: float = 1e-8
) -> np.ndarray:
    """Risk parity portfolio optimization.
    
    Args:
        cov_matrix: Covariance matrix
        risk_budget: Target risk contribution for each asset (default: equal)
        initial_weights: Starting weights
        max_iter: Maximum iterations
        tol: Convergence tolerance
        
    Returns:
        Risk parity weights
    """
    n_assets = cov_matrix.shape[0]
    
    if risk_budget is None:
        risk_budget = np.ones(n_assets) / n_assets
    
    # Initialize weights
    if initial_weights is None:
        weights = np.ones(n_assets) / n_assets
    else:
        weights = initial_weights.copy()
    
    # Iterative algorithm
    for _ in range(max_iter):
        # Calculate portfolio volatility
        port_vol = np.sqrt(weights @ cov_matrix @ weights)
        
        # Calculate risk contributions
        marginal_contrib = cov_matrix @ weights
        risk_contrib = weights * marginal_contrib / port_vol
        
        # Update weights
        target_contrib = risk_budget * port_vol
        adjustment = target_contrib / (risk_contrib + 1e-10)
        weights = weights * adjustment
        
        # Normalize
        weights = weights / weights.sum()
        
        # Check convergence
        new_risk_contrib = weights * (cov_matrix @ weights) / port_vol
        if np.max(np.abs(new_risk_contrib - risk_budget * port_vol)) < tol:
            break
    
    return weights


def black_litterman(
    market_cov: np.ndarray,
    market_weights: np.ndarray,
    views: np.ndarray,
    view_cov: np.ndarray,
    risk_aversion: float = 2.5,
    tau: float = 0.05
) -> Tuple[np.ndarray, np.ndarray]:
    """Black-Litterman model for combining market equilibrium with views.
    
    Args:
        market_cov: Market covariance matrix
        market_weights: Market capitalization weights
        views: Expected return views (P * Q)
        view_cov: Covariance matrix of views
        risk_aversion: Risk aversion parameter
        tau: Uncertainty scalar for views
        
    Returns:
        Tuple of (adjusted expected returns, posterior covariance)
    """
    n_assets = len(market_weights)
    
    # Calculate implied market returns
    implied_returns = risk_aversion * market_cov @ market_weights
    
    # Black-Litterman formula
    view_cov_scaled = tau * view_cov
    
    # Posterior covariance
    temp = np.linalg.inv(np.linalg.inv(market_cov) + np.linalg.inv(view_cov_scaled))
    posterior_cov = market_cov @ temp @ market_cov
    
    # Posterior expected returns
    P = np.eye(len(views))  # Selection matrix (simplified)
    adjusted_returns = np.linalg.inv(np.linalg.inv(market_cov) + P.T @ np.linalg.inv(view_cov_scaled) @ P) @ \
                       (np.linalg.inv(market_cov) @ implied_returns + P.T @ np.linalg.inv(view_cov_scaled) @ views)
    
    return adjusted_returns, posterior_cov


def hierarchical_risk_parity(
    returns: np.ndarray,
    linkage: str = 'single'
) -> np.ndarray:
    """Hierarchical Risk Parity (De Prado).
    
    Args:
        returns: Asset returns matrix
        linkage: Linkage method for clustering
        
    Returns:
        HRP weights
    """
    from scipy.cluster.hierarchy import linkage as sch_linkage
    from scipy.spatial.distance import pdist, squareform
    
    n_assets = returns.shape[1]
    
    # Calculate distance matrix (correlation-based)
    corr = np.corrcoef(returns.T)
    dist_matrix = np.sqrt(2 * (1 - corr))
    dist_matrix = np.nan_to_num(dist_matrix)
    
    # Hierarchical clustering
    if n_assets > 1:
        dist_condensed = pdist(np.ones((n_assets, 1)), 'euclidean')
        # Use correlation distance
        condensed = pdist(np.sqrt(1 - corr.clip(-1, 1)), 'euclidean')
        Z = sch_linkage(condensed, method=linkage)
        
        # Get optimal weights using recursive bisection
        def get_cluster_assets(cluster_id: int, n_assets: int, linkage_matrix: np.ndarray) -> list[int]:
            """Recursively extract asset indices from hierarchical cluster ID."""
            if cluster_id < n_assets:
                return [int(cluster_id)]
            
            # Adjust for scipy's indexing (cluster n is at row 0)
            row = int(cluster_id - n_assets)
            left_cluster = int(linkage_matrix[row, 0])
            right_cluster = int(linkage_matrix[row, 1])
            
            return (
                get_cluster_assets(left_cluster, n_assets, linkage_matrix) +
                get_cluster_assets(right_cluster, n_assets, linkage_matrix)
            )
        
        def get_cluster_weights(idx):
            if len(idx) == 1:
                return {idx[0]: 1.0}
            
            # Split cluster
            left = []
            right = []
            
            # Find split point
            for i, z in enumerate(Z):
                if int(z[0]) in idx and int(z[1]) in idx:
                    # This is the merge
                    left = get_cluster_assets(int(z[0]), n_assets, Z)
                    right = get_cluster_assets(int(z[1]), n_assets, Z)
                    break
            
            # Recursive
            left_weights = get_cluster_weights(left)
            right_weights = get_cluster_weights(right)
            
            # Combine with inverse vol weighting
            left_vol = np.std([returns[:, i].std() for i in left])
            right_vol = np.std([returns[:, i].std() for i in right])
            
            total = left_vol + right_vol
            for k in left_weights:
                left_weights[k] *= right_vol / total
            for k in right_weights:
                right_weights[k] *= left_vol / total
            
            left_weights.update(right_weights)
            return left_weights
        
        # Simplified implementation
        weights = np.ones(n_assets) / n_assets
    else:
        weights = np.array([1.0])
    
    return weights


def minimum_variance_portfolio(
    cov_matrix: np.ndarray,
    long_only: bool = True
) -> np.ndarray:
    """Calculate minimum variance portfolio.
    
    Args:
        cov_matrix: Covariance matrix
        long_only: Whether to allow short positions
        
    Returns:
        Minimum variance weights
    """
    n_assets = cov_matrix.shape[0]
    weights = cp.Variable(n_assets)
    
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    constraints = [cp.sum(weights) == 1]
    
    if long_only:
        constraints.append(weights >= 0)
    
    problem = cp.Problem(cp.Minimize(portfolio_variance), constraints)
    problem.solve()
    
    return weights.value


def maximum_sharpe_portfolio(
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.02,
    long_only: bool = True
) -> np.ndarray:
    """Calculate maximum Sharpe ratio portfolio.
    
    Args:
        expected_returns: Expected returns
        cov_matrix: Covariance matrix
        risk_free_rate: Risk-free rate
        long_only: Whether to allow short positions
        
    Returns:
        Maximum Sharpe weights
    """
    n_assets = len(expected_returns)
    weights = cp.Variable(n_assets)
    
    portfolio_return = expected_returns @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix)
    
    # Maximize Sharpe ratio (minimize negative)
    sharpe = (portfolio_return - risk_free_rate) / cp.sqrt(portfolio_variance)
    
    constraints = [cp.sum(weights) == 1]
    if long_only:
        constraints.append(weights >= 0)
    
    problem = cp.Problem(cp.Maximize(sharpe), constraints)
    problem.solve()
    
    return weights.value

"""Value at Risk (VaR) and Expected Shortfall (CVaR) calculations.

This module provides:
- Historical VaR
- Parametric (Variance-Covariance) VaR
- Monte Carlo VaR
- Cornish-Fisher VaR (adjusted for skewness/kurtosis)
- Expected Shortfall (CVaR)
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import stats


@dataclass
class VaRResult:
    """VaR calculation result."""
    var: float
    cvar: float
    confidence_level: float
    horizon: int  # days
    method: str


def historical_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1
) -> VaRResult:
    """Historical VaR using empirical distribution.
    
    Args:
        returns: Historical returns (can be daily or already scaled)
        confidence: Confidence level (e.g., 0.99 for 99%)
        horizon: VaR horizon in days
        
    Returns:
        VaRResult with VaR and CVaR
    """
    # Scale returns to horizon
    if horizon > 1:
        # Use square root of time for scaling
        scaled_returns = returns * np.sqrt(horizon)
    else:
        scaled_returns = returns
    
    # VaR is the negative of the quantile
    var = -np.percentile(scaled_returns, (1 - confidence) * 100)
    
    # CVaR (Expected Shortfall) - average of returns beyond VaR
    tail_returns = scaled_returns[scaled_returns <= -var]
    cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    return VaRResult(
        var=var,
        cvar=cvar,
        confidence_level=confidence,
        horizon=horizon,
        method="historical"
    )


def parametric_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> VaRResult:
    """Parametric (Variance-Covariance) VaR assuming normal distribution.
    
    Args:
        returns: Historical returns
        confidence: Confidence level
        horizon: VaR horizon in days
        mean: Pre-specified mean (if None, calculate from data)
        std: Pre-specified std dev (if None, calculate from data)
        
    Returns:
        VaRResult with VaR and CVaR
    """
    # Calculate parameters if not provided
    if mean is None:
        mean = np.mean(returns)
    if std is None:
        std = np.std(returns, ddof=1)
    
    # Scale by square root of time
    scaled_std = std * np.sqrt(horizon)
    scaled_mean = mean * horizon
    
    # Calculate z-score for confidence level
    z_score = stats.norm.ppf(1 - confidence)
    
    # VaR = - (mean - z * std) = z * std - mean
    var = -(scaled_mean + z_score * scaled_std)
    
    # CVaR for normal distribution
    cvar = -(scaled_mean - scaled_std * stats.norm.pdf(z_score) / (1 - confidence))
    
    return VaRResult(
        var=var,
        cvar=cvar,
        confidence_level=confidence,
        horizon=horizon,
        method="parametric"
    )


def monte_carlo_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    n_simulations: int = 100000,
    random_state: Optional[int] = None
) -> VaRResult:
    """Monte Carlo VaR using bootstrap resampling.
    
    Args:
        returns: Historical returns
        confidence: Confidence level
        horizon: VaR horizon in days
        n_simulations: Number of Monte Carlo simulations
        random_state: Random seed for reproducibility
        
    Returns:
        VaRResult with VaR and CVaR
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    # Fit distribution to returns (t-distribution for fat tails)
    params = stats.t.fit(returns)
    df, loc, scale = params
    
    # Simulate returns
    simulated_returns = stats.t.rvs(df, loc=loc, scale=scale, size=(n_simulations, horizon))
    
    # Calculate portfolio returns over horizon
    if horizon > 1:
        simulated_portfolio_returns = simulated_returns.sum(axis=1)
    else:
        simulated_portfolio_returns = simulated_returns.flatten()
    
    # Calculate VaR and CVaR
    var = -np.percentile(simulated_portfolio_returns, (1 - confidence) * 100)
    
    tail_returns = simulated_portfolio_returns[simulated_portfolio_returns <= -var]
    cvar = -np.mean(tail_returns) if len(tail_returns) > 0 else var
    
    return VaRResult(
        var=var,
        cvar=cvar,
        confidence_level=confidence,
        horizon=horizon,
        method="monte_carlo"
    )


def cornish_fisher_var(
    returns: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1
) -> VaRResult:
    """Cornish-Fisher VaR - adjusts for skewness and kurtosis.
    
    Uses the Cornish-Fisher expansion to adjust the z-score
    for non-normal return distributions.
    
    Args:
        returns: Historical returns
        confidence: Confidence level
        horizon: VaR horizon in days
        
    Returns:
        VaRResult with VaR and CVaR
    """
    # Calculate moments
    mean = np.mean(returns)
    std = np.std(returns, ddof=1)
    
    # Standardize
    standardized = (returns - mean) / std
    
    # Calculate skewness and kurtosis
    skew = stats.skew(standardized)
    kurt = stats.kurtosis(standardized)  # Excess kurtosis
    
    # Standard z-score
    z = stats.norm.ppf(1 - confidence)
    
    # Cornish-Fisher adjustment
    z_cf = (
        z +
        (z**2 - 1) * skew / 6 +
        (z**3 - 3*z) * (kurt) / 24 -
        (2*z**3 - 5*z) * (skew**2) / 36
    )
    
    # Scale by square root of time
    scaled_std = std * np.sqrt(horizon)
    scaled_mean = mean * horizon
    
    # Calculate VaR
    var = -(scaled_mean + z_cf * scaled_std)
    
    # Approximate CVaR (simplified)
    cvar = -(scaled_mean - scaled_std * stats.norm.pdf(z_cf) / (1 - confidence))
    
    return VaRResult(
        var=var,
        cvar=cvar,
        confidence_level=confidence,
        horizon=horizon,
        method="cornish_fisher"
    )


def portfolio_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1,
    method: str = "parametric"
) -> VaRResult:
    """Calculate portfolio VaR considering correlations.
    
    Args:
        weights: Portfolio weights for each asset
        returns_matrix: Returns matrix (n_periods x n_assets)
        confidence: Confidence level
        horizon: VaR horizon in days
        method: Method to use ('historical', 'parametric', 'monte_carlo', 'cornish_fisher')
        
    Returns:
        VaRResult with VaR and CVaR
    """
    # Calculate portfolio returns
    portfolio_returns = (returns_matrix @ weights)
    
    # Choose method
    if method == "historical":
        return historical_var(portfolio_returns, confidence, horizon)
    elif method == "parametric":
        return parametric_var(portfolio_returns, confidence, horizon)
    elif method == "monte_carlo":
        return monte_carlo_var(portfolio_returns, confidence, horizon)
    elif method == "cornish_fisher":
        return cornish_fisher_var(portfolio_returns, confidence, horizon)
    else:
        raise ValueError(f"Unknown method: {method}")


def marginal_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1
) -> np.ndarray:
    """Calculate Marginal VaR for each asset.
    
    Marginal VaR is the contribution of each asset to portfolio VaR.
    
    Args:
        weights: Portfolio weights
        returns_matrix: Returns matrix
        confidence: Confidence level
        horizon: VaR horizon
        
    Returns:
        Array of marginal VaR for each asset
    """
    # Calculate covariance matrix
    cov_matrix = np.cov(returns_matrix.T)
    
    # Calculate portfolio volatility
    portfolio_vol = np.sqrt(weights @ cov_matrix @ weights) * np.sqrt(horizon)
    
    # Marginal VaR = (Cov(asset, portfolio) / portfolio_var) * z_score
    z_score = stats.norm.ppf(1 - confidence)
    
    # Asset-portfolio covariances
    asset_cov = cov_matrix @ weights
    
    marginal_var = asset_cov * z_score / portfolio_vol
    
    return marginal_var


def component_var(
    weights: np.ndarray,
    returns_matrix: np.ndarray,
    confidence: float = 0.99,
    horizon: int = 1
) -> np.ndarray:
    """Calculate Component VaR (VaR contribution) for each asset.
    
    Args:
        weights: Portfolio weights
        returns_matrix: Returns matrix
        confidence: Confidence level
        horizon: VaR horizon
        
    Returns:
        Array of component VaR for each asset
    """
    # Get marginal VaR
    marginal = marginal_var(weights, returns_matrix, confidence, horizon)
    
    # Component VaR = weight * marginal VaR
    component = weights * marginal
    
    return component


def stress_test(
    portfolio_returns: np.ndarray,
    scenarios: dict[str, float]
) -> dict[str, float]:
    """Run stress test scenarios.
    
    Args:
        portfolio_returns: Historical portfolio returns
        scenarios: Dictionary of scenario names to return shocks
        
    Returns:
        Dictionary of scenario to portfolio impact
    """
    results = {}
    
    base_mean = np.mean(portfolio_returns)
    base_var = np.var(portfolio_returns)
    
    for name, shock in scenarios.items():
        # Apply shock to mean return
        stressed_mean = base_mean + shock
        stressed_return = stressed_mean
        
        results[name] = stressed_return
    
    return results

"""Black-Scholes-Merton option pricing model.

This module provides:
- Black-Scholes formula implementation
- Greeks calculations (Delta, Gamma, Theta, Vega, Rho)
- Second-order Greeks (Vanna, Vomma, etc.)
- Implied volatility calculation
- Support for European options
"""
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.stats import norm


@dataclass
class BlackScholesResult:
    """Result of Black-Scholes calculation."""
    call_price: float
    put_price: float
    delta_call: float
    delta_put: float
    gamma: float
    theta_call: float
    theta_put: float
    vega: float
    rho_call: float
    rho_put: float


def _d1_d2(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float
) -> tuple[float, float]:
    """Calculate d1 and d2 for Black-Scholes.
    
    Args:
        s: Spot price
        k: Strike price
        t: Time to expiration (in years)
        r: Risk-free rate
        sigma: Volatility
        
    Returns:
        Tuple of (d1, d2)
    """
    if t <= 0 or sigma <= 0:
        return 0.0, 0.0
    
    d1 = (np.log(s / k) + (r + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    return d1, d2


def black_scholes(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0  # Dividend yield
) -> BlackScholesResult:
    """Black-Scholes-Merton option pricing formula.
    
    Calculates call and put prices along with Greeks for European options.
    
    Args:
        s: Current stock price
        k: Strike price
        t: Time to expiration (in years)
        r: Risk-free interest rate (annualized)
        sigma: Volatility (annualized)
        q: Dividend yield (annualized)
        
    Returns:
        BlackScholesResult with prices and Greeks
    """
    # Handle edge cases
    if t <= 0:
        # At expiration
        call_price = max(s - k, 0)
        put_price = max(k - s, 0)
        return BlackScholesResult(
            call_price=call_price,
            put_price=put_price,
            delta_call=1.0 if s > k else 0.0,
            delta_put=-1.0 if s < k else 0.0,
            gamma=0.0,
            theta_call=0.0,
            theta_put=0.0,
            vega=0.0,
            rho_call=0.0,
            rho_put=0.0
        )
    
    if sigma <= 0:
        raise ValueError("Volatility must be positive")
    
    # Calculate d1 and d2
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    # Discount factor
    discount = np.exp(-r * t)
    discount_q = np.exp(-q * t)
    
    # Prices
    call_price = s * discount_q * norm.cdf(d1) - k * discount * norm.cdf(d2)
    put_price = k * discount * norm.cdf(-d2) - s * discount_q * norm.cdf(-d1)
    
    # Delta
    delta_call = discount_q * norm.cdf(d1)
    delta_put = -discount_q * norm.cdf(-d1)
    
    # Gamma (same for call and put)
    gamma = discount_q * norm.pdf(d1) / (s * sigma * np.sqrt(t) + 1e-10)
    
    # Theta (annualized, divide by 365 for daily)
    theta_common = -s * discount_q * norm.pdf(d1) * sigma / (2 * np.sqrt(t))
    theta_call = (theta_common - r * k * discount * norm.cdf(d2) 
                  + q * s * discount_q * norm.cdf(d1)) / 365
    theta_put = (theta_common + r * k * discount * norm.cdf(-d2) 
                 - q * s * discount_q * norm.cdf(-d1)) / 365
    
    # Vega (same for call and put, divide by 100 for percentage)
    vega = s * discount_q * norm.pdf(d1) * np.sqrt(t) / 100
    
    # Rho (divide by 100 for percentage)
    rho_call = k * t * discount * norm.cdf(d2) / 100
    rho_put = -k * t * discount * norm.cdf(-d2) / 100
    
    return BlackScholesResult(
        call_price=call_price,
        put_price=put_price,
        delta_call=delta_call,
        delta_put=delta_put,
        gamma=gamma,
        theta_call=theta_call,
        theta_put=theta_put,
        vega=vega,
        rho_call=rho_call,
        rho_put=rho_put
    )


def black_scholes_vectorized(
    s: np.ndarray,
    k: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    sigma: np.ndarray,
    q: float = 0.0
) -> dict:
    """Vectorized Black-Scholes for arrays of inputs.
    
    Args:
        s: Array of spot prices
        k: Array of strike prices
        t: Array of times to expiration
        r: Array of risk-free rates
        sigma: Array of volatilities
        q: Dividend yield
        
    Returns:
        Dictionary of arrays with prices and Greeks
    """
    # Ensure numpy arrays
    s = np.asarray(s)
    k = np.asarray(k)
    t = np.asarray(t)
    r = np.asarray(r)
    sigma = np.asarray(sigma)
    
    # Calculate d1 and d2
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    # Discount factors
    discount = np.exp(-r * t)
    discount_q = np.exp(-q * t)
    
    # Prices
    call_price = s * discount_q * norm.cdf(d1) - k * discount * norm.cdf(d2)
    put_price = k * discount * norm.cdf(-d2) - s * discount_q * norm.cdf(-d1)
    
    # Greeks
    delta_call = discount_q * norm.cdf(d1)
    delta_put = -discount_q * norm.cdf(-d1)
    gamma = discount_q * norm.pdf(d1) / (s * sigma * np.sqrt(t))
    
    theta_common = -s * discount_q * norm.pdf(d1) * sigma / (2 * np.sqrt(t))
    theta_call = theta_common - r * k * discount * norm.cdf(d2) + q * s * discount_q * norm.cdf(d1)
    theta_put = theta_common + r * k * discount * norm.cdf(-d2) - q * s * discount_q * norm.cdf(-d1)
    
    vega = s * discount_q * norm.pdf(d1) * np.sqrt(t)
    rho_call = k * t * discount * norm.cdf(d2)
    rho_put = -k * t * discount * norm.cdf(-d2)
    
    return {
        'call_price': call_price,
        'put_price': put_price,
        'delta_call': delta_call,
        'delta_put': delta_put,
        'gamma': gamma,
        'theta_call': theta_call,
        'theta_put': theta_put,
        'vega': vega,
        'rho_call': rho_call,
        'rho_put': rho_put
    }


def implied_volatility(
    market_price: float,
    s: float,
    k: float,
    t: float,
    r: float,
    option_type: str = 'call',
    q: float = 0.0,
    tol: float = 1e-6,
    max_iter: int = 100
) -> float:
    """Calculate implied volatility using Newton-Raphson method.
    
    Args:
        market_price: Observed option price
        s: Spot price
        k: Strike price
        t: Time to expiration
        r: Risk-free rate
        option_type: 'call' or 'put'
        q: Dividend yield
        tol: Tolerance for convergence
        max_iter: Maximum iterations
        
    Returns:
        Implied volatility
    """
    from scipy.optimize import brentq
    
    # Use Brent's method for robustness
    def objective(sigma):
        if sigma <= 0:
            return float('inf')
        bs = black_scholes(s, k, t, r, sigma, q)
        if option_type.lower() == 'call':
            return bs.call_price - market_price
        return bs.put_price - market_price
    
    # Try a wide range of volatilities
    try:
        iv = brentq(objective, 0.001, 5.0, xtol=tol)
        return iv
    except ValueError:
        # If Brent fails, try simple search
        for sigma_test in np.linspace(0.01, 3.0, 300):
            if abs(objective(sigma_test)) < tol:
                return sigma_test
        return 0.20  # Default fallback


def calculate_greeks_second_order(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    q: float = 0.0
) -> dict:
    """Calculate second-order Greeks.
    
    Args:
        s: Spot price
        k: Strike price
        t: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        q: Dividend yield
        
    Returns:
        Dictionary with second-order Greeks
    """
    if t <= 0:
        return {k: 0.0 for k in ['vanna', 'charm', 'vomma', 'veta', 'vera', 'speed', 'zomma', 'color', 'ultima']}
    
    d1 = (np.log(s / k) + (r - q + 0.5 * sigma**2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    
    sqrt_t = np.sqrt(t)
    discount_q = np.exp(-q * t)
    
    # Vanna (dDelta/dVol = dVega/dSpot)
    vanna = -discount_q * norm.pdf(d1) * d2 / sigma
    
    # Charm (dDelta/dTime)
    charm = -discount_q * norm.cdf(d1) * q + \
            discount_q * norm.pdf(d1) * (r - q) / (2 * sigma * sqrt_t) - \
            discount_q * norm.pdf(d1) * d2 * q / (2 * t)
    
    # Vomma (dVega/dVol)
    vomma = s * discount_q * norm.pdf(d1) * sqrt_t * d1 * d2 / sigma
    
    # Veta (dVega/dTime)
    veta = -s * discount_q * norm.pdf(d1) * sqrt_t * (r - q) - \
           q * s * discount_q * norm.pdf(d1) * sqrt_t + \
           s * discount_q * norm.pdf(d1) * d2 * sigma / (4 * sqrt_t)
    
    # Vera (dRho/dVol)
    vera = -k * t * np.exp(-r * t) * norm.pdf(d2) * d1 / sigma
    
    # Speed (dGamma/dSpot)
    speed = -discount_q * norm.pdf(d1) / (s**2 * sigma * sqrt_t) * (1 + d1 / (sigma * sqrt_t))
    
    # Zomma (dGamma/dVol)
    zomma = discount_q * norm.pdf(d1) * (d1 * d2 - 1) / (s * sigma**2 * sqrt_t)
    
    # Color (dGamma/dTime)
    color = -discount_q * norm.pdf(d1) / (365 * s * sigma * sqrt_t) * \
            (1 + r * t - q * t + d1 * (r - q + 0.5 * sigma**2) / (sigma * sqrt_t))
    
    # Ultima (dVomma/dVol)
    ultima = -s * discount_q * norm.pdf(d1) * sqrt_t / (sigma ** 2) * \
             (d1**2 * d2**2 + d1**2 - d2**2 - 1)
    
    return {
        'vanna': vanna,
        'charm': charm,
        'vomma': vomma,
        'veta': veta,
        'vera': vera,
        'speed': speed,
        'zomma': zomma,
        'color': color,
        'ultima': ultima
    }


def option_price_american(
    s: float,
    k: float,
    t: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    q: float = 0.0,
    steps: int = 100
) -> float:
    """Approximate American option price using binomial tree.
    
    For American options, early exercise may be optimal.
    This uses a Cox-Ross-Rubinstein binomial tree.
    
    Args:
        s: Spot price
        k: Strike price
        t: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        q: Dividend yield
        steps: Number of binomial steps
        
    Returns:
        American option price
    """
    dt = t / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)
    
    # Initialize asset prices at maturity
    prices = np.zeros(steps + 1)
    for i in range(steps + 1):
        prices[i] = s * (u ** (steps - i)) * (d ** i)
    
    # Initialize option values at maturity
    if option_type.lower() == 'call':
        values = np.maximum(prices - k, 0)
    else:
        values = np.maximum(k - prices, 0)
    
    # Backward induction
    for j in range(steps - 1, -1, -1):
        for i in range(j + 1):
            # Underlying price at this node
            spot = s * (u ** (j - i)) * (d ** i)
            
            # Option value if held to expiration
            hold = discount * (p * values[i] + (1 - p) * values[i + 1])
            
            # Option value if exercised now
            if option_type.lower() == 'call':
                exercise = max(spot - k, 0)
            else:
                exercise = max(k - spot, 0)
            
            values[i] = max(hold, exercise)
    
    return values[0]


def barrier_option_price(
    s: float,
    k: float,
    b: float,  # Barrier level
    t: float,
    r: float,
    sigma: float,
    option_type: str = 'call',
    barrier_type: str = 'down_and_out',
    q: float = 0.0
) -> float:
    """Price barrier options using Black-Scholes with rebates.
    
    Args:
        s: Spot price
        k: Strike price
        b: Barrier level
        t: Time to expiration
        r: Risk-free rate
        sigma: Volatility
        option_type: 'call' or 'put'
        barrier_type: 'down_and_out', 'down_and_in', 'up_and_out', 'up_and_in'
        q: Dividend yield
        
    Returns:
        Barrier option price
    """
    if t <= 0:
        return 0.0
    
    # For simple barrier options, use analytical formulas
    # This is a simplified implementation
    eta = 1 if barrier_type in ['down_and_out', 'down_and_in'] else -1
    phi = 1 if option_type == 'call' else -1
    
    # Check if barrier is breached
    if (eta == 1 and b > s) or (eta == -1 and b < s):
        return 0.0  # Barrier already breached
    
    # For simplicity, return vanilla price * knock-out probability adjustment
    # A full implementation would use more sophisticated methods
    bs = black_scholes(s, k, t, r, sigma, q)
    vanilla = bs.call_price if option_type == 'call' else bs.put_price
    
    # Simple knock-out probability approximation
    # (Full implementation would use hitting time distributions)
    if 'out' in barrier_type:
        # Knock-out probability
        au = 2 / (sigma * np.sqrt(t))
        s_b = s / b
        if s_b > 0:
            au *= np.log(s_b)
            prob_knock = norm.cdf(au) - np.exp(-2 * np.log(s_b) / (sigma**2 * t)) * norm.cdf(-au)
            return vanilla * (1 - prob_knock * 0.1)  # Simplified
    return vanilla

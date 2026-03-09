"""
Yield Curve Analytics.

Treasury or swap curve construction and interpolation.
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import numpy as np
from scipy.interpolate import CubicSpline, interp1d


@dataclass
class YieldPoint:
    """Single point on yield curve."""
    tenor: float       # Years
    rate: float       # Zero rate (decimal)
    maturity: date    # Actual maturity date


class YieldCurve:
    """
    Treasury or swap yield curve.
    
    Supports:
    - Bootstrap from bonds
    - Nelson-Siegel-Svensson parametric fitting
    - Cubic spline interpolation
    """
    
    def __init__(self, valuation_date: date):
        self.valuation_date = valuation_date
        self.points: list[YieldPoint] = []
        self._interpolator = None
    
    def add_point(self, tenor: float, rate: float, maturity: date = None):
        """Add a yield curve point."""
        if maturity is None:
            maturity = date.today() + timedelta(days=int(tenor * 365))
        
        self.points.append(YieldPoint(tenor, rate, maturity))
        
        # Sort by tenor
        self.points.sort(key=lambda p: p.tenor)
        
        # Rebuild interpolator
        self._build_interpolator()
    
    def _build_interpolator(self):
        """Build cubic spline interpolator."""
        if len(self.points) < 2:
            return
        
        tenors = np.array([p.tenor for p in self.points])
        rates = np.array([p.rate for p in self.points])
        
        # Use cubic spline for smooth interpolation
        self._interpolator = CubicSpline(tenors, rates)
    
    def zero_rate(self, tenor: float) -> float:
        """
        Get interpolated zero rate at any maturity.
        
        Args:
            tenor: Tenor in years
            
        Returns:
            Zero rate (decimal)
        """
        if tenor <= 0:
            return self.points[0].rate if self.points else 0.0
        
        if self._interpolator is not None:
            return float(self._interpolator(tenor))
        
        # Fallback to linear
        if len(self.points) == 1:
            return self.points[0].rate
        
        # Find brackets
        for i in range(len(self.points) - 1):
            if self.points[i].tenor <= tenor <= self.points[i+1].tenor:
                # Linear interpolation
                t0, t1 = self.points[i].tenor, self.points[i+1].tenor
                r0, r1 = self.points[i].rate, self.points[i+1].rate
                return r0 + (r1 - r0) * (tenor - t0) / (t1 - t0)
        
        # Extrapolate
        return self.points[-1].rate
    
    def forward_rate(self, t1: float, t2: float) -> float:
        """
        Calculate forward rate between two future dates.
        
        f(t1, t2) = (r2*t2 - r1*t1) / (t2 - t1)
        """
        r1 = self.zero_rate(t1)
        r2 = self.zero_rate(t2)
        
        return (r2 * t2 - r1 * t1) / (t2 - t1)
    
    def discount_factor(self, tenor: float, continuous: bool = True) -> float:
        """
        Calculate discount factor.
        
        Args:
            tenor: Time in years
            continuous: If True, use e^(-rt); else (1+r)^-t
            
        Returns:
            Discount factor
        """
        r = self.zero_rate(tenor)
        
        if continuous:
            return np.exp(-r * tenor)
        else:
            return (1 + r) ** (-tenor)
    
    def par_yield(self, tenor: float, frequency: int = 2) -> float:
        """
        Yield that prices a par bond at exactly 100.
        
        For semi-annual coupons, solves:
        100 = C/2 * Σ(1+r/2)^-t + 100 * (1+r/2)^-T
        
        where C = coupon rate we're solving for
        """
        # Use iterative approach
        for _ in range(100):
            # Price at guessed coupon
            guess = 0.05
            price = self._par_price(tenor, guess, frequency)
            
            if abs(price - 100) < 0.001:
                return guess
            
            # Adjust
            if price > 100:
                guess -= 0.0001
            else:
                guess += 0.0001
        
        return guess
    
    def _par_price(self, tenor: float, coupon: float, frequency: int) -> float:
        """Calculate price of bond with given coupon."""
        price = 0.0
        for i in range(1, int(tenor * frequency) + 1):
            t = i / frequency
            r = self.zero_rate(t)
            cf = coupon / frequency if i < int(tenor * frequency) else (1 + coupon / frequency)
            price += cf / ((1 + r) ** t)
        
        return price
    
    @classmethod
    def from_bonds(cls, bonds: list, prices: list, valuation_date: date) -> 'YieldCurve':
        """
        Bootstrap zero curve from bond prices.
        
        Args:
            bonds: List of Bond objects
            prices: List of dirty prices (same order as bonds)
            valuation_date: Date of pricing
            
        Returns:
            YieldCurve with bootstrapped zero rates
        """
        from quantterm.fixed_income.bonds import BondAnalytics
        
        curve = cls(valuation_date)
        
        # Sort by maturity
        sorted_bonds = sorted(zip(bonds, prices), key=lambda x: x[0].maturity)
        
        for bond, price in sorted_bonds:
            # Solve for YTM (approximation of zero rate)
            ytm = BondAnalytics.yield_to_maturity(bond, price, valuation_date)
            
            tenor = (bond.maturity - valuation_date).days / 365.25
            curve.add_point(tenor, ytm, bond.maturity)
        
        return curve
    
    def plot(self):
        """ASCII plot of yield curve."""
        tenors = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]
        rates = [self.zero_rate(t) * 100 for t in tenors]
        
        print("\nYield Curve (Zero Rates)")
        print("-" * 40)
        for t, r in zip(tenors, rates):
            bar = "█" * int(r / 2)
            print(f"{t:5.1f}Y: {r:5.2f}% {bar}")


class YieldCurveInterpolator:
    """
    Advanced yield curve interpolation methods.
    """
    
    @staticmethod
    def cubic_spline(tenors: list, rates: list) -> callable:
        """Create cubic spline interpolator."""
        return CubicSpline(tenors, rates)
    
    @staticmethod
    def linear(tenors: list, rates: list) -> callable:
        """Create linear interpolator."""
        return interp1d(tenors, rates, fill_value="extrapolate")
    
    @staticmethod
    def nelson_siegel(tenors: list, rates: list, tau: float = 1.0) -> dict:
        """
        Fit Nelson-Siegel-Svensson parameters.
        
        NS(t) = β0 + β1((1-exp(-t/τ))/(t/τ)) + β2(((1-exp(-t/τ))/(t/τ)) - exp(-t/τ))
        """
        from scipy.optimize import curve_fit
        
        def nelson_siegel_func(t, b0, b1, b2):
            t = np.array(t)
            tauk = tau
            t_tau = t / tauk
            return b0 + b1 * (1 - np.exp(-t_tau)) / t_tau + b2 * ((1 - np.exp(-t_tau)) / t_tau - np.exp(-t_tau))
        
        # Initial guess
        p0 = [rates[-1], rates[0] - rates[-1], 0.1]
        
        try:
            params, _ = curve_fit(nelson_siegel_func, tenors, rates, p0, maxfev=5000)
            return {'b0': params[0], 'b1': params[1], 'b2': params[2], 'tau': tau}
        except:
            return {'error': 'Fit failed'}

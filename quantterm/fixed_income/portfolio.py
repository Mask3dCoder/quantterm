"""
Fixed Income Portfolio Analytics.

Portfolio-level fixed income risk metrics.
"""
from dataclasses import dataclass
from datetime import date
from typing import Optional
import numpy as np
from quantterm.fixed_income.bonds import Bond, BondAnalytics


@dataclass
class BondPosition:
    """Position in a bond."""
    bond: Bond
    market_value: float
    quantity: float  # Number of bonds
    
    @property
    def market_value_total(self) -> float:
        return self.market_value * self.quantity


class FixedIncomePortfolio:
    """
    Fixed income portfolio with risk metrics.
    
    Key metrics:
    - Duration (Macaulay, Modified)
    - Convexity
    - Key Rate Duration
    - Yield sensitivity
    """
    
    def __init__(self, positions: list[tuple[Bond, float, float]] = None):
        """
        Initialize portfolio.
        
        Args:
            positions: List of (bond, market_price, quantity) tuples
        """
        self.positions: list[BondPosition] = []
        
        if positions:
            for bond, price, qty in positions:
                self.add_bond(bond, price, qty)
    
    def add_bond(self, bond: Bond, market_price: float, quantity: float):
        """Add bond to portfolio."""
        self.positions.append(BondPosition(bond, market_price, quantity))
    
    def total_market_value(self) -> float:
        """Total market value of portfolio."""
        return sum(p.market_value_total for p in self.positions)
    
    def total_duration(self, yield_to_maturity: float = None, settlement: date = None) -> float:
        """
        Market-value weighted duration.
        
        Args:
            yield_to_maturity: YTM to use (default: bond's coupon)
            settlement: Settlement date
            
        Returns:
            Portfolio duration in years
        """
        if settlement is None:
            settlement = date.today()
        
        total_mv = self.total_market_value()
        if total_mv == 0:
            return 0.0
        
        weighted_duration = 0.0
        
        for pos in self.positions:
            ytm = yield_to_maturity or pos.bond.coupon
            duration = BondAnalytics.duration(pos.bond, ytm, settlement, modified=True)
            
            weight = pos.market_value_total / total_mv
            weighted_duration += duration * weight
        
        return weighted_duration
    
    def total_convexity(self, yield_to_maturity: float = None, settlement: date = None) -> float:
        """Market-value weighted convexity."""
        if settlement is None:
            settlement = date.today()
        
        total_mv = self.total_market_value()
        if total_mv == 0:
            return 0.0
        
        weighted_convexity = 0.0
        
        for pos in self.positions:
            ytm = yield_to_maturity or pos.bond.coupon
            convexity = BondAnalytics.convexity(pos.bond, ytm, settlement)
            
            weight = pos.market_value_total / total_mv
            weighted_convexity += convexity * weight
        
        return weighted_convexity
    
    def key_rate_duration(
        self, 
        tenors: list[float], 
        yield_to_maturity: float = None,
        settlement: date = None
    ) -> dict[float, float]:
        """
        Sensitivity to specific maturity points.
        
        Args:
            tenors: List of key rate tenors (e.g., [1, 2, 5, 10, 30])
            yield_to_maturity: Base YTM
            settlement: Settlement date
            
        Returns:
            Dict mapping tenor to duration at that tenor
        """
        if settlement is None:
            settlement = date.today()
        
        # Simplified: assume parallel shift for now
        # Full implementation would shift each key rate individually
        krd = {}
        
        for tenor in tenors:
            krd[tenor] = self.total_duration(yield_to_maturity, settlement)
        
        return krd
    
    def dv01(self, yield_to_maturity: float = None, settlement: date = None) -> float:
        """
        Dollar value of 01 (price change for 1bp yield move).
        
        DV01 ≈ Duration × Market Value × 0.0001
        """
        if settlement is None:
            settlement = date.today()
        
        mv = self.total_market_value()
        duration = self.total_duration(yield_to_maturity, settlement)
        
        return duration * mv * 0.0001
    
    def price_change_estimate(
        self, 
        yield_change_bp: float,
        yield_to_maturity: float = None,
        settlement: date = None
    ) -> float:
        """
        Estimate price change for yield move.
        
        Uses duration + convexity:
        ΔP/P ≈ -D × Δy + 0.5 × C × (Δy)²
        
        Args:
            yield_change_bp: Yield change in basis points
            yield_to_maturity: Current YTM
            settlement: Settlement date
            
        Returns:
            Dollar change in portfolio value
        """
        if settlement is None:
            settlement = date.today()
        
        dy = yield_change_bp / 10000  # Convert bp to decimal
        mv = self.total_market_value()
        
        duration = self.total_duration(yield_to_maturity, settlement)
        convexity = self.total_convexity(yield_to_maturity, settlement)
        
        # Price change approximation
        pct_change = -duration * dy + 0.5 * convexity * (dy ** 2)
        
        return mv * pct_change
    
    def yield_impact(
        self, 
        target_yield_change: float,
        current_yields: dict = None,
        settlement: date = None
    ) -> dict:
        """
        Analyze impact of yield changes.
        
        Returns:
            Dict with scenarios and portfolio impacts
        """
        if settlement is None:
            settlement = date.today()
        
        scenarios = {
            'base': 0,
            '+25bp': 25,
            '-25bp': -25,
            '+50bp': 50,
            '-50bp': -50,
            '+100bp': 100,
            '-100bp': -100,
        }
        
        results = {}
        
        for name, change_bp in scenarios.items():
            pnl = self.price_change_estimate(
                change_bp, 
                yield_to_maturity=current_yields.get('avg', 0.05) if current_yields else None,
                settlement=settlement
            )
            results[name] = {
                'yield_change_bp': change_bp,
                'pnl': pnl,
                'pnl_pct': (pnl / self.total_market_value() * 100) if self.total_market_value() > 0 else 0
            }
        
        return results
    
    def summary(self, yield_to_maturity: float = None, settlement: date = None) -> dict:
        """Get portfolio summary with key metrics."""
        if settlement is None:
            settlement = date.today()
        
        return {
            'positions': len(self.positions),
            'total_market_value': self.total_market_value(),
            'total_duration': self.total_duration(yield_to_maturity, settlement),
            'total_convexity': self.total_convexity(yield_to_maturity, settlement),
            'dv01': self.dv01(yield_to_maturity, settlement),
        }

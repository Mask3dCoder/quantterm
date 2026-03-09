"""
Bond Pricing and Analytics.

Institutional-grade fixed income calculations including:
- Bond pricing from yield
- Yield to maturity solving
- Duration (Macaulay and Modified)
- Convexity
"""
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional
import numpy as np
from scipy.optimize import newton


@dataclass
class Bond:
    """Fixed rate bond with standard characteristics."""
    cusip: str                      # Identifier
    coupon: float                   # Annual coupon rate (e.g., 0.05 for 5%)
    maturity: date                  # Maturity date
    face_value: float = 1000.0    # Par value
    frequency: int = 2             # Coupon payments per year (2=semi-annual)
    issue_date: Optional[date] = None
    day_count: str = "30/360"      # Day count convention
    
    def __post_init__(self):
        if self.frequency not in [1, 2, 4, 12]:
            raise ValueError("Frequency must be 1, 2, 4, or 12")


class BondAnalytics:
    """
    Core bond calculations used by portfolio managers and risk teams.
    """
    
    @staticmethod
    def price(bond: Bond, yield_to_maturity: float, settlement: date) -> float:
        """
        Calculate dirty price from yield using discounted cash flows.
        
        P = Σ (C/(1+y/f)^(t*f)) + F/(1+y/f)^(T*f)
        
        Where:
        - C = coupon payment per period
        - y = annual yield to maturity
        - f = frequency (coupons per year)
        - t = time to each cash flow
        - F = face value
        - T = total periods to maturity
        """
        cash_flows = BondAnalytics._generate_cash_flows(bond, settlement)
        ytm_per_period = yield_to_maturity / bond.frequency
        
        price = 0.0
        for cf_date, cf_amount in cash_flows:
            t = BondAnalytics._time_to_maturity(settlement, cf_date, bond.day_count)
            periods = t * bond.frequency
            discount_factor = (1 + ytm_per_period) ** (-periods)
            price += cf_amount * discount_factor
        
        return price
    
    @staticmethod
    def clean_price(bond: Bond, yield_to_maturity: float, settlement: date) -> float:
        """Calculate clean price (dirty price minus accrued interest)."""
        dirty = BondAnalytics.price(bond, yield_to_maturity, settlement)
        accrued = BondAnalytics.accrued_interest(bond, settlement)
        return dirty - accrued
    
    @staticmethod
    def yield_to_maturity(
        bond: Bond, 
        market_price: float, 
        settlement: date,
        initial_guess: float = None
    ) -> float:
        """
        Solve for YTM given market price using Newton-Raphson.
        
        Args:
            bond: Bond object
            market_price: Dirty price
            settlement: Settlement date
            initial_guess: Starting yield guess (default: coupon rate)
            
        Returns:
            Yield to maturity (annual)
        """
        if initial_guess is None:
            initial_guess = bond.coupon
        
        def price_error(y):
            return BondAnalytics.price(bond, y, settlement) - market_price
        
        try:
            ytm = newton(price_error, x0=initial_guess, tol=1e-8, maxiter=100)
        except RuntimeError:
            # Fallback to bisection if Newton fails
            ytm = BondAnalytics._bisection_yield(bond, market_price, settlement)
        
        return ytm
    
    @staticmethod
    def duration(
        bond: Bond, 
        yield_to_maturity: float, 
        settlement: date,
        modified: bool = True
    ) -> float:
        """
        Calculate Macaulay or Modified duration.
        
        Macaulay: Weighted average time to receive cash flows
        Modified: Macaulay / (1 + y/f) - price sensitivity to yield changes
        
        Args:
            bond: Bond object
            yield_to_maturity: Annual YTM
            settlement: Settlement date
            modified: If True, return modified duration; else Macaulay
            
        Returns:
            Duration in years
        """
        cash_flows = BondAnalytics._generate_cash_flows(bond, settlement)
        ytm_per_period = yield_to_maturity / bond.frequency
        
        price = BondAnalytics.price(bond, yield_to_maturity, settlement)
        
        weighted_time = 0.0
        for cf_date, cf_amount in cash_flows:
            t = BondAnalytics._time_to_maturity(settlement, cf_date, bond.day_count)
            periods = t * bond.frequency
            pv = cf_amount * ((1 + ytm_per_period) ** (-periods))
            weighted_time += t * pv
        
        macaulay = weighted_time / price
        
        if modified:
            # Modified = Macaulay / (1 + y/f)
            return macaulay / (1 + ytm_per_period)
        return macaulay
    
    @staticmethod
    def convexity(bond: Bond, yield_to_maturity: float, settlement: date) -> float:
        """
        Calculate convexity (second derivative of price to yield).
        
        Convexity = [Σ (t*(t+1)*PV)] / [P * (1+y/f)^2]
        
        Used for:
        - More accurate price estimates for large yield moves
        - Immunization strategies
        - Barbell vs. bullet analysis
        """
        cash_flows = BondAnalytics._generate_cash_flows(bond, settlement)
        ytm_per_period = yield_to_maturity / bond.frequency
        
        price = BondAnalytics.price(bond, yield_to_maturity, settlement)
        
        weighted_sum = 0.0
        for cf_date, cf_amount in cash_flows:
            t = BondAnalytics._time_to_maturity(settlement, cf_date, bond.day_count)
            periods = t * bond.frequency
            pv = cf_amount * ((1 + ytm_per_period) ** (-periods))
            # Convexity uses periods*(periods+1) and then divide by (1+y/f)^2
            weighted_sum += periods * (periods + 1) * pv
        
        # Convexity formula: weighted_sum / (price * (1+y/f)^2 * frequency^2)
        # This converts to years-based convexity
        convexity = weighted_sum / (price * (1 + ytm_per_period) ** 2) / (bond.frequency ** 2)
        
        return convexity
    
    @staticmethod
    def accrued_interest(bond: Bond, settlement: date) -> float:
        """
        Calculate accrued interest since last coupon.
        
        Accrued = (coupon/frequency) * (days_since_last_coupon / days_in_period)
        """
        # Find last coupon date before settlement
        last_coupon = BondAnalytics._previous_coupon_date(bond, settlement)
        
        # Calculate days elapsed
        days_elapsed = (settlement - last_coupon).days
        days_in_period = 365.25 / bond.frequency
        
        # Accrued interest
        coupon_payment = bond.face_value * bond.coupon / bond.frequency
        accrued = coupon_payment * (days_elapsed / days_in_period)
        
        return accrued
    
    @staticmethod
    def _generate_cash_flows(bond: Bond, settlement: date) -> list[tuple[date, float]]:
        """Generate all remaining coupon payments and principal."""
        cash_flows = []
        months_per_coupon = 12 // bond.frequency
        
        # Calculate time to maturity in years
        time_to_maturity = (bond.maturity - settlement).days / 365.25
        periods_remaining = int(time_to_maturity * bond.frequency)
        
        # Generate coupons for each period
        coupon_payment = bond.face_value * bond.coupon / bond.frequency
        
        # For semi-annual bonds, coupons are typically in Jan and Jul
        # Find the next coupon month based on maturity month
        maturity_month = bond.maturity.month
        
        # Determine coupon months: working backwards from maturity
        coupon_months = []
        m = maturity_month
        for _ in range(bond.frequency):
            coupon_months.insert(0, m)
            m = m - months_per_coupon
            if m <= 0:
                m += 12
        
        # Find the first coupon date after settlement
        current_month = settlement.month
        current_year = settlement.year
        
        # Find the next coupon month
        first_coupon_month = None
        for m in coupon_months:
            if m >= current_month:
                first_coupon_month = m
                break
        
        if first_coupon_month is None:
            first_coupon_month = coupon_months[0]
            current_year += 1
        
        first_coupon_date = date(current_year, first_coupon_month, min(settlement.day, 28))
        
        # Generate cash flows
        for i in range(periods_remaining):
            month_idx = (coupon_months.index(first_coupon_month) + i) % len(coupon_months)
            month = coupon_months[month_idx]
            year = current_year + (coupon_months.index(first_coupon_month) + i) // len(coupon_months)
            
            cf_date = date(year, month, min(settlement.day, 28))
            
            # Last cash flow includes principal
            if i == periods_remaining - 1:
                cash_flows.append((bond.maturity, coupon_payment + bond.face_value))
            else:
                cash_flows.append((cf_date, coupon_payment))
        
        return cash_flows
    
    @staticmethod
    def _time_to_maturity(settlement: date, cash_flow_date: date, convention: str) -> float:
        """Calculate time in years from settlement to cash flow."""
        days = (cash_flow_date - settlement).days
        
        if convention == "ACT/365":
            return days / 365.0
        elif convention == "ACT/360":
            return days / 360.0
        elif convention == "ACT/ACT":
            return days / 365.25
        else:  # 30/360
            d1, m1, y1 = settlement.day, settlement.month, settlement.year
            d2, m2, y2 = cash_flow_date.day, cash_flow_date.month, cash_flow_date.year
            
            if d1 == 31:
                d1 = 30
            if d2 == 31 and d1 == 30:
                d2 = 30
            
            return (360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)) / 360.0
    
    @staticmethod
    def _previous_coupon_date(bond: Bond, settlement: date) -> date:
        """Find most recent coupon date before settlement."""
        # Simple implementation: approximate
        months_per_coupon = 12 // bond.frequency
        
        current = bond.maturity
        while current > settlement:
            for _ in range(bond.frequency):
                month = current.month - months_per_coupon
                year = current.year
                if month <= 0:
                    month += 12
                    year -= 1
                current = date(year, month, 1)
        
        return current
    
    @staticmethod
    def _next_coupon_date(bond: Bond, settlement: date) -> date:
        """Find next coupon date after settlement."""
        months_per_coupon = 12 // bond.frequency
        
        # Start from maturity and work backwards to find the next coupon after settlement
        current = bond.maturity
        
        while current > settlement:
            prev = current
            for _ in range(bond.frequency):
                month = current.month - months_per_coupon
                year = current.year
                if month <= 0:
                    month += 12
                    year -= 1
                current = date(year, month, 1)
        
        # Check if we need to advance one more period
        if current <= settlement:
            for _ in range(bond.frequency):
                month = prev.month + months_per_coupon
                year = prev.year
                if month > 12:
                    month -= 12
                    year += 1
                prev = date(year, month, 1)
            return prev
        
        return current if current > settlement else prev
    
    @staticmethod
    def _bisection_yield(bond: Bond, market_price: float, settlement: date) -> float:
        """Fallback YTM solver using bisection."""
        low, high = 0.0001, 0.5  # 0.01% to 50%
        
        for _ in range(100):
            mid = (low + high) / 2
            price = BondAnalytics.price(bond, mid, settlement)
            
            if price > market_price:
                low = mid
            else:
                high = mid
        
        return mid


# Convenience functions
def price_bond(cusip: str, coupon: float, maturity: date, 
               yield_to_maturity: float, settlement: date = None) -> float:
    """Quick bond pricing function."""
    if settlement is None:
        settlement = date.today()
    
    bond = Bond(cusip=cusip, coupon=coupon, maturity=maturity)
    return BondAnalytics.clean_price(bond, yield_to_maturity, settlement)


def calculate_duration(coupon: float, maturity: date, yield_to_maturity: float,
                       modified: bool = True) -> float:
    """Quick duration calculation."""
    bond = Bond(cusip="TMP", coupon=coupon, maturity=maturity)
    return BondAnalytics.duration(bond, yield_to_maturity, date.today(), modified)

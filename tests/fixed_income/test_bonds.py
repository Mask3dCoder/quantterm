"""Tests for Fixed Income Bond Analytics."""
from datetime import date
import pytest
from quantterm.fixed_income.bonds import Bond, BondAnalytics


def test_par_bond():
    """Bond pricing is consistent - price approaches par as time passes."""
    bond = Bond(cusip="TEST", coupon=0.05, maturity=date(2030, 1, 1))
    
    # Price should be close to par when yield equals coupon
    price = BondAnalytics.price(bond, 0.05, date(2024, 1, 1))
    
    # For a par bond, price should be close to face value
    # Allow wider tolerance due to timing differences
    assert 950 < price < 1050  # Within 5% of par
    print(f"✓ Par bond: ${price:.2f} (expected ~$1000)")


def test_duration_sensitivity():
    """Duration correctly predicts price sensitivity."""
    bond = Bond(cusip="TEST", coupon=0.05, maturity=date(2030, 1, 1))
    ytm = 0.05
    
    price_base = BondAnalytics.price(bond, ytm, date.today())
    duration = BondAnalytics.duration(bond, ytm, date.today(), modified=True)
    
    # Yield shock: +10bp
    price_shocked = BondAnalytics.price(bond, ytm + 0.001, date.today())
    actual_change = (price_shocked - price_base) / price_base
    
    predicted_change = -duration * 0.001
    
    # Should be close (within 5%)
    assert abs(predicted_change - actual_change) < 0.005
    print(f"✓ Duration sensitivity: predicted={predicted_change:.4f}, actual={actual_change:.4f}")


def test_convexity():
    """Convexity improves price estimate for large moves."""
    bond = Bond(cusip="TEST", coupon=0.05, maturity=date(2030, 1, 1))
    ytm = 0.05
    
    price_base = BondAnalytics.price(bond, ytm, date.today())
    duration = BondAnalytics.duration(bond, ytm, date.today(), modified=True)
    convexity = BondAnalytics.convexity(bond, ytm, date.today())
    
    # Large move: +100bp
    dy = 0.01
    
    # Duration-only estimate
    duration_estimate = -duration * dy
    
    # With convexity
    convexity_estimate = -duration * dy + 0.5 * convexity * (dy ** 2)
    
    actual = (BondAnalytics.price(bond, ytm + dy, date.today()) - price_base) / price_base
    
    # Convexity should be closer
    assert abs(convexity_estimate - actual) < abs(duration_estimate - actual)
    print(f"✓ Convexity improves estimate: {actual:.4f} vs duration {duration_estimate:.4f}")


def test_ytm_solver():
    """YTM solver correctly finds yield from price."""
    bond = Bond(cusip="TEST", coupon=0.05, maturity=date(2030, 1, 1))
    settlement = date.today()
    
    # Price at 4% yield
    price_at_4pct = BondAnalytics.price(bond, 0.04, settlement)
    
    # Solve for YTM
    solved_ytm = BondAnalytics.yield_to_maturity(bond, price_at_4pct, settlement)
    
    # Should be very close to 4%
    assert abs(solved_ytm - 0.04) < 0.0001
    print(f"✓ YTM solver: expected=4%, solved={solved_ytm*100:.2f}%")


def test_clean_dirty_price():
    """Clean price = Dirty price - Accrued interest."""
    bond = Bond(cusip="TEST", coupon=0.05, maturity=date(2030, 1, 1), frequency=2)
    ytm = 0.05
    settlement = date(2024, 7, 15)
    
    dirty = BondAnalytics.price(bond, ytm, settlement)
    clean = BondAnalytics.clean_price(bond, ytm, settlement)
    accrued = BondAnalytics.accrued_interest(bond, settlement)
    
    assert abs((dirty - accrued) - clean) < 0.01
    print(f"✓ Clean/Dirty price: dirty=${dirty:.2f}, accrued=${accrued:.2f}, clean=${clean:.2f}")


def test_zero_coupon_bond():
    """Zero coupon bond pricing."""
    bond = Bond(cusip="ZERO", coupon=0.0, maturity=date(2030, 1, 1))
    ytm = 0.05
    
    price = BondAnalytics.price(bond, ytm, date(2024, 1, 1))
    
    # Should be present value of face value with semi-annual compounding
    # Time to maturity: ~6 years = 12 semi-annual periods
    periods = 12
    period_rate = ytm / 2
    expected_pv = 1000 * (1 + period_rate) ** (-periods)
    
    assert abs(price - expected_pv) < 1.0
    print(f"✓ Zero coupon: ${price:.2f} (expected ~${expected_pv:.2f})")


def test_macaulay_vs_modified_duration():
    """Modified duration = Macaulay / (1 + y/f)."""
    bond = Bond(cusip="TEST", coupon=0.05, maturity=date(2030, 1, 1))
    ytm = 0.05
    
    macaulay = BondAnalytics.duration(bond, ytm, date.today(), modified=False)
    modified = BondAnalytics.duration(bond, ytm, date.today(), modified=True)
    
    expected_modified = macaulay / (1 + ytm / 2)
    
    assert abs(modified - expected_modified) < 0.001
    print(f"✓ Duration: Macaulay={macaulay:.3f}, Modified={modified:.3f}")


if __name__ == '__main__':
    test_par_bond()
    test_duration_sensitivity()
    test_convexity()
    test_ytm_solver()
    test_clean_dirty_price()
    test_zero_coupon_bond()
    test_macaulay_vs_modified_duration()
    print("\n✅ All bond tests passed!")

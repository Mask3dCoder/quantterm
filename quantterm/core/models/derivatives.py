"""Derivatives models for QuantTerm."""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field, field_validator

from quantterm.core.enums import OptionStyle, OptionType


class OptionContract(BaseModel):
    """Option contract specification."""
    underlying: str
    expiration: datetime
    strike: float
    option_type: OptionType
    style: OptionStyle = OptionStyle.EUROPEAN
    multiplier: float = 100.0  # Standard equity option multiplier
    exchange: Optional[str] = None
    currency: str = "USD"

    @field_validator("strike")
    @classmethod
    def validate_strike(cls, v: float) -> float:
        """Validate strike price is positive."""
        if v <= 0:
            raise ValueError("Strike must be positive")
        return v

    @property
    def is_call(self) -> bool:
        """Is this a call option?"""
        return self.option_type == OptionType.CALL

    @property
    def is_put(self) -> bool:
        """Is this a put option?"""
        return self.option_type == OptionType.PUT

    @property
    def is_american(self) -> bool:
        """Is this an American-style option?"""
        return self.style == OptionStyle.AMERICAN

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        now = datetime.utcnow()
        delta = self.expiration - now
        return max(0, delta.days)

    @property
    def time_to_expiration(self) -> float:
        """Time to expiration in years."""
        return self.days_to_expiration / 365.0

    @property
    def symbol(self) -> str:
        """Generate option symbol (e.g., AAPL230616C00180000)."""
        exp_str = self.expiration.strftime("%y%m%d")
        type_char = "C" if self.is_call else "P"
        strike_str = f"{int(self.strike * 1000):08d}"
        return f"{self.underlying}{exp_str}{type_char}{strike_str}"

    @classmethod
    def from_symbol(cls, symbol: str) -> "OptionContract":
        """Parse option symbol to contract."""
        # AAPL230616C00180000
        # Underlying: AAPL (first 4 chars, may vary)
        # Expiration: 230616 (YYMMDD)
        # Type: C or P
        # Strike: 00180000 (8 digits, strike * 1000)

        # Find where the expiration starts (6 digits after underlying)
        if len(symbol) < 12:
            raise ValueError(f"Invalid option symbol: {symbol}")

        # Try to parse - underlying is typically 1-5 chars
        # Find expiration date (YYMMDD format)
        import re
        match = re.match(r"^([A-Z]{1,5})(\d{6})([CP])([\d]{8})$", symbol)
        if not match:
            raise ValueError(f"Invalid option symbol format: {symbol}")

        underlying, exp_str, type_char, strike_str = match.groups()
        expiration = datetime.strptime(exp_str, "%y%m%d")
        strike = int(strike_str) / 1000.0
        option_type = OptionType.CALL if type_char == "C" else OptionType.PUT

        return cls(
            underlying=underlying,
            expiration=expiration,
            strike=strike,
            option_type=option_type,
        )


class GreeksValues(BaseModel):
    """Option Greeks values."""
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0

    # Second order Greeks
    vanna: float = 0.0
    charm: float = 0.0
    vomma: float = 0.0
    veta: float = 0.0
    vera: float = 0.0

    # Third order Greeks
    speed: float = 0.0
    zomma: float = 0.0
    color: float = 0.0
    ultima: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary."""
        return self.model_dump()


class OptionQuote(BaseModel):
    """Option price quote with Greeks."""
    contract: OptionContract
    bid: float = 0.0
    ask: float = 0.0
    last: Optional[float] = None
    mark: Optional[float] = None  # Mid or last traded
    volume: int = 0
    open_interest: int = 0
    implied_volatility: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def mid(self) -> float:
        """Mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.mark or self.last or 0.0

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_pct(self) -> float:
        """Spread as percentage of mid."""
        if self.mid > 0:
            return self.spread / self.mid * 100
        return 0.0


class OptionChain(BaseModel):
    """Complete option chain for an underlying."""
    underlying: str
    expiration: datetime
    calls: list[OptionQuote] = Field(default_factory=list)
    puts: list[OptionQuote] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def expirations(self) -> list[datetime]:
        """List of available expirations."""
        return [self.expiration]

    def get_call(self, strike: float) -> Optional[OptionQuote]:
        """Get call option by strike."""
        for c in self.calls:
            if c.contract.strike == strike:
                return c
        return None

    def get_put(self, strike: float) -> Optional[OptionQuote]:
        """Get put option by strike."""
        for p in self.puts:
            if p.contract.strike == strike:
                return p
        return None

    @property
    def strikes(self) -> list[float]:
        """All available strikes."""
        return sorted(set([c.contract.strike for c in self.calls] +
                        [p.contract.strike for p in self.puts]))

    @property
    def atm_strike(self) -> Optional[float]:
        """At-the-money strike (closest to underlying price)."""
        if not self.calls:
            return None
        # Assuming underlying price is from first call's context
        # In practice, this would be passed in
        return None


class FutureContract(BaseModel):
    """Futures contract specification."""
    underlying: str
    expiration: datetime
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float = 0.01  # multiplier * tick_size
    exchange: Optional[str] = None
    currency: str = "USD"

    @property
    def symbol(self) -> str:
        """Generate futures symbol (e.g., ES2023-06)."""
        return f"{self.underlying}{self.expiration.strftime('%Y-%m')}"

    @property
    def days_to_expiration(self) -> int:
        """Days until expiration."""
        now = datetime.utcnow()
        delta = self.expiration - now
        return max(0, delta.days)


class FutureQuote(BaseModel):
    """Futures price quote."""
    contract: FutureContract
    bid: float = 0.0
    ask: float = 0.0
    last: Optional[float] = None
    previous_close: Optional[float] = None
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    volume: int = 0
    open_interest: int = 0
    settlement: Optional[float] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def mid(self) -> float:
        """Mid price."""
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2
        return self.last or self.settlement or 0.0


class VolatilitySurface(BaseModel):
    """Volatility surface (smile/skew)."""
    underlying: str
    reference_date: datetime
    strikes: list[float] = Field(default_factory=list)
    expirations: list[float] = Field(default_factory=list)  # in years
    volatilities: list[list[float]] = Field(default_factory=list)  # 2D grid

    def get_vol(self, strike: float, expiration: float) -> float:
        """Get interpolated volatility for given strike and expiration."""
        # Simple bilinear interpolation
        # In practice, would use more sophisticated methods (SVI, etc.)
        if not self.strikes or not self.expirations:
            return 0.20  # Default 20% vol

        # Find surrounding strikes
        min_strike = min(self.strikes)
        max_strike = max(self.strikes)

        if strike < min_strike:
            strike = min_strike
        elif strike > max_strike:
            strike = max_strike

        # Find surrounding expirations
        min_exp = min(self.expirations)
        max_exp = max(self.expirations)

        if expiration < min_exp:
            expiration = min_exp
        elif expiration > max_exp:
            expiration = max_exp

        # Return a simple interpolated vol (placeholder)
        # Would implement proper interpolation in production
        base_vol = 0.20  # Would be interpolated
        return base_vol

"""Market data models for QuantTerm."""
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator


class Quote(BaseModel):
    """Real-time quote data."""
    symbol: str
    bid: float
    ask: float
    bid_size: int = 0
    ask_size: int = 0
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @property
    def mid(self) -> float:
        """Mid price."""
        return (self.bid + self.ask) / 2

    @property
    def spread(self) -> float:
        """Bid-ask spread."""
        return self.ask - self.bid

    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points."""
        return (self.spread / self.mid) * 10000

    @field_validator("bid", "ask")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price is positive."""
        if v <= 0:
            raise ValueError("Price must be positive")
        return v


class Trade(BaseModel):
    """Trade/transaction data."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    side: Optional[str] = None  # "buy" or "sell" if available
    exchange: Optional[str] = None

    @field_validator("price")
    @classmethod
    def validate_price(cls, v: float) -> float:
        """Validate price is positive."""
        if v <= 0:
            raise ValueError("Price must be positive")
        return v


class OHLCV(BaseModel):
    """OHLCV bar data."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    vwap: Optional[float] = None
    number_of_trades: Optional[int] = None

    @field_validator("open", "high", "low", "close")
    @classmethod
    def validate_prices(cls, v: float) -> float:
        """Validate price is non-negative."""
        if v < 0:
            raise ValueError("Price cannot be negative")
        return v

    def is_valid(self) -> bool:
        """Check if OHLCV data is valid (sanity checks)."""
        # High must be >= all other prices
        if self.high < self.open or self.high < self.close:
            return False
        # Low must be <= all other prices
        if self.low > self.open or self.low > self.close:
            return False
        # All prices should be consistent
        if self.high < self.low:
            return False
        # Volume must be positive
        if self.volume < 0:
            return False
        return True

    @property
    def typical_price(self) -> float:
        """Typical price (HLC average)."""
        return (self.high + self.low + self.close) / 3

    @property
    def weighted_price(self) -> float:
        """Weighted typical price."""
        return (self.high + self.low + 2 * self.close) / 4

    @property
    def range(self) -> float:
        """High - Low range."""
        return self.high - self.low

    @property
    def body(self) -> float:
        """Absolute difference between open and close."""
        return abs(self.close - self.open)

    @property
    def upper_shadow(self) -> float:
        """Upper shadow (high - max(open, close))."""
        return self.high - max(self.open, self.close)

    @property
    def lower_shadow(self) -> float:
        """Lower shadow (min(open, close) - low)."""
        return min(self.open, self.close) - self.low

    @property
    def is_bullish(self) -> bool:
        """Is this a bullish candle?"""
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        """Is this a bearish candle?"""
        return self.close < self.open

    @property
    def is_doji(self) -> bool:
        """Is this a doji (minimal body)?"""
        return self.body <= self.range * 0.1


class BarSeries(BaseModel):
    """Series of OHLCV bars."""
    symbol: str
    bars: list[OHLCV]
    timeframe: str = "1d"

    @property
    def timestamps(self) -> list[datetime]:
        """Get all timestamps."""
        return [b.timestamp for b in self.bars]

    @property
    def opens(self) -> np.ndarray:
        """Get open prices as numpy array."""
        return np.array([b.open for b in self.bars])

    @property
    def highs(self) -> np.ndarray:
        """Get high prices as numpy array."""
        return np.array([b.high for b in self.bars])

    @property
    def lows(self) -> np.ndarray:
        """Get low prices as numpy array."""
        return np.array([b.low for b in self.bars])

    @property
    def closes(self) -> np.ndarray:
        """Get close prices as numpy array."""
        return np.array([b.close for b in self.bars])

    @property
    def volumes(self) -> np.ndarray:
        """Get volumes as numpy array."""
        return np.array([b.volume for b in self.bars])

    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        return pd.DataFrame([b.model_dump() for b in self.bars])


class TickData(BaseModel):
    """Tick-by-tick market data."""
    symbol: str
    trades: list[Trade] = Field(default_factory=list)
    quotes: list[Quote] = Field(default_factory=list)

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the tick data."""
        self.trades.append(trade)

    def add_quote(self, quote: Quote) -> None:
        """Add a quote to the tick data."""
        self.quotes.append(quote)

    @property
    def trade_count(self) -> int:
        """Number of trades."""
        return len(self.trades)

    @property
    def quote_count(self) -> int:
        """Number of quotes."""
        return len(self.quotes)


class MarketDepth(BaseModel):
    """Level 2 market depth data."""
    symbol: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    bids: list[tuple[float, int]] = Field(default_factory=list)  # (price, size)
    asks: list[tuple[float, int]] = Field(default_factory=list)  # (price, size)

    @property
    def best_bid(self) -> Optional[float]:
        """Best bid price."""
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Best ask price."""
        return self.asks[0][0] if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        """Mid price."""
        if self.best_bid and self.best_ask:
            return (self.best_bid + self.best_ask) / 2
        return None

    @property
    def spread(self) -> Optional[float]:
        """Bid-ask spread."""
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None

    @property
    def depth_bid(self) -> int:
        """Total bid size."""
        return sum(size for _, size in self.bids)

    @property
    def depth_ask(self) -> int:
        """Total ask size."""
        return sum(size for _, size in self.asks)

    @property
    def imbalance(self) -> float:
        """Order imbalance (-1 to 1)."""
        total = self.depth_bid + self.depth_ask
        if total == 0:
            return 0
        return (self.depth_bid - self.depth_ask) / total

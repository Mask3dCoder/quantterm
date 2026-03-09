"""Portfolio models for QuantTerm."""
from datetime import datetime
from typing import Optional

import numpy as np
from pydantic import BaseModel, Field, field_validator

from quantterm.core.enums import AssetClass, OrderSide


class Position(BaseModel):
    """Single position in a security."""
    symbol: str
    quantity: float  # Can be fractional for crypto
    avg_cost: float
    asset_class: AssetClass = AssetClass.EQUITY

    # Current market state
    current_price: Optional[float] = None
    market_value: Optional[float] = None
    unrealized_pnl: Optional[float] = None
    realized_pnl: float = 0.0
    weight: float = 0.0  # Portfolio weight

    # Position metadata
    open_date: Optional[datetime] = None
    exchange: Optional[str] = None

    @field_validator("quantity", "avg_cost")
    @classmethod
    def validate_positive(cls, v: float) -> float:
        """Validate quantity and cost are non-negative."""
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return self.quantity * self.avg_cost

    def update_price(self, price: float) -> None:
        """Update current price and recalculate metrics."""
        self.current_price = price
        self.market_value = self.quantity * price
        self.unrealized_pnl = self.market_value - self.cost_basis

    def close(self, quantity: Optional[float] = None, price: float = 0.0) -> float:
        """Close position (or portion) and return realized PnL."""
        close_qty = quantity if quantity else self.quantity
        if close_qty > self.quantity:
            raise ValueError("Cannot close more than current position")

        close_value = close_qty * price
        cost_closed = close_qty * self.avg_cost
        realized = close_value - cost_closed

        self.realized_pnl += realized
        self.quantity -= close_qty

        if self.quantity == 0:
            self.avg_cost = 0.0
            self.current_price = None
            self.market_value = None
            self.unrealized_pnl = None

        return realized


class OptionPosition(BaseModel):
    """Position in an option contract."""
    contract_id: str  # Option contract identifier
    symbol: str  # Underlying
    quantity: int  # Number of contracts
    avg_cost: float  # Per contract
    multiplier: float = 100.0

    current_price: Optional[float] = None
    mark: Optional[float] = None
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @property
    def cost_basis(self) -> float:
        """Total cost basis."""
        return abs(self.quantity) * self.avg_cost * self.multiplier

    @property
    def market_value(self) -> float:
        """Current market value."""
        if self.mark is None:
            return 0.0
        return abs(self.quantity) * self.mark * self.multiplier


class Holding(BaseModel):
    """Complete holding with all position details."""
    symbol: str
    quantity: float
    avg_cost: float
    current_price: float
    market_value: float
    cost_basis: float
    unrealized_pnl: float
    realized_pnl: float
    weight: float = 0.0  # Portfolio weight
    asset_class: AssetClass = AssetClass.EQUITY
    sector: Optional[str] = None
    industry: Optional[str] = None


class Portfolio(BaseModel):
    """Portfolio containing multiple positions."""
    name: str
    cash: float = 0.0
    positions: dict[str, Position] = Field(default_factory=dict)
    option_positions: dict[str, OptionPosition] = Field(default_factory=dict)
    currency: str = "USD"

    # Performance tracking
    starting_cash: float = 0.0
    total_commission: float = 0.0
    total_dividends: float = 0.0

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def positions_list(self) -> list[Position]:
        """Get positions as list."""
        return list(self.positions.values())

    @property
    def symbols(self) -> list[str]:
        """Get all symbols in portfolio."""
        return list(self.positions.keys())

    @property
    def total_market_value(self) -> float:
        """Total market value of all positions."""
        return sum(p.market_value or 0.0 for p in self.positions.values())

    @property
    def total_value(self) -> float:
        """Total portfolio value including cash."""
        return self.cash + self.total_market_value

    @property
    def total_cost_basis(self) -> float:
        """Total cost basis."""
        return sum(p.cost_basis for p in self.positions.values())

    @property
    def total_unrealized_pnl(self) -> float:
        """Total unrealized PnL."""
        return sum(p.unrealized_pnl or 0.0 for p in self.positions.values())

    @property
    def total_realized_pnl(self) -> float:
        """Total realized PnL."""
        return sum(p.realized_pnl for p in self.positions.values())

    @property
    def total_pnl(self) -> float:
        """Total PnL (realized + unrealized)."""
        return self.total_realized_pnl + self.total_unrealized_pnl

    @property
    def cash_weight(self) -> float:
        """Cash as percentage of total."""
        if self.total_value == 0:
            return 0.0
        return self.cash / self.total_value

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)

    def add_position(
        self,
        symbol: str,
        quantity: float,
        price: float,
        asset_class: AssetClass = AssetClass.EQUITY,
    ) -> Position:
        """Add or update a position."""
        if symbol in self.positions:
            pos = self.positions[symbol]
            # Update using average cost formula
            total_cost = pos.cost_basis + (quantity * price)
            new_qty = pos.quantity + quantity
            pos.avg_cost = total_cost / new_qty if new_qty > 0 else 0.0
            pos.quantity = new_qty
            pos.update_price(price)
        else:
            pos = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost=price,
                asset_class=asset_class,
                current_price=price,
                market_value=quantity * price,
                unrealized_pnl=0.0,
                open_date=datetime.utcnow(),
            )
            self.positions[symbol] = pos

        # Deduct cash for purchase
        self.cash -= quantity * price
        self.updated_at = datetime.utcnow()
        return pos

    def close_position(
        self, symbol: str, quantity: Optional[float] = None, price: float = 0.0
    ) -> float:
        """Close position (or portion) and return realized PnL."""
        if symbol not in self.positions:
            raise ValueError(f"No position for {symbol}")

        pos = self.positions[symbol]
        close_qty = quantity if quantity else pos.quantity

        # Add cash from sale
        self.cash += close_qty * price

        # Calculate realized PnL
        realized = pos.close(close_qty, price)

        if pos.quantity == 0:
            del self.positions[symbol]

        self.updated_at = datetime.utcnow()
        return realized

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update all position prices."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].update_price(price)

    def calculate_weights(self) -> None:
        """Calculate portfolio weights."""
        total_mv = self.total_market_value
        if total_mv == 0:
            return

        for pos in self.positions.values():
            if pos.market_value:
                pos.weight = pos.market_value / total_mv

    def get_holdings(self) -> list[Holding]:
        """Get complete holdings with all details."""
        holdings = []
        for pos in self.positions.values():
            if pos.market_value:
                holdings.append(
                    Holding(
                        symbol=pos.symbol,
                        quantity=pos.quantity,
                        avg_cost=pos.avg_cost,
                        current_price=pos.current_price or 0.0,
                        market_value=pos.market_value or 0.0,
                        cost_basis=pos.cost_basis,
                        unrealized_pnl=pos.unrealized_pnl or 0.0,
                        realized_pnl=pos.realized_pnl,
                        weight=pos.weight,
                        asset_class=pos.asset_class,
                    )
                )
        return holdings


class PortfolioSnapshot(BaseModel):
    """Point-in-time portfolio snapshot."""
    timestamp: datetime
    total_value: float
    cash: float
    market_value: float
    unrealized_pnl: float
    realized_pnl: float
    return_pct: float = 0.0  # Since inception
    positions_count: int = 0


class Trade(BaseModel):
    """Trade execution record."""
    timestamp: datetime
    symbol: str
    side: OrderSide
    quantity: float
    price: float
    commission: float = 0.0
    fees: float = 0.0
    order_id: Optional[str] = None

    @property
    def total_cost(self) -> float:
        """Total cost including commission and fees."""
        return self.quantity * self.price + self.commission + self.fees


class TransactionHistory(BaseModel):
    """Complete transaction history."""
    portfolio_name: str
    trades: list[Trade] = Field(default_factory=list)
    deposits: list[float] = Field(default_factory=list)
    withdrawals: list[float] = Field(default_factory=list)

    def add_trade(self, trade: Trade) -> None:
        """Add a trade to history."""
        self.trades.append(trade)

    @property
    def total_deposits(self) -> float:
        """Total deposits."""
        return sum(self.deposits)

    @property
    def total_withdrawals(self) -> float:
        """Total withdrawals."""
        return sum(self.withdrawals)

    def get_trades_by_symbol(self, symbol: str) -> list[Trade]:
        """Get all trades for a symbol."""
        return [t for t in self.trades if t.symbol == symbol]

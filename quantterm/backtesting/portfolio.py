"""Portfolio tracker for the QuantTerm backtesting engine.

Simple portfolio implementation that tracks:
- Cash balance
- Positions (symbol -> quantity)
- Cost basis per position
- Realized P&L from closed trades
- Unrealized P&L (mark-to-market)
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from quantterm.backtesting.events import FillEvent, OrderEvent


class Portfolio:
    """Simple portfolio tracker for backtesting.
    
    Tracks positions, cash, and calculates realized/unrealized P&L.
    No margin or short selling restrictions.
    """
    
    def __init__(self, initial_cash: float):
        """Initialize portfolio with starting cash.
        
        Args:
            initial_cash: Starting cash amount.
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, int] = {}  # symbol -> quantity
        self.cost_basis: Dict[str, float] = {}  # symbol -> total cost
        self.realized_pnl: float = 0.0
        self.trades: list[FillEvent] = []
    
    def process_fill(self, fill: FillEvent) -> None:
        """Update portfolio after a fill.
        
        Args:
            fill: The fill event from trade execution.
        """
        symbol = fill.symbol
        quantity = fill.quantity
        trade_value = fill.price * abs(quantity)
        commission = fill.commission
        
        if quantity > 0:  # BUY
            # Update cost basis
            current_cost = self.cost_basis.get(symbol, 0.0)
            self.cost_basis[symbol] = current_cost + trade_value + commission
            self.cash -= (trade_value + commission)
        else:  # SELL
            # Calculate realized P&L for this sale
            current_position = self.positions.get(symbol, 0)
            if current_position != 0:
                # Average cost per share
                avg_cost = self.cost_basis.get(symbol, 0.0) / abs(current_position)
                # P&L = (sell price - avg cost) * shares sold
                pnl = (fill.price - avg_cost) * abs(quantity)
                self.realized_pnl += pnl
            
            # Update cost basis (reduce proportionally)
            if current_position != 0:
                sold_ratio = abs(quantity) / current_position
                self.cost_basis[symbol] *= (1 - sold_ratio)
                if abs(current_position + quantity) < 0.01:
                    del self.cost_basis[symbol]
            
            self.cash += trade_value - commission
        
        # Update position
        current_position = self.positions.get(symbol, 0)
        self.positions[symbol] = current_position + quantity
        
        # Remove zero positions
        if self.positions[symbol] == 0:
            del self.positions[symbol]
        
        # Track trade
        self.trades.append(fill)
    
    def get_position(self, symbol: str) -> int:
        """Get current position quantity for a symbol.
        
        Args:
            symbol: The ticker symbol.
            
        Returns:
            Position quantity (positive for long, negative for short).
        """
        return self.positions.get(symbol, 0)
    
    def get_unrealized_pnl(self, current_prices: Dict[str, float]) -> float:
        """Calculate unrealized P&L based on current prices.
        
        Args:
            current_prices: Dictionary of symbol -> current price.
            
        Returns:
            Unrealized P&L from open positions.
        """
        unrealized = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices and position != 0:
                current_price = current_prices[symbol]
                avg_cost = self.cost_basis.get(symbol, 0.0) / abs(position)
                unrealized += (current_price - avg_cost) * position
        
        return unrealized
    
    def get_total_value(self, current_prices: Dict[str, float]) -> float:
        """Calculate total portfolio value including cash and positions.
        
        Args:
            current_prices: Dictionary of symbol -> current price.
            
        Returns:
            Total portfolio value (cash + position value at current prices).
        """
        position_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                position_value += position * current_prices[symbol]
        
        return self.cash + position_value
    
    def get_realized_pnl(self) -> float:
        """Get realized P&L from closed trades.
        
        Returns:
            Realized P&L amount.
        """
        return self.realized_pnl


class MarginPortfolio:
    """Portfolio with margin and short selling support.
    
    Critical calculations:
    - Gross exposure (sum of absolute positions)
    - Net exposure (long - short)  
    - Margin requirement (Reg T: 50% for stocks)
    - Maintenance margin (25% minimum)
    - Short borrow costs (hard-to-borrow fees)
    """
    
    def __init__(
        self,
        initial_cash: float,
        margin_requirement: float = 0.50,  # Reg T: 50%
        short_borrow_cost: float = 0.03,     # 3% annual
    ):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.margin_requirement = margin_requirement
        self.short_borrow_cost = short_borrow_cost
        
        # Track positions separately for long/short
        self.long_positions: dict[str, int] = {}   # symbol -> quantity
        self.short_positions: dict[str, int] = {}  # symbol -> quantity
        
        # Cost basis tracking
        self.cost_basis: dict[str, float] = {}  # symbol -> avg cost
        
        self.realized_pnl: float = 0.0
        self.short_borrow_accrued: float = 0.0
        
        # Daily borrow cost tracking
        self.daily_borrow_costs: list[tuple[datetime, float]] = []
    
    def total_long_value(self, prices: dict[str, float]) -> float:
        """Total value of long positions."""
        return sum(
            self.long_positions.get(s, 0) * prices.get(s, 0)
            for s in self.long_positions
        )
    
    def total_short_value(self, prices: dict[str, float]) -> float:
        """Total value of short positions."""
        return sum(
            abs(self.short_positions.get(s, 0)) * prices.get(s, 0)
            for s in self.short_positions
        )
    
    def gross_exposure(self, prices: dict[str, float]) -> float:
        """Total absolute exposure (long + short)."""
        return self.total_long_value(prices) + self.total_short_value(prices)
    
    def net_exposure(self, prices: dict[str, float]) -> float:
        """Net exposure (long - short)."""
        return self.total_long_value(prices) - self.total_short_value(prices)
    
    def buying_power(self, prices: dict[str, float]) -> float:
        """Calculate available buying power.
        With 50% margin, you can buy 2x the cash balance.
        """
        # Total margin used
        long_margin = self.total_long_value(prices) * self.margin_requirement
        short_margin = self.total_short_value(prices) * self.margin_requirement
        
        # Cash + margin capacity
        total_equity = self.cash + self.get_total_equity(prices)
        margin_used = long_margin + short_margin
        
        # Buying power = total equity - margin used (simplified)
        return max(0, total_equity - margin_used)
    
    def get_total_equity(self, prices: dict[str, float]) -> float:
        """Total equity = cash + long positions + short P&L"""
        long_val = self.total_long_value(prices)
        
        # Short P&L: (entry price - current price) * shares
        short_pnl = 0.0
        for symbol, shares in self.short_positions.items():
            if symbol in self.cost_basis and symbol in prices:
                short_pnl += (self.cost_basis[symbol] - prices[symbol]) * abs(shares)
        
        return self.cash + long_val + short_pnl
    
    def get_total_value(self, prices: dict[str, float]) -> float:
        """Get total portfolio value including all positions."""
        return self.get_total_equity(prices)
    
    def get_position(self, symbol: str) -> int:
        """Get net position (long - short)."""
        long_shares = self.long_positions.get(symbol, 0)
        short_shares = self.short_positions.get(symbol, 0)
        return long_shares - short_shares
    
    def can_execute(self, order: OrderEvent, prices: dict[str, float]) -> bool:
        """Check if order would violate margin constraints."""
        if order.quantity > 0:  # Buying long
            # Would this exceed margin capacity?
            required_margin = order.quantity * prices[order.symbol] * self.margin_requirement
            return required_margin <= self.buying_power(prices)
        else:  # Short sale
            # Short sales also require margin
            required_margin = abs(order.quantity) * prices[order.symbol] * self.margin_requirement
            return required_margin <= self.buying_power(prices)
    
    def process_fill(self, fill: FillEvent):
        """Process a fill event, updating positions."""
        symbol = fill.symbol
        quantity = fill.quantity  # positive = bought, negative = sold
        price = fill.price
        commission = fill.commission
        
        if quantity > 0:  # Bought (going long or covering short)
            if symbol in self.short_positions and self.short_positions[symbol] < 0:
                # Covering a short position
                shares_to_cover = min(quantity, abs(self.short_positions[symbol]))
                # Calculate P&L
                entry_price = self.cost_basis.get(symbol, price)
                pnl = (entry_price - price) * shares_to_cover
                self.realized_pnl += pnl
                
                # Update short position
                self.short_positions[symbol] += shares_to_cover
                if self.short_positions[symbol] == 0:
                    del self.short_positions[symbol]
                
                # Remaining shares go long
                remaining = quantity - shares_to_cover
                if remaining > 0:
                    self._add_long(symbol, remaining, price)
            else:
                # Adding to long position
                self._add_long(symbol, quantity, price)
        
        elif quantity < 0:  # Sold (going short or selling long)
            if symbol in self.long_positions and self.long_positions[symbol] > 0:
                # Selling a long position
                shares_to_sell = min(abs(quantity), self.long_positions[symbol])
                entry_price = self.cost_basis.get(symbol, price)
                pnl = (price - entry_price) * shares_to_sell
                self.realized_pnl += pnl
                
                # Update long position
                self.long_positions[symbol] -= shares_to_sell
                if self.long_positions[symbol] == 0:
                    del self.long_positions[symbol]
                
                # Remaining shares go short
                remaining = abs(quantity) - shares_to_sell
                if remaining > 0:
                    self._add_short(symbol, remaining, price)
            else:
                # Opening short position
                self._add_short(symbol, abs(quantity), price)
        
        # Deduct commission
        self.cash -= commission
    
    def _add_long(self, symbol: str, shares: int, price: float):
        """Add to long position."""
        current = self.long_positions.get(symbol, 0)
        current_cost = self.cost_basis.get(symbol, 0) * current
        
        new_cost = current_cost + shares * price
        self.long_positions[symbol] = current + shares
        self.cost_basis[symbol] = new_cost / self.long_positions[symbol]
        
        # Deduct cash
        self.cash -= shares * price
    
    def _add_short(self, symbol: str, shares: int, price: float):
        """Add to short position (receive cash)."""
        current = self.short_positions.get(symbol, 0)
        current_cost = self.cost_basis.get(symbol, 0) * abs(current)
        
        new_cost = current_cost + shares * price
        self.short_positions[symbol] = current - shares  # negative
        self.cost_basis[symbol] = new_cost / abs(self.short_positions[symbol])
        
        # Receive cash from short sale
        self.cash += shares * price
    
    def update_short_borrow_costs(self, date: datetime, prices: dict[str, float]):
        """Daily accrual of short borrow fees.
        Called at end of each trading day.
        """
        daily_cost = 0.0
        for symbol, shares in self.short_positions.items():
            if symbol in prices:
                position_value = abs(shares) * prices[symbol]
                # Annual rate / 252 trading days
                daily_cost += position_value * (self.short_borrow_cost / 252)
        
        self.short_borrow_accrued += daily_cost
        self.cash -= daily_cost
        self.daily_borrow_costs.append((date, daily_cost))
    
    def get_unrealized_pnl(self, prices: dict[str, float]) -> float:
        """Calculate unrealized P&L."""
        unrealized = 0.0
        
        # Long positions
        for symbol, shares in self.long_positions.items():
            if symbol in prices and symbol in self.cost_basis:
                unrealized += (prices[symbol] - self.cost_basis[symbol]) * shares
        
        # Short positions
        for symbol, shares in self.short_positions.items():
            if symbol in prices and symbol in self.cost_basis:
                unrealized += (self.cost_basis[symbol] - prices[symbol]) * abs(shares)
        
        return unrealized

"""
Paper trading engine - simulates live trading with real market data.

Key features:
- Realistic fill simulation using bid/ask
- Slippage modeling
- Commission tracking
- Position management
- P&L calculation (realized + unrealized)
"""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional
import pandas as pd


@dataclass
class PaperPosition:
    """Current position in a symbol."""
    symbol: str
    quantity: float  # positive = long, negative = short
    avg_entry: float
    market_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        return self.quantity * self.market_price
    
    @property
    def cost_basis(self) -> float:
        return abs(self.quantity) * self.avg_entry
    
    @property
    def unrealized_pnl(self) -> float:
        if self.quantity > 0:
            return (self.market_price - self.avg_entry) * self.quantity
        else:
            return (self.avg_entry - self.market_price) * abs(self.quantity)


@dataclass
class PaperTrade:
    """Record of a paper trade."""
    timestamp: datetime
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    price: float
    commission: float
    pnl: float = 0.0  # Realized P&L (for closes)


class PaperTradingEngine:
    """
    Paper trading engine with realistic execution.
    
    Simulates:
    - Bid/ask spreads
    - Slippage (worse fill than mid)
    - Commission
    - Short borrow costs
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.0,
        slippage_bps: float = 5.0,  # 5 bps = 0.05%
        short_borrow_rate: float = 0.03  # 3% annual
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.slippage_bps = slippage_bps
        self.short_borrow_rate = short_borrow_rate
        
        self.positions: Dict[str, PaperPosition] = {}
        self.trades: list[PaperTrade] = []
        self.equity_curve: list[dict] = []
        
        # Track short borrow costs
        self.short_borrow_accrued: float = 0.0
    
    def get_total_equity(self, current_prices: Dict[str, float]) -> float:
        """Calculate total account equity."""
        # Update position prices
        for symbol, pos in self.positions.items():
            if symbol in current_prices:
                pos.market_price = current_prices[symbol]
        
        # Total equity = cash + position market values
        position_value = sum(pos.market_value for pos in self.positions.values())
        
        return self.cash + position_value
    
    def get_position(self, symbol: str) -> float:
        """Get current position quantity."""
        if symbol in self.positions:
            return self.positions[symbol].quantity
        return 0.0
    
    def execute_market_order(
        self,
        symbol: str,
        side: str,  # 'buy' or 'sell'
        quantity: float,
        current_price: float,
        bid: float = None,
        ask: float = None
    ) -> PaperTrade:
        """
        Execute market order with realistic slippage.
        
        Args:
            symbol: Ticker symbol
            side: 'buy' or 'sell'
            quantity: Number of shares
            current_price: Current market price
            bid: Current bid (optional)
            ask: Current ask (optional)
            
        Returns:
            PaperTrade with execution details
        """
        # Determine fill price with slippage
        # Slippage is applied to the total cost, not per-share
        slippage_multiplier = 1 + (self.slippage_bps / 10000)
        
        if side == 'buy':
            if ask:
                # Fill at ask + slippage
                fill_price = ask * slippage_multiplier
            else:
                fill_price = current_price * slippage_multiplier
        else:  # sell
            if bid:
                fill_price = bid / slippage_multiplier  # Slippage reduces sell price
            else:
                fill_price = current_price / slippage_multiplier
        
        # Calculate costs - slippage is already in the price, just add commission per share
        cost = quantity * fill_price
        commission_per_share = self.commission
        commission = quantity * commission_per_share
        
        # Execute
        if side == 'buy':
            if cost + commission > self.cash:
                raise ValueError(f"Insufficient cash: need ${cost + commission:.2f}, have ${self.cash:.2f}")
            
            self.cash -= (cost + commission)
            
            if symbol in self.positions:
                # Add to existing position
                pos = self.positions[symbol]
                if pos.quantity > 0:
                    # Adding to long
                    total_cost = pos.quantity * pos.avg_entry + cost
                    pos.quantity += quantity
                    pos.avg_entry = total_cost / pos.quantity
                else:
                    # Covering short first
                    shares_to_cover = min(quantity, abs(pos.quantity))
                    pnl = (pos.avg_entry - fill_price) * shares_to_cover
                    pos.quantity += shares_to_cover
                    
                    remaining = quantity - shares_to_cover
                    if remaining > 0:
                        # Add to long
                        total_cost = pos.quantity * pos.avg_entry + remaining * fill_price
                        pos.quantity += remaining
                        pos.avg_entry = total_cost / pos.quantity
            else:
                # New long position
                self.positions[symbol] = PaperPosition(
                    symbol=symbol,
                    quantity=quantity,
                    avg_entry=fill_price,
                    market_price=fill_price
                )
        else:  # sell
            if symbol not in self.positions or self.positions[symbol].quantity < quantity:
                raise ValueError(f"Insufficient shares to sell: have {self.get_position(symbol)}, need {quantity}")
            
            self.cash += (cost - commission)
            
            pos = self.positions[symbol]
            if pos.quantity > 0:
                # Selling long
                shares_to_sell = min(quantity, pos.quantity)
                pnl = (fill_price - pos.avg_entry) * shares_to_sell
                pos.quantity -= shares_to_sell
                
                remaining = quantity - shares_to_sell
                if remaining > 0:
                    # Open short
                    self.positions[symbol] = PaperPosition(
                        symbol=symbol,
                        quantity=-remaining,
                        avg_entry=fill_price,
                        market_price=fill_price
                    )
            else:
                # Adding to short
                total_cost = abs(pos.quantity) * pos.avg_entry + quantity * fill_price
                pos.quantity -= quantity
                pos.avg_entry = total_cost / abs(pos.quantity)
            
            # Remove if closed
            if symbol in self.positions and abs(self.positions[symbol].quantity) < 0.01:
                del self.positions[symbol]
        
        # Record trade
        trade = PaperTrade(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=fill_price,
            commission=commission
        )
        self.trades.append(trade)
        
        return trade
    
    def update_market_prices(self, prices: Dict[str, float]):
        """Update position marks to market."""
        for symbol, price in prices.items():
            if symbol in self.positions:
                self.positions[symbol].market_price = price
    
    def update_short_borrow_costs(self, prices: Dict[str, float]):
        """Accrue daily short borrow costs."""
        daily_rate = self.short_borrow_rate / 252
        
        for symbol, pos in self.positions.items():
            if pos.quantity < 0 and symbol in prices:
                position_value = abs(pos.quantity) * prices[symbol]
                daily_cost = position_value * daily_rate
                self.short_borrow_accrued += daily_cost
                self.cash -= daily_cost
    
    def flatten(self, current_prices: Dict[str, float]):
        """Close all positions at current prices."""
        symbols = list(self.positions.keys())
        
        for symbol in symbols:
            pos = self.positions[symbol]
            if pos.quantity > 0:
                self.execute_market_order(symbol, 'sell', pos.quantity, current_prices.get(symbol, pos.market_price))
            elif pos.quantity < 0:
                self.execute_market_order(symbol, 'buy', abs(pos.quantity), current_prices.get(symbol, pos.market_price))
    
    def get_summary(self, current_prices: Dict[str, float]) -> dict:
        """Get trading summary."""
        self.update_market_prices(current_prices)
        
        total_equity = self.get_total_equity(current_prices)
        total_pnl = total_equity - self.initial_capital
        pnl_pct = (total_pnl / self.initial_capital) * 100
        
        return {
            'cash': self.cash,
            'total_equity': total_equity,
            'total_pnl': total_pnl,
            'pnl_pct': pnl_pct,
            'positions': len(self.positions),
            'trades': len(self.trades),
            'short_borrow_costs': self.short_borrow_accrued
        }

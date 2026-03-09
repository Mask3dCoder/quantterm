"""Strategy base classes for QuantTerm backtesting engine.

Provides abstract base class for trading strategies and example implementations.
Supports both single and multi-symbol strategies.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List

import pandas as pd

from quantterm.backtesting.events import BarEvent, OrderEvent, FillEvent


class Strategy(ABC):
    """Abstract base class for trading strategies.
    
    All strategies must implement the on_bar method to generate
    trading signals from market data. For multi-symbol strategies,
    implement on_bar_multi instead.
    """
    
    def __init__(
        self, 
        name: str, 
        portfolio, 
        data_handler,
        symbols: Optional[List[str]] = None,
        target_weights: Optional[Dict[str, float]] = None
    ):
        """Initialize strategy.
        
        Args:
            name: Strategy name for identification.
            portfolio: Portfolio instance for position tracking.
            data_handler: DataHandler instance for market data access.
            symbols: List of symbols (for multi-symbol strategies).
            target_weights: Dict of symbol -> target weight.
        """
        self.name = name
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.symbols = symbols or []
        self.target_weights = target_weights or {}
    
    @abstractmethod
    def on_bar(self, bar: BarEvent) -> Optional[OrderEvent]:
        """Called on each bar. Return an OrderEvent to trade, or None.
        
        Args:
            bar: Current bar event with OHLCV data.
            
        Returns:
            OrderEvent if want to trade, None otherwise.
        """
        pass
    
    def on_bar_multi(
        self, 
        bars: Dict[str, pd.Series], 
        date: pd.Timestamp
    ) -> List[Optional[OrderEvent]]:
        """Multi-symbol bar handler.
        
        Override this method for multi-symbol strategies.
        Default implementation returns empty list.
        
        Args:
            bars: Dict of symbol -> bar data (Open, High, Low, Close, Volume).
            date: Current date.
            
        Returns:
            List of OrderEvents to execute.
        """
        return []
    
    def on_fill(self, fill: FillEvent):
        """Called when an order is filled.
        
        Args:
            fill: FillEvent with execution details.
        """
        pass


class BuyAndHoldStrategy(Strategy):
    """Simple buy-and-hold strategy for testing.
    
    Buys shares on the first bar and holds until the end.
    Uses 95% of available cash.
    """
    
    def on_bar(self, bar: BarEvent) -> Optional[OrderEvent]:
        """Execute buy-and-hold logic.
        
        Args:
            bar: Current bar event.
            
        Returns:
            OrderEvent to buy on first bar, None otherwise.
        """
        # Buy on first bar only
        if self.portfolio.get_position(bar.symbol) == 0:
            # Use 95% of cash to buy shares
            shares = int((self.portfolio.cash * 0.95) / bar.close)
            if shares > 0:
                return OrderEvent(
                    timestamp=bar.timestamp,
                    symbol=bar.symbol,
                    quantity=shares,  # positive = BUY
                    order_type="market"
                )
        return None

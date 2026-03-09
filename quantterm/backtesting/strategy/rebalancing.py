"""Rebalancing strategy for QuantTerm backtesting engine.

Equal-weight portfolio rebalanced at fixed frequency with transaction cost tracking.
"""

from typing import Optional, Dict, List
import pandas as pd
from datetime import timedelta

from quantterm.backtesting.strategy.base import Strategy
from quantterm.backtesting.events import OrderEvent


class RebalancingStrategy(Strategy):
    """Equal-weight portfolio rebalanced at fixed frequency.
    
    Features:
    - Target allocation weights
    - Configurable rebalancing frequency (daily, weekly, monthly)
    - Transaction cost tracking
    - Partial shares handling via integer share calculation
    """
    
    def __init__(
        self,
        name: str,
        portfolio,
        data_handler,
        symbols: List[str],
        target_weights: Dict[str, float],
        rebalance_freq: str = 'M',  # 'D', 'W', 'M'
    ):
        """Initialize rebalancing strategy.
        
        Args:
            name: Strategy name for identification.
            portfolio: Portfolio instance for position tracking.
            data_handler: DataHandler instance for market data access.
            symbols: List of symbols to trade.
            target_weights: Dict of symbol -> target weight.
            rebalance_freq: Rebalancing frequency ('D', 'W', 'M').
        """
        super().__init__(name, portfolio, data_handler, symbols, target_weights)
        self.rebalance_freq = rebalance_freq
        self.last_rebalance: Optional[pd.Timestamp] = None
        self.next_rebalance: Optional[pd.Timestamp] = None
        
        # Normalize weights to sum to 1.0
        total = sum(target_weights.values())
        if total > 0:
            self.target_weights = {k: v/total for k, v in target_weights.items()}
        else:
            self.target_weights = {k: 1.0/len(symbols) for k in symbols}
    
    def _should_rebalance(self, date: pd.Timestamp) -> bool:
        """Check if we should rebalance today.
        
        Args:
            date: Current date to check.
            
        Returns:
            True if rebalancing should occur.
        """
        if self.next_rebalance is None:
            # First time - set next rebalance date
            self.next_rebalance = self._get_next_rebalance_date(date)
            return True
        
        if date >= self.next_rebalance:
            self.last_rebalance = self.next_rebalance
            self.next_rebalance = self._get_next_rebalance_date(date)
            return True
        
        return False
    
    def _get_next_rebalance_date(self, date: pd.Timestamp) -> pd.Timestamp:
        """Calculate next rebalance date based on frequency.
        
        Args:
            date: Current date.
            
        Returns:
            Next rebalance date.
        """
        if self.rebalance_freq == 'D':
            return date + timedelta(days=1)
        elif self.rebalance_freq == 'W':
            # Next Monday
            days_ahead = 7 - date.dayofweek
            return (date + timedelta(days=days_ahead)).normalize()
        else:  # Monthly
            # Next month first day
            if date.month == 12:
                return pd.Timestamp(year=date.year + 1, month=1, day=1)
            else:
                return pd.Timestamp(year=date.year, month=date.month + 1, day=1)
    
    def on_bar_multi(
        self,
        bars: Dict[str, pd.Series],
        date: pd.Timestamp
    ) -> List[OrderEvent]:
        """Multi-symbol bar handler with rebalancing.
        
        Args:
            bars: Dict of symbol -> bar data (with 'Close' key).
            date: Current date.
            
        Returns:
            List of OrderEvents to execute.
        """
        orders: List[OrderEvent] = []
        
        # Check if rebalance needed
        if not self._should_rebalance(date):
            return orders
        
        # Calculate current prices and total portfolio value
        prices = {}
        for symbol in self.symbols:
            if symbol in bars:
                prices[symbol] = bars[symbol]['Close']
        
        if not prices:
            return orders
        
        total_value = self.portfolio.get_total_value(prices)
        
        if total_value <= 0:
            return orders
        
        # Calculate target positions for each symbol
        for symbol in self.symbols:
            if symbol not in prices:
                continue
            
            price = prices[symbol]
            if price <= 0:
                continue
                
            target_weight = self.target_weights.get(symbol, 0)
            target_value = total_value * target_weight
            target_shares = int(target_value / price)
            
            # Get current position
            current_shares = self.portfolio.get_position(symbol)
            
            # Calculate required change
            diff = target_shares - current_shares
            
            # Only trade if meaningful change (>1 share and >1% of position)
            if abs(diff) >= 1 and abs(diff) / max(abs(current_shares), 1) > 0.01:
                orders.append(OrderEvent(
                    timestamp=pd.Timestamp(date),
                    symbol=symbol,
                    quantity=diff,
                    order_type="market"
                ))
        
        return orders
    
    def on_bar(self, bar) -> List[OrderEvent]:
        """Single-symbol bar handler (not used for multi-symbol strategies).
        
        Args:
            bar: Current bar event.
            
        Returns:
            Empty list (multi-symbol rebalancing handled in on_bar_multi).
        """
        return []
    
    def on_fill(self, fill):
        """Called when an order is filled.
        
        Args:
            fill: FillEvent with execution details.
        """
        pass

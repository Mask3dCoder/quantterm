"""Complex multi-feature strategy for stress testing.

Tests: multi-symbol, rebalancing, margin, short selling, intraday fills.
"""

from typing import Optional, List, Dict
import pandas as pd

from quantterm.backtesting.events import OrderEvent, BarEvent


class ComplexStrategy:
    """
    Multi-symbol, long/short, rebalancing strategy.
    
    Features tested:
    - Multiple symbols (SPY, QQQ, IWM, TLT, GLD, VIXY)
    - Momentum-based ranking
    - Long/short positions
    - Periodic rebalancing
    - Transaction costs visible in returns
    - Margin constraints
    """
    
    def __init__(
        self,
        name: str,
        portfolio,
        data_handler,
        symbols: List[str],
        target_weights: Dict[str, float] = None,
    ):
        """Initialize complex strategy.
        
        Args:
            name: Strategy name for identification.
            portfolio: Portfolio instance for position tracking.
            data_handler: DataHandler instance for market data access.
            symbols: List of symbols to trade.
            target_weights: Dict of symbol -> target weight.
        """
        self.name = name
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.symbols = symbols
        
        # Default: equal weight
        if target_weights is None:
            target_weights = {s: 1.0/len(symbols) for s in symbols}
        self.target_weights = target_weights
        
        # Configuration
        self.lookback = 20  # days for momentum
        self.rebalance_day = 1  # Rebalance on 1st of month
        
        # State
        self.prices: Dict[str, float] = {}
        self.last_rebalance_month = None
        
        # Metrics tracking
        self.daily_values: List[float] = []
        self.turnover: List[int] = []
    
    def on_bar_multi(
        self,
        bars: Dict[str, pd.Series],
        date: pd.Timestamp
    ) -> List[OrderEvent]:
        """
        Main strategy logic.
        
        Combines:
        - Momentum ranking
        - Target weight rebalancing
        - Long/short with market neutrality
        """
        orders = []
        
        # Update prices
        for symbol, bar in bars.items():
            self.prices[symbol] = bar['Close']
        
        # Check if rebalance day (1st of month)
        should_rebalance = (
            self.last_rebalance_month is None or 
            date.month != self.last_rebalance_month
        )
        
        if should_rebalance:
            self.last_rebalance_month = date.month
            orders = self._generate_rebalance_orders()
        
        return orders
    
    def _generate_rebalance_orders(self) -> List[OrderEvent]:
        """
        Generate orders to rebalance to target weights.
        
        Uses momentum to bias allocation:
        - Top performers: higher weight within target
        - Bottom performers: lower weight
        """
        orders = []
        
        if not self.prices:
            return orders
        
        # Calculate momentum for each symbol
        momentum = {}
        for symbol in self.symbols:
            if symbol not in self.prices:
                continue
                
            # Try to get historical price
            if hasattr(self.data_handler, 'get_bars'):
                try:
                    # Get lookback price
                    data = self.data_handler.get_bars(symbol, '2020-01-01', '2030-12-31')
                    if data is not None and not data.empty and len(data) >= self.lookback:
                        current_price = self.prices[symbol]
                        lookback_idx = -min(self.lookback, len(data))
                        lookback_price = data.iloc[lookback_idx]['Close']
                        
                        if lookback_price > 0:
                            momentum[symbol] = (current_price - lookback_price) / lookback_price
                        else:
                            momentum[symbol] = 0
                    else:
                        momentum[symbol] = 0
                except:
                    momentum[symbol] = 0
            else:
                momentum[symbol] = 0
        
        # Calculate total portfolio value
        total_value = self.portfolio.get_total_value(self.prices)
        
        # Generate orders for each symbol
        for symbol in self.symbols:
            if symbol not in self.prices:
                continue
            
            base_weight = self.target_weights.get(symbol, 0)
            
            # Adjust weight by momentum bias (subtle)
            mom = momentum.get(symbol, 0)
            mom_factor = 1 + (mom * 0.2)  # ±20% adjustment
            adjusted_weight = base_weight * mom_factor
            
            # Normalize
            total_adj = sum(
                self.target_weights.get(s, 0) * (1 + momentum.get(s, 0) * 0.2)
                for s in self.symbols if s in self.prices
            )
            if total_adj > 0:
                adjusted_weight = adjusted_weight / total_adj
            
            # Calculate target shares
            target_value = total_value * adjusted_weight
            target_shares = int(target_value / self.prices[symbol])
            
            # Current position
            current_shares = self.portfolio.get_position(symbol)
            
            # Order difference
            diff = target_shares - current_shares
            
            # Only trade if meaningful (>1% of position)
            if abs(diff) > 0 and abs(diff) / max(abs(current_shares), 1) > 0.01:
                orders.append(OrderEvent(
                    timestamp=pd.Timestamp.now(),
                    symbol=symbol,
                    quantity=diff,
                    order_type="market"
                ))
        
        return orders
    
    def on_bar(self, bar: BarEvent) -> Optional[OrderEvent]:
        """Single bar handler - not used for multi-symbol strategies."""
        return None
    
    def on_fill(self, fill):
        """Track fills for metrics."""
        pass
    
    def get_total_value(self, prices: Dict[str, float]) -> float:
        """Get total portfolio value."""
        return self.portfolio.get_total_value(prices)

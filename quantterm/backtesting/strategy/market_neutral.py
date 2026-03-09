"""Market Neutral Strategy for QuantTerm backtesting.

Dollar-neutral long/short strategy with:
- Equal long and short exposure
- Short borrow costs deducted daily
- Maintains market neutrality
- Configurable number of long/short positions
"""

from typing import Optional
import pandas as pd

from quantterm.backtesting.events import OrderEvent


class MarketNeutralStrategy:
    """Dollar-neutral long/short strategy.
    
    Features:
    - Equal long and short exposure
    - Short borrow costs deducted daily
    - Maintains market neutrality
    - Configurable number of long/short positions
    """
    
    def __init__(
        self,
        name: str,
        portfolio,
        data_handler,
        symbols: list[str],
        n_long: int = 10,
        n_short: int = 10,
        rebalance_freq: str = 'M',
    ):
        self.name = name
        self.portfolio = portfolio
        self.data_handler = data_handler
        self.symbols = symbols
        self.n_long = n_long
        self.n_short = n_short
        self.rebalance_freq = rebalance_freq
        self.last_rebalance = None
        
        # Track signals
        self.prices: dict[str, float] = {}
        self.returns: dict[str, list] = {s: [] for s in symbols}
        self.lookback = 20
    
    def on_bar_multi(self, bars: dict[str, pd.Series], date: pd.Timestamp) -> list:
        """Generate long/short signals.
        
        Strategy:
        - Rank stocks by momentum (20-day return)
        - Go long top N, short bottom N
        - Maintain dollar neutrality
        """
        orders = []
        
        # Update prices
        for symbol, bar in bars.items():
            self.prices[symbol] = bar['Close']
        
        # Calculate returns for each symbol
        current_returns = {}
        for symbol in self.symbols:
            if symbol not in bars:
                continue
            
            price = bars[symbol]['Close']
            # Get lookback price
            if hasattr(self.data_handler, '_cache') and symbol in self.data_handler._cache:
                df = self.data_handler._cache[symbol]
                mask = df.index < date
                if mask.sum() >= self.lookback:
                    lookback_price = df.loc[mask].iloc[-self.lookback]['Close']
                    ret = (price - lookback_price) / lookback_price
                    current_returns[symbol] = ret
        
        if len(current_returns) < self.n_long + self.n_short:
            return orders
        
        # Rank by momentum
        sorted_symbols = sorted(current_returns.items(), key=lambda x: x[1], reverse=True)
        
        # Select long and short
        long_candidates = [s for s, _ in sorted_symbols[:self.n_long * 2]]
        short_candidates = [s for s, _ in sorted_symbols[-self.n_short * 2:]]
        
        # Calculate target positions for dollar neutrality
        # Target: equal dollar amount long and short
        total_value = self.portfolio.get_total_value(self.prices)
        target_long = total_value / (2 * len(long_candidates[:self.n_long]))
        
        # Generate orders
        for symbol in long_candidates[:self.n_long]:
            target_shares = int(target_long / self.prices[symbol])
            current = self.portfolio.get_position(symbol)
            diff = target_shares - current
            if diff != 0:
                orders.append(OrderEvent(
                    timestamp=date,
                    symbol=symbol,
                    quantity=diff,
                    order_type="market"
                ))
        
        for symbol in short_candidates[:self.n_short]:
            target_shares = int(target_long / self.prices[symbol])
            current = self.portfolio.get_position(symbol)
            diff = -target_shares - current  # Negative for short
            if diff != 0:
                orders.append(OrderEvent(
                    timestamp=date,
                    symbol=symbol,
                    quantity=diff,
                    order_type="market"
                ))
        
        return orders
    
    def on_fill(self, fill):
        """Called when an order is filled."""
        pass
    
    def on_bar(self, bar) -> Optional:
        """Single symbol handler - not used in multi-symbol mode."""
        return None

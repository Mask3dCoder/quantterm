"""
Live strategy runner - bridges backtest strategies to live data.
"""
import asyncio
from typing import Optional
from quantterm.live.data_feed import LiveDataFeed, LiveTick, BarAggregator
from quantterm.live.paper_trading import PaperTradingEngine
from quantterm.backtesting.events import BarEvent


class LiveStrategyRunner:
    """
    Run a strategy in live paper trading mode.
    
    Connects:
    - LiveDataFeed (market data)
    - PaperTradingEngine (order execution)
    - Strategy (signal generation)
    """
    
    def __init__(
        self,
        strategy,
        symbols: list[str],
        data_feed: LiveDataFeed,
        trading_engine: PaperTradingEngine,
        bar_interval: int = 60  # seconds
    ):
        self.strategy = strategy
        self.symbols = symbols
        self.data = data_feed
        self.trading = trading_engine
        self.bar_aggregator = BarAggregator(symbols, bar_interval)
        
        self._running = False
    
    async def start(self):
        """Start the live strategy."""
        self._running = True
        
        # Register tick callbacks
        for symbol in self.symbols:
            self.data.on_tick(symbol, self._on_tick)
        
        # Start data feed
        await self.data.start(self.symbols)
    
    async def _on_tick(self, tick: LiveTick):
        """Process incoming tick."""
        if not self._running:
            return
        
        # Update trading engine with current price
        self.trading.update_market_prices({tick.symbol: tick.price})
        
        # Aggregate to bars
        bar = self.bar_aggregator.add_tick(tick)
        
        if bar is not None:
            # Create bar event for strategy
            event = BarEvent(
                timestamp=bar['timestamp'],
                symbol=bar['symbol'],
                open=bar['open'],
                high=bar['high'],
                low=bar['low'],
                close=bar['close'],
                volume=int(bar['volume'])
            )
            
            # Get signal from strategy
            order = self.strategy.on_bar(event)
            
            if order is not None:
                # Execute through paper trading
                try:
                    side = 'buy' if order.quantity > 0 else 'sell'
                    trade = self.trading.execute_market_order(
                        symbol=order.symbol,
                        side=side,
                        quantity=abs(order.quantity),
                        current_price=tick.price,
                        bid=tick.bid,
                        ask=tick.ask
                    )
                    print(f"📝 PAPER TRADE: {side.upper()} {trade.quantity} {order.symbol} @ ${trade.price:.2f}")
                    
                except Exception as e:
                    print(f"❌ Order failed: {e}")
    
    def stop(self):
        """Stop the strategy."""
        self._running = False
        self.data.stop()
        
        # Flatten positions
        prices = {s: self.data.get_latest_price(s) for s in self.symbols}
        self.trading.flatten(prices)
    
    def get_status(self) -> dict:
        """Get current status."""
        prices = {s: self.data.get_latest_price(s) for s in self.symbols}
        return self.trading.get_summary(prices)

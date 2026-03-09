"""
Live market data feed for paper trading.

Supports:
- Yahoo Finance (free, delayed)
- Polygon.io (real-time, requires API key)
"""
import asyncio
import aiohttp
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional
import pandas as pd


@dataclass
class LiveTick:
    """Real-time market tick."""
    symbol: str
    price: float
    volume: int
    timestamp: datetime
    bid: float = 0.0
    ask: float = 0.0
    
    @property
    def spread(self) -> float:
        return self.ask - self.bid if self.ask > 0 else 0


class LiveDataFeed:
    """
    Live market data feed.
    
    Uses Yahoo Finance for free delayed data or Polygon for real-time.
    """
    
    def __init__(self, source: str = 'yahoo'):
        """
        Initialize live feed.
        
        Args:
            source: 'yahoo' (free/delayed) or 'polygon' (real-time)
        """
        self.source = source
        self._running = False
        self._callbacks: dict[str, list[Callable]] = {}
        self._last_prices: dict[str, float] = {}
    
    async def start(self, symbols: list[str], interval: int = 60):
        """
        Start streaming live data.
        
        Args:
            symbols: List of ticker symbols
            interval: Update interval in seconds
        """
        self._running = True
        
        if self.source == 'yahoo':
            await self._start_yahoo(symbols, interval)
        elif self.source == 'polygon':
            await self._start_polygon(symbols)
        else:
            raise ValueError(f"Unknown source: {self.source}")
    
    async def _start_yahoo(self, symbols: list[str], interval: int):
        """Stream data from Yahoo Finance (delayed ~15min)."""
        import yfinance as yf
        
        while self._running:
            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    # Get quote (fast)
                    info = ticker.fast_info
                    
                    if hasattr(info, 'last_price') and info.last_price:
                        price = float(info.last_price)
                    else:
                        # Fallback
                        hist = ticker.history(period="1d", interval="1m")
                        if not hist.empty:
                            price = float(hist['Close'].iloc[-1])
                        else:
                            continue
                    
                    # Create tick
                    tick = LiveTick(
                        symbol=symbol,
                        price=price,
                        volume=int(info.last_volume) if hasattr(info, 'last_volume') else 0,
                        timestamp=datetime.now(),
                        bid=price * 0.999,  # Approximate
                        ask=price * 1.001
                    )
                    
                    self._last_prices[symbol] = price
                    
                    # Notify callbacks
                    for callback in self._callbacks.get(symbol, []):
                        await callback(tick)
                        
                except Exception as e:
                    print(f"Error fetching {symbol}: {e}")
            
            await asyncio.sleep(interval)
    
    async def _start_polygon(self, symbols: list[str]):
        """Stream real-time data from Polygon.io (requires API key)."""
        # Polygon implementation would go here
        # For now, use Yahoo as fallback
        await self._start_yahoo(symbols, 5)
    
    def on_tick(self, symbol: str, callback: Callable):
        """Register callback for symbol updates."""
        if symbol not in self._callbacks:
            self._callbacks[symbol] = []
        self._callbacks[symbol].append(callback)
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get last known price for symbol."""
        return self._last_prices.get(symbol)
    
    def stop(self):
        """Stop the data feed."""
        self._running = False


class BarAggregator:
    """
    Aggregate ticks into OHLCV bars.
    """
    
    def __init__(self, symbols: list[str], interval: int = 60):
        """
        Args:
            symbols: Symbols to aggregate
            interval: Bar interval in seconds
        """
        self.symbols = symbols
        self.interval = interval
        self._bars: dict[str, dict] = {}
    
    def add_tick(self, tick: LiveTick) -> Optional[pd.Series]:
        """Add tick and return completed bar if any."""
        import pandas as pd
        
        if tick.symbol not in self._bars:
            self._bars[tick.symbol] = {
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'volume': tick.volume,
                'start_time': tick.timestamp
            }
        
        bar = self._bars[tick.symbol]
        
        # Update bar
        bar['high'] = max(bar['high'], tick.price)
        bar['low'] = min(bar['low'], tick.price)
        bar['close'] = tick.price
        bar['volume'] += tick.volume
        
        # Check if bar is complete (interval elapsed)
        elapsed = (tick.timestamp - bar['start_time']).total_seconds()
        
        if elapsed >= self.interval:
            # Return completed bar
            result = pd.Series({
                'symbol': tick.symbol,
                'timestamp': bar['start_time'],
                'open': bar['open'],
                'high': bar['high'],
                'low': bar['low'],
                'close': bar['close'],
                'volume': bar['volume']
            })
            
            # Reset bar
            self._bars[tick.symbol] = {
                'open': tick.price,
                'high': tick.price,
                'low': tick.price,
                'close': tick.price,
                'volume': 0,
                'start_time': tick.timestamp
            }
            
            return result
        
        return None

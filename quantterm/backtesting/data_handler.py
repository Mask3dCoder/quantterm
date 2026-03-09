"""Data handler for fetching market data from Yahoo Finance.

Provides a simple interface to download OHLCV data for backtesting.
Supports both single and multi-symbol data retrieval.
"""

from datetime import datetime
from typing import Optional, Dict

import pandas as pd
import yfinance as yf


class DataHandler:
    """Yahoo Finance data handler for backtesting.

    Fetches historical OHLCV data using yfinance library.
    Uses adjusted prices to account for corporate actions.
    """

    def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
    ) -> pd.DataFrame:
        """Get OHLCV bars from Yahoo Finance.

        Args:
            symbol: Ticker symbol (e.g., 'SPY', 'AAPL').
            start: Start date in 'YYYY-MM-DD' format.
            end: End date in 'YYYY-MM-DD' format.

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume.
            The timestamp is set as the index.

        Raises:
            ValueError: If no data is returned from Yahoo Finance.
        """
        # Download data from Yahoo Finance with auto_adjust=True
        # to use adjusted close prices
        df = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
        )
        
        if df.empty:
            raise ValueError(f"No data found for {symbol} between {start} and {end}")
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df.columns, pd.MultiIndex):
            # Flatten the MultiIndex columns
            df.columns = df.columns.get_level_values(0)
        
        # Reset index to make datetime a column
        df = df.reset_index()
        
        # Rename columns to match expected format
        # Yahoo Finance returns: Date, Open, High, Low, Close, Volume
        df = df.rename(columns={"Date": "timestamp"})
        
        # Ensure timestamp is datetime
        if not isinstance(df["timestamp"].iloc[0], datetime):
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        
        # Select and reorder columns
        df = df[["timestamp", "Open", "High", "Low", "Close", "Volume"]]
        df = df.rename(columns={"Volume": "volume"})
        
        return df

    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol.

        Args:
            symbol: Ticker symbol.

        Returns:
            Latest closing price, or None if unavailable.
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d", auto_adjust=True)
            if df.empty:
                return None
            return float(df["Close"].iloc[-1])
        except Exception:
            return None


class MultiSymbolDataHandler:
    """Handle multiple symbols with different start dates and missing data.
    
    Provides methods for loading multi-symbol data, finding common trading
    days, and handling missing data (halts, suspensions).
    """
    
    def __init__(self):
        """Initialize the multi-symbol data handler."""
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_bars(
        self, 
        symbols: list[str], 
        start: str, 
        end: str
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols.
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).
            
        Returns:
            Dict mapping symbol to OHLCV DataFrame.
        """
        result = {}
        for symbol in symbols:
            try:
                # Download data from Yahoo Finance using yf.download
                df = yf.download(
                    symbol, 
                    start=start, 
                    end=end, 
                    auto_adjust=True,
                    progress=False
                )
                if df is not None and not df.empty:
                    # Handle MultiIndex columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        # Flatten the MultiIndex columns
                        df.columns = df.columns.get_level_values(0)
                    # Select and rename columns
                    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    df = df.rename(columns={'Volume': 'volume'})
                    result[symbol] = df
            except Exception:
                # Skip symbols that fail to download
                continue
        return result
    
    def get_common_dates(
        self, 
        data: Dict[str, pd.DataFrame]
    ) -> pd.DatetimeIndex:
        """Return trading days where ALL symbols have data.
        
        Critical for portfolio backtests to avoid lookahead bias.
        Finds the intersection of all symbol date indices.
        
        Args:
            data: Dict mapping symbol to OHLCV DataFrame.
            
        Returns:
            Sorted DatetimeIndex of common trading days.
        """
        if not data:
            return pd.DatetimeIndex([])
        
        # Find intersection of all symbol dates
        common = None
        for symbol, df in data.items():
            dates = df.index
            if common is None:
                common = dates
            else:
                common = common.intersection(dates)
        
        return common.sort_values()
    
    def get_latest_bars(
        self, 
        data: Dict[str, pd.DataFrame], 
        date: pd.Timestamp
    ) -> Dict[str, pd.Series]:
        """Get latest bar for each symbol up to date.
        
        Handles missing data (halted stocks) by returning the last
        available bar for each symbol.
        
        Args:
            data: Dict mapping symbol to OHLCV DataFrame.
            date: Current date.
            
        Returns:
            Dict mapping symbol to latest bar Series.
        """
        result = {}
        for symbol, df in data.items():
            # Get bars up to and including this date
            mask = df.index <= date
            if mask.any():
                # Get the last available bar
                result[symbol] = df.loc[mask].iloc[-1]
        return result
    
    def get_symbol_dates(
        self, 
        data: Dict[str, pd.DataFrame], 
        symbol: str
    ) -> pd.DatetimeIndex:
        """Get available trading dates for a specific symbol.
        
        Args:
            data: Dict mapping symbol to OHLCV DataFrame.
            symbol: Symbol to get dates for.
            
        Returns:
            DatetimeIndex of trading dates for the symbol.
        """
        if symbol not in data:
            return pd.DatetimeIndex([])
        return data[symbol].index


class IntradayDataHandler:
    """Handle intraday bar data.
    
    Features:
    - Multiple timeframes (1m, 5m, 15m, 1h, 1d)
    - Market hours filtering (9:30-16:00 ET)
    - Pre/post market handling
    - Gap detection (trading halts)
    """
    
    VALID_INTERVALS = ('1m', '5m', '15m', '1h', '1d')
    
    def __init__(self, interval: str = '5m'):
        """Initialize with timeframe.
        
        Args:
            interval: Yahoo Finance interval (1m, 5m, 15m, 1h, 1d)
        """
        if interval not in self.VALID_INTERVALS:
            raise ValueError(f"Invalid interval: {interval}. Valid: {self.VALID_INTERVALS}")
        self.interval = interval
        self._cache: Dict[str, pd.DataFrame] = {}
    
    def get_bars(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = None,
    ) -> pd.DataFrame:
        """Load intraday bars for a symbol.
        
        Args:
            symbol: Ticker symbol
            start: Start date (YYYY-MM-DD)
            end: End date (YYYY-MM-DD)
            interval: Optional override for timeframe
            
        Returns:
            DataFrame with OHLCV data
        """
        import yfinance as yf
        
        intvl = interval or self.interval
        
        # yfinance uses different format for intraday
        df = yf.download(
            symbol,
            start=start,
            end=end,
            interval=intvl,
            progress=False
        )
        
        if df.empty:
            return df
        
        # Handle MultiIndex columns (yfinance quirk)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        return df
    
    def get_day_bars(
        self,
        symbol: str,
        date: str,  # YYYY-MM-DD
        interval: str = None,
    ) -> pd.DataFrame:
        """Load intraday bars for a single day.
        
        Args:
            symbol: Ticker symbol
            date: Specific date (YYYY-MM-DD)
            interval: Timeframe
            
        Returns:
            DataFrame with intraday bars for the day
        """
        # Load a bit extra to cover the day
        df = self.get_bars(symbol, date, date, interval)
        
        if df.empty:
            return df
        
        # Filter to market hours (9:30 AM - 4:00 PM ET)
        # Convert index to timezone-aware
        try:
            df = df.tz_convert('America/New_York')
        except TypeError:
            # Already timezone-aware or no timezone
            pass
        
        # Market hours: 9:30 - 16:00
        market_open = df.index.indexer_between_time('9:30', '9:30')
        market_close = df.index.indexer_between_time('15:59', '16:00')
        
        if len(market_open) > 0 and len(market_close) > 0:
            df = df.iloc[market_open[0]:market_close[-1]+1]
        
        return df
    
    def get_bars_for_dates(
        self,
        symbol: str,
        dates: list[str],
        interval: str = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load intraday bars for multiple dates.
        
        Returns:
            Dict mapping date -> DataFrame of bars
        """
        result = {}
        for date in dates:
            df = self.get_day_bars(symbol, date, interval)
            if not df.empty:
                result[date] = df
        return result
    
    def detect_gaps(
        self,
        bars: pd.DataFrame,
        threshold_pct: float = 5.0
    ) -> list:
        """Detect price gaps between consecutive bars.
        
        Args:
            bars: DataFrame with OHLCV data
            threshold_pct: Minimum gap percentage to detect
            
        Returns:
            List of dicts with gap information
        """
        if len(bars) < 2:
            return []
        
        gaps = []
        for i in range(1, len(bars)):
            prev_close = bars.iloc[i-1]['Close']
            curr_open = bars.iloc[i]['Open']
            gap_pct = abs(curr_open - prev_close) / prev_close * 100
            
            if gap_pct >= threshold_pct:
                gaps.append({
                    'timestamp': bars.index[i],
                    'direction': 'up' if curr_open > prev_close else 'down',
                    'gap_pct': gap_pct,
                    'prev_close': prev_close,
                    'open': curr_open
                })
        
        return gaps


class IntradayFillModel:
    """Determine fill price within OHLC bar for backtesting.
    
    Conservative approach:
    - Buy limit: fill at min(ask, limit) if low <= limit
    - Sell limit: fill at max(bid, limit) if high >= limit  
    - Market orders: fill at close (worst case for estimation)
    """
    
    @staticmethod
    def get_market_fill(bar: pd.Series, quantity: int) -> float:
        """Market order fill - assume worst case (close price).
        
        For more realism:
        - Buy: fill at high (optimistic) or close (conservative)
        - Sell: fill at low (optimistic) or close (conservative)
        """
        if quantity > 0:  # Buy
            # Conservative: fill at close
            return bar['Close']
        else:  # Sell
            return bar['Close']
    
    @staticmethod
    def get_limit_fill(bar: pd.Series, quantity: int, limit_price: float) -> tuple:
        """Limit order fill simulation.
        
        Returns:
            (filled: bool, fill_price: float)
        """
        if quantity > 0:  # Buy limit
            # Filled if we touched or crossed the limit
            if bar['Low'] <= limit_price:
                # Best case: limit price, worst case: low
                fill_price = min(limit_price, bar['Open'])
                return True, fill_price
            return False, None
        else:  # Sell limit
            if bar['High'] >= limit_price:
                fill_price = max(limit_price, bar['Open'])
                return True, fill_price
            return False, None
    
    @staticmethod
    def get_stop_fill(bar: pd.Series, quantity: int, stop_price: float) -> tuple:
        """Stop order fill simulation.
        
        Returns:
            (filled: bool, fill_price: float)
        """
        if quantity > 0:  # Buy stop
            if bar['High'] >= stop_price:
                # Filled at stop or better (high for buy stop)
                fill_price = min(stop_price, bar['High'])
                return True, fill_price
            return False, None
        else:  # Sell stop
            if bar['Low'] <= stop_price:
                fill_price = max(stop_price, bar['Low'])
                return True, fill_price
            return False, None

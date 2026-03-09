"""
Feature Engineering for ML-Based Trading Strategies.

CRITICAL: No lookahead bias. All features are computed using only
data available at prediction time.
"""
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime


class FeatureEngineer:
    """
    Transform price/volume data into ML features.
    
    Features computed:
    - Trend: Returns over various windows
    - Volatility: Realized volatility
    - Volume: Volume trends and ratios
    - Technical: RSI, moving average distances
    """
    
    def __init__(
        self,
        lookback_windows: list[int] = None,
        target_horizon: int = 5
    ):
        """
        Initialize feature engineer.
        
        Args:
            lookback_windows: Windows for feature computation
            target_horizon: Days ahead for target prediction
        """
        self.windows = lookback_windows or [5, 10, 20, 60]
        self.target_horizon = target_horizon
    
    def create_features(
        self,
        data: pd.DataFrame,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Optional[pd.Series]:
        """
        Generate feature vector at specific point in time.
        
        Args:
            data: DataFrame with OHLCV columns
            timestamp: Point in time (if None, uses last row)
            
        Returns:
            Series with features, or None if insufficient history
        """
        if timestamp is None:
            # Use all data up to and including current point
            data_slice = data
        else:
            # Get data up to timestamp (no lookahead)
            data_slice = data.loc[:timestamp]
        
        if len(data_slice) < max(self.windows) + 10:
            return None
        
        features = {}
        # Handle case-insensitive column names
        close_col = 'Close' if 'Close' in data_slice.columns else 'close'
        volume_col = 'Volume' if 'Volume' in data_slice.columns else 'volume'
        
        close = data_slice[close_col]
        volume = data_slice[volume_col]
        
        # Price-based features
        for window in self.windows:
            if len(data_slice) < window:
                continue
                
            subset = close.tail(window)
            
            # Returns (momentum)
            ret = (close.iloc[-1] - subset.iloc[0]) / subset.iloc[0] if subset.iloc[0] != 0 else 0
            features[f'return_{window}d'] = ret
            
            # Volatility
            rets = close.pct_change().tail(window)
            features[f'volatility_{window}d'] = rets.std() * np.sqrt(252) if len(rets) > 1 else 0
            
            # RSI
            features[f'rsi_{window}d'] = self._calculate_rsi(close, window)
            
            # Distance from SMA
            sma = subset.mean()
            features[f'dist_sma_{window}'] = (close.iloc[-1] / sma - 1) if sma != 0 else 0
        
        # Volume features
        if len(volume) >= 20:
            features['volume_vs_avg20'] = volume.iloc[-1] / volume.tail(20).mean()
            features['volume_trend'] = volume.tail(5).mean() / volume.tail(20).mean()
        else:
            features['volume_vs_avg20'] = 1.0
            features['volume_trend'] = 1.0
        
        # Price momentum (recent vs longer-term)
        if len(close) >= 20:
            short_ret = (close.iloc[-1] - close.tail(5).iloc[0]) / close.tail(5).iloc[0]
            long_ret = (close.iloc[-1] - close.tail(20).iloc[0]) / close.tail(20).iloc[0]
            features['momentum_ratio'] = short_ret - long_ret if long_ret != 0 else 0
        else:
            features['momentum_ratio'] = 0
        
        return pd.Series(features)
    
    def create_features_batch(
        self,
        data: pd.DataFrame,
        min_history: int = None
    ) -> pd.DataFrame:
        """
        Create features for all timestamps in a dataset.
        
        Args:
            data: DataFrame with OHLCV data
            min_history: Minimum history required before computing features
            
        Returns:
            DataFrame with features for each timestamp
        """
        min_hist = min_history or (max(self.windows) + 10)
        
        if len(data) < min_hist:
            return pd.DataFrame()
        
        features_list = []
        
        for i in range(min_hist, len(data)):
            timestamp = data.index[i]
            hist_data = data.iloc[:i+1]  # No lookahead
            
            feats = self.create_features(hist_data, timestamp)
            if feats is not None:
                feats.name = timestamp
                features_list.append(feats)
        
        if not features_list:
            return pd.DataFrame()
        
        return pd.DataFrame(features_list)
    
    def create_target(
        self,
        data: pd.DataFrame,
        timestamp: pd.Timestamp,
        horizon: int = None
    ) -> Optional[float]:
        """
        Create target: Forward return over horizon.
        
        Args:
            data: DataFrame with Close prices
            timestamp: Current timestamp
            horizon: Days ahead (uses self.target_horizon if None)
            
        Returns:
            Forward return, or None if cannot compute
        """
        h = horizon or self.target_horizon
        
        # Get current price
        current_data = data.loc[:timestamp]
        if len(current_data) < 1:
            return None
            
        current_price = current_data['Close'].iloc[-1]
        
        # Get future price
        future_idx = data.index.get_loc(timestamp) + h
        if future_idx >= len(data):
            return None
            
        future_price = data['Close'].iloc[future_idx]
        
        # Forward return
        return (future_price / current_price) - 1
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI indicator."""
        if len(prices) < period + 1:
            return 50.0  # Neutral
        
        deltas = prices.diff()
        gains = deltas.where(deltas > 0, 0)
        losses = -deltas.where(deltas < 0, 0)
        
        avg_gain = gains.tail(period).mean()
        avg_loss = losses.tail(period).mean()
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

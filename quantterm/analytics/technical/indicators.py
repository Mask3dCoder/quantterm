"""Technical analysis indicators for QuantTerm.

This module provides 50+ technical indicators including:
- Trend indicators (MA, EMA, MACD, Ichimoku)
- Momentum indicators (RSI, Stochastic, ROC)
- Volatility indicators (Bollinger Bands, ATR, Keltner)
- Volume indicators (OBV, VWAP, MFI)
- Pattern recognition utilities
"""
import numpy as np
import pandas as pd
from typing import Optional, Tuple, Union


# Trend Indicators

def sma(prices: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        SMA values (same length as input, NaN for first period-1 values)
    """
    if len(prices) < period:
        return np.full_like(prices, np.nan)
    
    # Use convolution for efficiency, then pad to maintain array length
    weights = np.ones(period) / period
    result = np.convolve(prices, weights, mode='valid')
    
    # Prepend NaNs to maintain alignment with input array
    return np.concatenate([np.full(period - 1, np.nan), result])


def ema(prices: np.ndarray, period: int) -> np.ndarray:
    """Exponential Moving Average.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        EMA values
    """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def wma(prices: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        WMA values
    """
    weights = np.arange(1, period + 1)
    wma = np.convolve(prices, weights/weights.sum(), mode='valid')
    return wma


def hma(prices: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        HMA values
    """
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    # WMA of half period
    wma_half = wma(prices, half_period)
    # WMA of full period
    wma_full = wma(prices, period)
    
    # Calculate HMA
    hma_raw = 2 * wma_half - wma_full
    hma = wma(hma_raw, sqrt_period)
    
    return hma


def macd(
    prices: np.ndarray,
    fast_period: int = 12,
    slow_period: int = 26,
    signal_period: int = 9
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MACD (Moving Average Convergence Divergence).
    
    Args:
        prices: Price series
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        signal_period: Signal line period
        
    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    fast_ema = ema(prices, fast_period)
    slow_ema = ema(prices, slow_period)
    
    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def ichimoku(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    tenkan_period: int = 9,
    kijun_period: int = 26,
    senkou_b_period: int = 52
) -> dict:
    """Ichimoku Cloud indicator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        tenkan_period: Tenkan-sen (Conversion Line) period
        kijun_period: Kijun-sen (Base Line) period
        senkou_b_period: Senkou Span B period
        
    Returns:
        Dictionary with Ichimoku components
    """
    # Tenkan-sen (Conversion Line)
    tenkan = (pd.Series(high).rolling(tenkan_period).max() + 
              pd.Series(low).rolling(tenkan_period).min()) / 2
    
    # Kijun-sen (Base Line)
    kijun = (pd.Series(high).rolling(kijun_period).max() + 
             pd.Series(low).rolling(kijun_period).min()) / 2
    
    # Senkou Span A (Leading Span A)
    senkou_a = ((tenkan + kijun) / 2).shift(kijun_period)
    
    # Senkou Span B (Leading Span B)
    senkou_b = ((pd.Series(high).rolling(senkou_b_period).max() + 
                 pd.Series(low).rolling(senkou_b_period).min()) / 2).shift(kijun_period)
    
    # Chikou Span (Lagging Span)
    chikou = pd.Series(close).shift(-kijun_period)
    
    return {
        'tenkan': tenkan.values,
        'kijun': kijun.values,
        'senkou_a': senkou_a.values,
        'senkou_b': senkou_b.values,
        'chikou': chikou.values
    }


# Momentum Indicators

def rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
    """Relative Strength Index.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        RSI values (0-100)
    """
    deltas = np.diff(prices)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    avg_gain = sma(gains, period)
    avg_loss = sma(losses, period)
    
    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def stochastic(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14,
    smooth_k: int = 3,
    smooth_d: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Stochastic Oscillator.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        smooth_k: %K smoothing period
        smooth_d: %D smoothing period
        
    Returns:
        Tuple of (%K, %D)
    """
    lowest_low = np.array([np.min(low[max(0, i-period+1):i+1]) 
                          for i in range(len(low))])
    highest_high = np.array([np.max(high[max(0, i-period+1):i+1]) 
                            for i in range(len(high))])
    
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    k_smooth = sma(k, smooth_k)
    d = sma(k_smooth, smooth_d)
    
    return k_smooth, d


def roc(prices: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change.
    
    Args:
        prices: Price series
        period: Lookback period
        
    Returns:
        ROC values
    """
    roc = np.zeros_like(prices)
    roc[period:] = ((prices[period:] - prices[:-period]) / prices[:-period]) * 100
    return roc


def cci(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 20
) -> np.ndarray:
    """Commodity Channel Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        
    Returns:
        CCI values
    """
    typical_price = (high + low + close) / 3
    sma_tp = sma(typical_price, period)
    mean_deviation = np.array([
        np.mean(np.abs(typical_price[max(0, i-period+1):i+1] - sma_tp[i]))
        for i in range(len(typical_price))
    ])
    
    cci = (typical_price - sma_tp) / (0.015 * mean_deviation + 1e-10)
    return cci


def williams_r(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """Williams %R.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        
    Returns:
        Williams %R values (-100 to 0)
    """
    highest_high = np.array([np.max(high[max(0, i-period+1):i+1]) 
                            for i in range(len(high))])
    lowest_low = np.array([np.min(low[max(0, i-period+1):i+1]) 
                          for i in range(len(low))])
    
    williams = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)
    return williams


# Volatility Indicators

def bollinger_bands(
    prices: np.ndarray,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands.
    
    Args:
        prices: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (Upper band, Middle band, Lower band)
    """
    # Use pandas for consistent NaN handling
    prices_series = pd.Series(prices)
    
    # Calculate middle band (SMA) using pandas for consistent NaN handling
    middle = prices_series.rolling(window=period).mean().to_numpy()
    
    # Calculate rolling standard deviation - same length as input with NaN at start
    rolling_std = prices_series.rolling(window=period).std().to_numpy()
    
    # Create masks for valid data (where both middle and rolling_std are not NaN)
    valid_mask = ~(np.isnan(middle) | np.isnan(rolling_std))
    
    # Initialize output arrays with NaN
    upper = np.full_like(prices, np.nan, dtype=np.float64)
    lower = np.full_like(prices, np.nan, dtype=np.float64)
    
    # Calculate bands only where we have valid data
    upper[valid_mask] = middle[valid_mask] + (rolling_std[valid_mask] * std_dev)
    lower[valid_mask] = middle[valid_mask] - (rolling_std[valid_mask] * std_dev)
    
    return upper, middle, lower


def atr(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ATR period
        
    Returns:
        ATR values
    """
    tr = np.zeros(len(high))
    tr[0] = high[0] - low[0]
    
    for i in range(1, len(high)):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr = sma(tr, period)
    return atr


def keltner_channels(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ema_period: int = 20,
    atr_period: int = 10,
    multiplier: float = 2.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keltner Channels.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        ema_period: EMA period for middle line
        atr_period: ATR period
        multiplier: ATR multiplier
        
    Returns:
        Tuple of (Upper, Middle, Lower channels)
    """
    middle = ema(close, ema_period)
    atr_values = atr(high, low, close, atr_period)
    
    upper = middle + (multiplier * atr_values)
    lower = middle - (multiplier * atr_values)
    
    return upper, middle, lower


def donchian_channels(
    high: np.ndarray,
    low: np.ndarray,
    period: int = 20
) -> Tuple[np.ndarray, np.ndarray]:
    """Donchian Channels.
    
    Args:
        high: High prices
        low: Low prices
        period: Lookback period
        
    Returns:
        Tuple of (Upper channel, Lower channel)
    """
    upper = np.array([np.max(high[max(0, i-period+1):i+1]) 
                     for i in range(len(high))])
    lower = np.array([np.min(low[max(0, i-period+1):i+1]) 
                     for i in range(len(low))])
    
    return upper, lower


# Volume Indicators

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On-Balance Volume.
    
    Args:
        close: Close prices
        volume: Volume
        
    Returns:
        OBV values
    """
    obv = np.zeros(len(close))
    obv[0] = volume[0]
    
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv[i] = obv[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv[i] = obv[i-1] - volume[i]
        else:
            obv[i] = obv[i-1]
    
    return obv


def vwap(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume Weighted Average Price.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        
    Returns:
        VWAP values
    """
    typical_price = (high + low + close) / 3
    cumulative_tpv = np.cumsum(typical_price * volume)
    cumulative_volume = np.cumsum(volume)
    
    vwap = cumulative_tpv / (cumulative_volume + 1e-10)
    return vwap


def mfi(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    period: int = 14
) -> np.ndarray:
    """Money Flow Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        period: MFI period
        
    Returns:
        MFI values (0-100)
    """
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    
    positive_flow = np.zeros(len(typical_price))
    negative_flow = np.zeros(len(typical_price))
    
    for i in range(1, len(typical_price)):
        if typical_price[i] > typical_price[i-1]:
            positive_flow[i] = money_flow[i]
        elif typical_price[i] < typical_price[i-1]:
            negative_flow[i] = money_flow[i]
    
    period_positive = sma(positive_flow, period)
    period_negative = sma(negative_flow, period)
    
    mfi = 100 - (100 / (1 + period_positive / (period_negative + 1e-10)))
    
    return mfi


def ad_line(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Accumulation/Distribution Line.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        volume: Volume
        
    Returns:
        A/D Line values
    """
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-10)
    money_flow_multiplier = np.nan_to_num(money_flow_multiplier)
    
    money_flow_volume = money_flow_multiplier * volume
    
    ad = np.cumsum(money_flow_volume)
    return ad


def adx(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    period: int = 14
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average Directional Index.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: ADX period
        
    Returns:
        Tuple of (ADX, +DI, -DI)
    """
    plus_dm = np.zeros(len(high))
    minus_dm = np.zeros(len(high))
    tr = np.zeros(len(high))
    
    for i in range(1, len(high)):
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]
        
        plus_dm[i] = up_move if up_move > down_move and up_move > 0 else 0
        minus_dm[i] = down_move if down_move > up_move and down_move > 0 else 0
        
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )
    
    atr_values = atr(high, low, close, period)
    
    plus_di = 100 * sma(plus_dm, period) / (atr_values + 1e-10)
    minus_di = 100 * sma(minus_dm, period) / (atr_values + 1e-10)
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = sma(dx, period)
    
    return adx, plus_di, minus_di


# Utility functions

def pivots(
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray
) -> dict:
    """Calculate pivot points and support/resistance levels.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        
    Returns:
        Dictionary with pivot levels
    """
    pivot = (high + low + close) / 3
    
    r1 = 2 * pivot - low
    s1 = 2 * pivot - high
    
    r2 = pivot + (high - low)
    s2 = pivot - (high - low)
    
    r3 = high + 2 * (pivot - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        'pivot': pivot,
        'r1': r1,
        'r2': r2,
        'r3': r3,
        's1': s1,
        's2': s2,
        's3': s3
    }


def fibonacci_retracement(high: float, low: float) -> dict:
    """Calculate Fibonacci retracement levels.
    
    Args:
        high: High price
        low: Low price
        
    Returns:
        Dictionary with Fibonacci levels
    """
    diff = high - low
    
    return {
        '0.0': high,
        '0.236': high - 0.236 * diff,
        '0.382': high - 0.382 * diff,
        '0.5': high - 0.5 * diff,
        '0.618': high - 0.618 * diff,
        '0.786': high - 0.786 * diff,
        '1.0': low
    }

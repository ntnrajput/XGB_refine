# utils/helpers.py - Helper functions for calculations

import pandas as pd
import numpy as np


def calculate_ema(series: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()


def calculate_sma(series: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average"""
    return series.rolling(window=period).mean()


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        period: Lookback period
        
    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    return atr


def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """
    Calculate Bollinger Bands.
    
    Args:
        series: Price series
        period: Moving average period
        std_dev: Standard deviation multiplier
        
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower


def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """
    Calculate MACD.
    
    Args:
        series: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
        
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = calculate_ema(series, fast)
    ema_slow = calculate_ema(series, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def safe_divide(numerator, denominator, fill_value=0):
    """
    Safely divide two series, handling division by zero.
    
    Args:
        numerator: Numerator series or scalar
        denominator: Denominator series or scalar
        fill_value: Value to use when denominator is 0
        
    Returns:
        Result series with inf/nan replaced
    """
    result = numerator / denominator.replace(0, np.nan)
    result = result.replace([np.inf, -np.inf], np.nan)
    
    return result.fillna(fill_value)


def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.
    
    Args:
        close: Close prices
        volume: Volume
        
    Returns:
        OBV series
    """
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    return obv


def calculate_momentum(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate price momentum.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        Momentum series
    """
    return series - series.shift(period)


def calculate_roc(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate Rate of Change.
    
    Args:
        series: Price series
        period: Lookback period
        
    Returns:
        ROC series (as percentage)
    """
    roc = ((series - series.shift(period)) / series.shift(period)) * 100
    return roc


def calculate_volatility(returns: pd.Series, period: int, annualize: bool = True) -> pd.Series:
    """
    Calculate rolling volatility.
    
    Args:
        returns: Return series
        period: Rolling window
        annualize: If True, annualize the volatility
        
    Returns:
        Volatility series
    """
    volatility = returns.rolling(period).std()
    
    if annualize:
        volatility = volatility * np.sqrt(252)
    
    return volatility * 100  # Convert to percentage


def calculate_zscore(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate rolling z-score.
    
    Args:
        series: Input series
        period: Rolling window
        
    Returns:
        Z-score series
    """
    mean = series.rolling(period).mean()
    std = series.rolling(period).std()
    
    zscore = (series - mean) / std.replace(0, np.nan)
    
    return zscore


def calculate_percentile_rank(series: pd.Series, period: int) -> pd.Series:
    """
    Calculate percentile rank over rolling window.
    
    Args:
        series: Input series
        period: Rolling window
        
    Returns:
        Percentile rank (0-100)
    """
    def percentile(x):
        if len(x) < 2:
            return 50
        return (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100
    
    return series.rolling(period).apply(percentile, raw=False)


__all__ = [
    'calculate_ema',
    'calculate_sma',
    'calculate_rsi',
    'calculate_atr',
    'calculate_bollinger_bands',
    'calculate_macd',
    'safe_divide',
    'calculate_obv',
    'calculate_momentum',
    'calculate_roc',
    'calculate_volatility',
    'calculate_zscore',
    'calculate_percentile_rank',
]

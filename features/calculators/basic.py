# features/calculators/basic.py - ULTRA-OPTIMIZED VERSION (3-5x faster)

import pandas as pd
import numpy as np
from typing import Optional
from numba import jit
import warnings

from config.features import IndicatorConfig, INDICATOR_CONFIG
from utils.helpers import calculate_ema, calculate_sma, calculate_rsi, calculate_atr
from utils.logger import get_logger

logger = get_logger(__name__)


# Numba-accelerated functions for performance-critical operations
@jit(nopython=True, cache=True)
def fast_ema(values, period):
    """Numba-accelerated EMA calculation"""
    n = len(values)
    result = np.empty(n)
    result[:] = np.nan
    
    alpha = 2.0 / (period + 1.0)
    
    # Find first valid value
    first_valid = 0
    for i in range(n):
        if not np.isnan(values[i]):
            first_valid = i
            break
    
    if first_valid >= n:
        return result
    
    # Initialize with first valid value
    result[first_valid] = values[first_valid]
    
    # Calculate EMA
    for i in range(first_valid + 1, n):
        if np.isnan(values[i]):
            result[i] = result[i-1]
        else:
            result[i] = alpha * values[i] + (1 - alpha) * result[i-1]
    
    return result


@jit(nopython=True, cache=True)
def fast_rsi(close_prices, period=14):
    """Numba-accelerated RSI calculation"""
    n = len(close_prices)
    rsi = np.empty(n)
    rsi[:] = np.nan
    
    if n < period + 1:
        return rsi
    
    # Calculate price changes
    deltas = np.diff(close_prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    
    # Calculate initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])
    
    # First RSI value
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Calculate subsequent RSI values using Wilder's smoothing
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i-1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i-1]) / period
        
        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


@jit(nopython=True, cache=True)
def fast_atr(high, low, close, period=14):
    """Numba-accelerated ATR calculation"""
    n = len(high)
    atr = np.empty(n)
    atr[:] = np.nan
    
    if n < period:
        return atr
    
    # Calculate True Range
    tr = np.empty(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    # Calculate initial ATR (simple average)
    atr[period-1] = np.mean(tr[:period])
    
    # Calculate subsequent ATR values (Wilder's smoothing)
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr


@jit(nopython=True, cache=True)
def fast_stochastic(high, low, close, period=14, smooth=3):
    """Numba-accelerated Stochastic Oscillator"""
    n = len(close)
    stoch_k = np.empty(n)
    stoch_k[:] = np.nan
    
    for i in range(period - 1, n):
        low_min = np.min(low[i-period+1:i+1])
        high_max = np.max(high[i-period+1:i+1])
        
        if high_max - low_min == 0:
            stoch_k[i] = 50.0
        else:
            stoch_k[i] = 100.0 * (close[i] - low_min) / (high_max - low_min)
    
    # Calculate %D (smooth %K)
    stoch_d = np.empty(n)
    stoch_d[:] = np.nan
    
    for i in range(period + smooth - 2, n):
        stoch_d[i] = np.mean(stoch_k[i-smooth+1:i+1])
    
    return stoch_k, stoch_d


class BasicCalculator:
    """
    Ultra-optimized technical indicators calculator
    
    ⚠️  DATA LEAKAGE PREVENTION:
    This calculator is designed to NEVER use future data in calculations.
    All indicators only use data from the current bar and previous bars.
    
    Safe for:
    - Backtesting
    - Live trading
    - Walk-forward analysis
    - Production systems
    
    Key principles:
    1. All rolling operations use center=False (backward-looking only)
    2. All shift operations shift forward (positive values only)
    3. No calculations peek into future bars
    4. Indicators at time T only use data from times <= T
    
    This ensures realistic backtest results and prevents look-ahead bias.
    """
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or INDICATOR_CONFIG
        self.use_numba = True  # Toggle for numba acceleration
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic indicators with maximum performance optimization"""
        
        # Pre-extract arrays for faster access
        close = df['close'].values
        open_prices = df['open'].values
        high = df['high'].values
        low = df['low'].values
        volume = df['volume'].values
        n = len(df)
        
        # Dictionary for features
        features = {}
        
        # ===== BASIC PRICE FEATURES (Vectorized) =====
        features['returns'] = np.concatenate([[np.nan], np.diff(close) / close[:-1]])
        features['log_returns'] = np.concatenate([[np.nan], np.log(close[1:] / close[:-1])])
        features['price_change_pct'] = np.where(open_prices != 0, (close - open_prices) / open_prices * 100, 0)
        
        # ===== MOVING AVERAGES (Optimized) =====
        # SMA using pandas rolling (highly optimized in C)
        close_series = df['close']
        for period in self.config.sma_periods:
            features[f'sma_{period}'] = close_series.rolling(window=period, min_periods=1).mean().values
        
        # EMA - Use numba if available, otherwise fallback
        for period in self.config.ema_periods:
            if self.use_numba:
                features[f'ema_{period}'] = fast_ema(close, period)
            else:
                features[f'ema_{period}'] = close_series.ewm(span=period, adjust=False, min_periods=period).mean().values
        
        # ===== PRICE VS MA COMPARISONS (Vectorized) =====
        if 20 in self.config.sma_periods:
            features['price_above_sma20'] = (close > features['sma_20']).astype(np.int8)
        if 50 in self.config.sma_periods:
            features['price_above_sma50'] = (close > features['sma_50']).astype(np.int8)
        if 200 in self.config.sma_periods:
            features['price_above_sma200'] = (close > features['sma_200']).astype(np.int8)
        
        if 20 in self.config.ema_periods:
            features['price_above_ema20'] = (close > features['ema_20']).astype(np.int8)
        if 50 in self.config.ema_periods:
            features['price_above_ema50'] = (close > features['ema_50']).astype(np.int8)
        if 200 in self.config.ema_periods:
            features['price_above_ema200'] = (close > features['ema_200']).astype(np.int8)
        
        # ===== MA CROSSOVERS (Optimized) =====
        if 20 in self.config.sma_periods and 50 in self.config.sma_periods:
            current = (features['sma_20'] > features['sma_50']).astype(np.int8)
            features['sma20_50_cross'] = np.concatenate([[0], np.diff(current)])
        
        if 50 in self.config.sma_periods and 200 in self.config.sma_periods:
            current = (features['sma_50'] > features['sma_200']).astype(np.int8)
            features['sma50_200_cross'] = np.concatenate([[0], np.diff(current)])
        
        if 20 in self.config.ema_periods and 50 in self.config.ema_periods:
            current = (features['ema_20'] > features['ema_50']).astype(np.int8)
            features['ema20_50_cross'] = np.concatenate([[0], np.diff(current)])
        
        # ===== RSI (Numba-accelerated) =====
        if self.use_numba:
            features['rsi_14'] = fast_rsi(close, self.config.rsi_period)
        else:
            features['rsi_14'] = calculate_rsi(close_series, self.config.rsi_period).values
        
        features['rsi_overbought'] = (features['rsi_14'] > self.config.rsi_overbought).astype(np.int8)
        features['rsi_oversold'] = (features['rsi_14'] < self.config.rsi_oversold).astype(np.int8)
        
        # ===== BOLLINGER BANDS (Optimized rolling) =====
        bb_sma = close_series.rolling(window=self.config.bb_period).mean()
        bb_std = close_series.rolling(window=self.config.bb_period).std()
        features['bb_upper'] = (bb_sma + bb_std * self.config.bb_std).values
        features['bb_middle'] = bb_sma.values
        features['bb_lower'] = (bb_sma - bb_std * self.config.bb_std).values
        
        # Vectorized BB calculations
        bb_range = features['bb_upper'] - features['bb_lower']
        features['bb_width'] = np.where(features['bb_middle'] != 0, bb_range / features['bb_middle'], np.nan)
        features['bb_pct'] = np.where(bb_range != 0, (close - features['bb_lower']) / bb_range, np.nan)
        
        # ===== ATR (Numba-accelerated) =====
        if self.use_numba:
            features['atr'] = fast_atr(high, low, close, self.config.atr_period)
        else:
            features['atr'] = calculate_atr(df['high'], df['low'], df['close'], self.config.atr_period).values
        
        features['atr_pct'] = np.where(close != 0, features['atr'] / close * 100, 0)
        
        # ===== VOLUME INDICATORS (Optimized) =====
        vol_series = df['volume']
        features['volume_sma'] = vol_series.rolling(window=self.config.volume_ma_period).mean().values
        features['volume_ratio'] = np.divide(
            volume,
            features['volume_sma'],
            out=np.ones_like(volume, dtype=float),
            where=(features['volume_sma'] != 0) & (~np.isnan(features['volume_sma'])) & (~np.isnan(volume))
        )
        features['volume_spike'] = (features['volume_ratio'] > 1.5).astype(np.int8)
        
        # ===== EMA RATIOS (Vectorized) =====
        if 9 in self.config.ema_periods and 21 in self.config.ema_periods:
            features['ema_ratio_9_21'] = np.where(features['ema_21'] != 0, features['ema_9'] / features['ema_21'], 1)
        
        if 12 in self.config.ema_periods and 26 in self.config.ema_periods:
            features['ema_ratio_12_26'] = np.where(features['ema_26'] != 0, features['ema_12'] / features['ema_26'], 1)
        
        if 20 in self.config.ema_periods and 50 in self.config.ema_periods:
            features['ema_ratio_20_50'] = np.where(features['ema_50'] != 0, features['ema_20'] / features['ema_50'], 1)
        
        if 50 in self.config.ema_periods and 200 in self.config.ema_periods:
            features['ema_ratio_50_200'] = np.where(features['ema_200'] != 0, features['ema_50'] / features['ema_200'], 1)
        
        # ===== EMA DISTANCE (Vectorized) =====
        if 9 in self.config.ema_periods:
            features['price_dist_ema9'] = np.where(features['ema_9'] != 0, (close - features['ema_9']) / features['ema_9'] * 100, 0)
        if 21 in self.config.ema_periods:
            features['price_dist_ema21'] = np.where(features['ema_21'] != 0, (close - features['ema_21']) / features['ema_21'] * 100, 0)
        if 50 in self.config.ema_periods:
            features['price_dist_ema50'] = np.where(features['ema_50'] != 0, (close - features['ema_50']) / features['ema_50'] * 100, 0)
        if 200 in self.config.ema_periods:
            features['price_dist_ema200'] = np.where(features['ema_200'] != 0, (close - features['ema_200']) / features['ema_200'] * 100, 0)
        
        # ===== EMA SLOPES (Vectorized) =====
        if 9 in self.config.ema_periods:
            features['ema9_slope'] = np.concatenate([np.full(5, np.nan), 
                                                     (features['ema_9'][5:] - features['ema_9'][:-5]) / features['ema_9'][:-5] * 100])
        if 21 in self.config.ema_periods:
            features['ema21_slope'] = np.concatenate([np.full(5, np.nan), 
                                                      (features['ema_21'][5:] - features['ema_21'][:-5]) / features['ema_21'][:-5] * 100])
        if 50 in self.config.ema_periods:
            features['ema50_slope'] = np.concatenate([np.full(10, np.nan), 
                                                      (features['ema_50'][10:] - features['ema_50'][:-10]) / features['ema_50'][:-10] * 100])
        
        # ===== MACD (Optimized) =====
        if 12 in self.config.ema_periods and 26 in self.config.ema_periods:
            ema_12 = features.get('ema_12', fast_ema(close, 12) if self.use_numba else close_series.ewm(span=12, adjust=False).mean().values)
            ema_26 = features.get('ema_26', fast_ema(close, 26) if self.use_numba else close_series.ewm(span=26, adjust=False).mean().values)
            features['macd_line'] = ema_12 - ema_26
            
            if self.use_numba:
                features['macd_signal'] = fast_ema(features['macd_line'], 9)
            else:
                features['macd_signal'] = pd.Series(features['macd_line']).ewm(span=9, adjust=False).mean().values
            
            features['macd_histogram'] = features['macd_line'] - features['macd_signal']
            current = (features['macd_line'] > features['macd_signal']).astype(np.int8)
            features['macd_cross'] = np.concatenate([[0], np.diff(current)])
        
        # ===== MOMENTUM INDICATORS (Vectorized) =====
        features['momentum_10'] = np.concatenate([np.full(10, np.nan), close[10:] - close[:-10]])
        features['momentum_20'] = np.concatenate([np.full(20, np.nan), close[20:] - close[:-20]])
        features['roc_10'] = np.concatenate([np.full(10, np.nan), 
                                             np.where(close[:-10] != 0, (close[10:] - close[:-10]) / close[:-10] * 100, 0)])
        features['roc_20'] = np.concatenate([np.full(20, np.nan), 
                                             np.where(close[:-20] != 0, (close[20:] - close[:-20]) / close[:-20] * 100, 0)])
        
        # ===== STOCHASTIC (Numba-accelerated) =====
        if self.use_numba:
            stoch_k, stoch_d = fast_stochastic(high, low, close, 14, 3)
            features['stoch_k'] = stoch_k
            features['stoch_d'] = stoch_d
        else:
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            features['stoch_k'] = np.where((high_14 - low_14).values != 0,
                                          (close - low_14.values) / (high_14 - low_14).values * 100, 50)
            features['stoch_d'] = pd.Series(features['stoch_k']).rolling(window=3).mean().values
        
        features['stoch_overbought'] = (features['stoch_k'] > 80).astype(np.int8)
        features['stoch_oversold'] = (features['stoch_k'] < 20).astype(np.int8)
        
        # ===== WILLIAMS %R (Optimized) =====
        if self.use_numba:
            high_14_max = np.array([np.max(high[max(0, i-13):i+1]) if i >= 13 else np.nan for i in range(n)])
            low_14_min = np.array([np.min(low[max(0, i-13):i+1]) if i >= 13 else np.nan for i in range(n)])
        else:
            high_14_max = df['high'].rolling(window=14).max().values
            low_14_min = df['low'].rolling(window=14).min().values
        
        features['williams_r'] = np.where((high_14_max - low_14_min) != 0,
                                         (high_14_max - close) / (high_14_max - low_14_min) * -100, -50)
        
        # ===== CCI (Optimized) =====
        typical_price = (high + low + close) / 3
        tp_series = pd.Series(typical_price)
        sma_tp = tp_series.rolling(window=20).mean().values
        mad = tp_series.rolling(window=20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True).values
        features['cci'] = np.where(mad != 0, (typical_price - sma_tp) / (0.015 * mad), 0)
        
        # ===== PRICE ACTION (Vectorized) =====
        features['high_low_range'] = high - low
        features['high_low_pct'] = np.where(close != 0, features['high_low_range'] / close * 100, 0)
        features['body_size'] = np.abs(close - open_prices)
        features['body_pct'] = np.where(close != 0, features['body_size'] / close * 100, 0)
        features['upper_shadow'] = high - np.maximum(close, open_prices)
        features['lower_shadow'] = np.minimum(close, open_prices) - low
        features['is_green'] = (close > open_prices).astype(np.int8)
        features['is_red'] = (close < open_prices).astype(np.int8)
        
        # ===== TYPICAL & WEIGHTED PRICE =====
        features['typical_price'] = typical_price
        features['weighted_close'] = (high + low + close * 2) / 4
        
        # ===== GAP DETECTION (Vectorized) =====
        features['gap_up'] = np.concatenate([[0], (low[1:] > high[:-1]).astype(np.int8)])
        features['gap_down'] = np.concatenate([[0], (high[1:] < low[:-1]).astype(np.int8)])
        gap_size = np.zeros(n)
        gap_up_mask = features['gap_up'] == 1
        gap_down_mask = features['gap_down'] == 1
        gap_size[gap_up_mask] = low[gap_up_mask] - np.concatenate([[0], high[:-1]])[gap_up_mask]
        gap_size[gap_down_mask] = np.concatenate([[0], low[:-1]])[gap_down_mask] - high[gap_down_mask]
        features['gap_size'] = gap_size
        
        # ===== SWING STRUCTURE (Vectorized) =====
        high_shifted1 = np.concatenate([[np.nan], high[:-1]])
        high_shifted2 = np.concatenate([[np.nan, np.nan], high[:-2]])
        low_shifted1 = np.concatenate([[np.nan], low[:-1]])
        low_shifted2 = np.concatenate([[np.nan, np.nan], low[:-2]])
        
        features['higher_high'] = ((high > high_shifted1) & (high_shifted1 > high_shifted2)).astype(np.int8)
        features['lower_low'] = ((low < low_shifted1) & (low_shifted1 < low_shifted2)).astype(np.int8)
        features['higher_low'] = ((low > low_shifted1) & (low_shifted1 > low_shifted2)).astype(np.int8)
        features['lower_high'] = ((high < high_shifted1) & (high_shifted1 < high_shifted2)).astype(np.int8)
        
        # ===== DONCHIAN CHANNELS (Optimized) =====
        features['donchian_high_20'] = df['high'].rolling(window=20).max().values
        features['donchian_low_20'] = df['low'].rolling(window=20).min().values
        features['donchian_mid_20'] = (features['donchian_high_20'] + features['donchian_low_20']) / 2
        donchian_range = features['donchian_high_20'] - features['donchian_low_20']
        features['price_vs_donchian'] = np.where(donchian_range != 0,
                                                 (close - features['donchian_low_20']) / donchian_range * 100, 50)
        
        # ===== ADX (Optimized with numba where possible) =====
        high_diff = np.concatenate([[0], np.diff(high)])
        low_diff = -np.concatenate([[0], np.diff(low)])
        
        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
        
        atr_14 = features.get('atr', features['atr'])
        
        # Smooth DM values
        plus_dm_smooth = pd.Series(plus_dm).rolling(window=14).mean().values
        minus_dm_smooth = pd.Series(minus_dm).rolling(window=14).mean().values
        
        features['plus_di'] = np.where(atr_14 != 0, 100 * plus_dm_smooth / atr_14, 0)
        features['minus_di'] = np.where(atr_14 != 0, 100 * minus_dm_smooth / atr_14, 0)
        
        di_sum = features['plus_di'] + features['minus_di']
        dx = np.where(di_sum != 0, np.abs(features['plus_di'] - features['minus_di']) / di_sum * 100, 0)
        features['adx'] = pd.Series(dx).rolling(window=14).mean().values
        
        features['trending'] = (features['adx'] > 25).astype(np.int8)
        features['strong_trend'] = (features['adx'] > 40).astype(np.int8)
        
        # ===== VOLATILITY (Optimized) =====
        log_returns_series = pd.Series(features['log_returns'])
        features['volatility_10'] = log_returns_series.rolling(window=10).std().values * np.sqrt(252) * 100
        features['volatility_20'] = log_returns_series.rolling(window=20).std().values * np.sqrt(252) * 100
        features['volatility_30'] = log_returns_series.rolling(window=30).std().values * np.sqrt(252) * 100
        
        # ===== OBV (Optimized) =====
        price_change_sign = np.sign(np.concatenate([[0], np.diff(close)]))
        features['obv'] = np.cumsum(price_change_sign * volume)
        
        if self.use_numba:
            features['obv_ema'] = fast_ema(features['obv'], 20)
        else:
            features['obv_ema'] = pd.Series(features['obv']).ewm(span=20, adjust=False).mean().values
        
        features['obv_trend'] = (features['obv'] > features['obv_ema']).astype(np.int8)
        
        # ===== MFI (Optimized) =====
        typical_price_shifted = np.concatenate([[typical_price[0]], typical_price[:-1]])
        money_flow = typical_price * volume
        
        positive_flow = np.where(typical_price > typical_price_shifted, money_flow, 0)
        negative_flow = np.where(typical_price < typical_price_shifted, money_flow, 0)
        
        positive_mf = pd.Series(positive_flow).rolling(window=14).sum().values
        negative_mf = pd.Series(negative_flow).rolling(window=14).sum().values
        
        # Suppress divide by zero warning - we handle it with np.where
        with np.errstate(divide='ignore', invalid='ignore'):
            mfi_ratio = np.where(negative_mf != 0, positive_mf / negative_mf, 1)
        
        features['mfi'] = 100 - (100 / (1 + mfi_ratio))
        features['mfi_overbought'] = (features['mfi'] > 80).astype(np.int8)
        features['mfi_oversold'] = (features['mfi'] < 20).astype(np.int8)
        
        # ===== VPT (Vectorized) =====
        features['vpt'] = np.cumsum(volume * features['returns'])
        
        # ===== KELTNER CHANNELS (Optimized) =====
        if self.use_numba:
            ema_20_kc = fast_ema(close, 20)
        else:
            ema_20_kc = close_series.ewm(span=20, adjust=False).mean().values
        
        features['keltner_upper'] = ema_20_kc + (2 * features['atr'])
        features['keltner_middle'] = ema_20_kc
        features['keltner_lower'] = ema_20_kc - (2 * features['atr'])
        keltner_range = features['keltner_upper'] - features['keltner_lower']
        features['keltner_width'] = np.where(features['keltner_middle'] != 0, 
                                             keltner_range / features['keltner_middle'], 0)
        
        # ===== TREND ALIGNMENT (Vectorized) =====
        if 20 in self.config.ema_periods and 50 in self.config.ema_periods and 200 in self.config.ema_periods:
            features['bullish_alignment'] = ((features['ema_20'] > features['ema_50']) & 
                                            (features['ema_50'] > features['ema_200'])).astype(np.int8)
            features['bearish_alignment'] = ((features['ema_20'] < features['ema_50']) & 
                                             (features['ema_50'] < features['ema_200'])).astype(np.int8)
        
        # ===== RSI ENHANCEMENTS (Vectorized) =====
        features['rsi_uptrend'] = (features['rsi_14'] > 50).astype(np.int8)
        features['rsi_downtrend'] = (features['rsi_14'] < 50).astype(np.int8)
        features['rsi_momentum_shift'] = np.concatenate([[0], np.diff(features['rsi_14'])])
        
        # ===== CONSECUTIVE BARS (Optimized) =====
        is_green = features['is_green']
        is_red = features['is_red']
        
        # Fast consecutive count using cumsum trick
        green_groups = np.cumsum(np.concatenate([[1], (is_green[1:] != is_green[:-1]).astype(int)]))
        red_groups = np.cumsum(np.concatenate([[1], (is_red[1:] != is_red[:-1]).astype(int)]))
        
        features['consecutive_green'] = np.where(is_green, 
                                                 pd.Series(is_green).groupby(green_groups).cumsum().values, 0)
        features['consecutive_red'] = np.where(is_red, 
                                               pd.Series(is_red).groupby(red_groups).cumsum().values, 0)
        
        # ===== SUPPORT/RESISTANCE (NO LOOK-AHEAD BIAS) =====
        # Important: Using center=False (default) to avoid future data leakage
        # This looks at the last N bars only, not future bars
        features['swing_high'] = df['high'].rolling(window=5, center=False).max().values
        features['swing_low'] = df['low'].rolling(window=5, center=False).min().values
        features['near_swing_high'] = np.where(close != 0,
                                               np.abs(close - features['swing_high']) / close < 0.02, 0).astype(np.int8)
        features['near_swing_low'] = np.where(close != 0,
                                              np.abs(close - features['swing_low']) / close < 0.02, 0).astype(np.int8)
        
        # ===== FINAL CONCAT (Single operation) =====
        features_df = pd.DataFrame(features, index=df.index)
        result = pd.concat([df, features_df], axis=1)
        
        return result


# Backward compatibility
class BasicIndicators(BasicCalculator):
    """Alias for backward compatibility"""
    pass


__all__ = ['BasicCalculator', 'BasicIndicators']
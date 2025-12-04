# config/features.py - COMPLETE feature configuration with ALL features

from dataclasses import dataclass, field
from typing import List


@dataclass
class IndicatorConfig:
    """Technical indicator parameters"""
    
    # Moving averages
    ema_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 200])
    sma_periods: List[int] = field(default_factory=lambda: [10, 20, 50, 100, 200])
    
    # Momentum indicators
    rsi_period: int = 14
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    
    # Volatility
    atr_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    
    # MACD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    
    # Volume
    volume_ma_period: int = 20
    
    # Trend
    adx_period: int = 14
    
    # Support/Resistance
    sr_lookback: int = 20
    sr_num_levels: int = 3


@dataclass
class FilterConfig:
    """Stock filtering criteria"""
    
    min_avg_volume: float = 0.1
    min_price: float = 50.0
    min_historical_days: int = 50
    max_price: float = 100000.0


@dataclass
class SwingConfig:
    """Swing trading parameters"""
    
    swing_lookback: int = 5
    swing_threshold: float = 0.02
    
    min_hold_days: int = 2
    max_hold_days: int = 20
    
    target_pct: float = 0.15
    stop_loss_pct: float = 0.05
    
    max_position_size: float = 0.10
    max_positions: int = 10
    
    max_daily_loss: float = 0.02
    trailing_stop: bool = True
    trailing_stop_pct: float = 0.02


# Global configuration instances
INDICATOR_CONFIG = IndicatorConfig()
FILTER_CONFIG = FilterConfig()
SWING_CONFIG = SwingConfig()


# ============================================================================
# COMPLETE FEATURE LIST - ALL 386+ FEATURES
# ============================================================================

# Basic price features
BASIC_PRICE_FEATURES = [
    'returns', 'log_returns', 'price_change_pct',
    'log_price', 'price_to_high', 'price_to_low', 'hl_ratio', 'close_to_open',
    'typical_price', 'weighted_close', 'median_price',
]

# Moving averages
MOVING_AVERAGE_FEATURES = [
    # SMA
    'sma_10', 'sma_20', 'sma_50', 'sma_100', 'sma_200',
    
    # EMA
    'ema_5', 'ema_10', 'ema_20', 'ema_50', 'ema_200',
    
    # Price vs MA
    'price_above_sma20', 'price_above_sma50', 'price_above_sma200',
    'price_above_ema20', 'price_above_ema50', 'price_above_ema200',
    
    # MA crossovers
    'sma20_50_cross', 'sma50_200_cross', 'ema20_50_cross',
    
    # MA ratios
    'ema_ratio_20_50', 'ema_ratio_50_200',
    'price_dist_ema50', 'price_dist_ema200',
    'ema50_slope',
    'ema50_ema200', 'sma50_sma200', 'ema200_price', 'sma200_price',
]

# RSI features
RSI_FEATURES = [
    'rsi_14', 'rsi_overbought', 'rsi_oversold',
    'rsi_7', 'rsi_7_oversold', 'rsi_7_overbought',
    'rsi_14_oversold', 'rsi_14_overbought',
    'rsi_21', 'rsi_21_oversold', 'rsi_21_overbought',
    'rsi_uptrend', 'rsi_downtrend', 'rsi_momentum_shift',
    'bearish_divergence_rsi', 'bullish_divergence_rsi',
]

# Stochastic features
STOCHASTIC_FEATURES = [
    'stoch_k', 'stoch_d', 'stoch_overbought', 'stoch_oversold',
    'stoch_k_14', 'stoch_d_14', 'stoch_signal_14',
    'stoch_k_21', 'stoch_d_21', 'stoch_signal_21',
]

# Williams %R
WILLIAMS_FEATURES = [
    'williams_r', 'williams_r_14', 'williams_r_28',
]

# CCI features
CCI_FEATURES = [
    'cci', 'cci_14', 'cci_14_extreme', 'cci_20', 'cci_20_extreme',
]

# Momentum features
MOMENTUM_FEATURES = [
    'momentum_5', 'momentum_10', 'momentum_20', 'momentum_50',
    'roc_5', 'roc_10', 'roc_20', 'roc_50',
    'momentum_score',
]

# Volatility features
VOLATILITY_FEATURES = [
    'volatility_10', 'volatility_20', 'volatility_30', 'volatility_50',
    'atr', 'atr_pct',
    'atr_14', 'atr_percent_14', 'atr_20', 'atr_percent_20',
    'chaikin_volatility',
    'volatility_score',
]

# Bollinger Bands
BOLLINGER_FEATURES = [
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_pct',
    'bb_upper_20', 'bb_middle_20', 'bb_lower_20', 'bb_width_20',
    'bb_width_ratio_20', 'bb_position_20', 'bb_squeeze_20',
    'bb_upper_50', 'bb_middle_50', 'bb_lower_50', 'bb_width_50',
    'bb_width_ratio_50', 'bb_position_50', 'bb_squeeze_50',
]

# Keltner Channels
KELTNER_FEATURES = [
    'keltner_upper', 'keltner_middle', 'keltner_lower', 'keltner_width',
    'kc_upper_20', 'kc_middle_20', 'kc_lower_20', 'kc_position_20',
]

# Donchian Channels
DONCHIAN_FEATURES = [
    'donchian_high_20', 'donchian_low_20', 'donchian_mid_20', 'price_vs_donchian',
    'dc_upper_20', 'dc_lower_20', 'dc_middle_20', 'dc_position_20',
    'dc_upper_50', 'dc_lower_50', 'dc_middle_50', 'dc_position_50',
]

# ADX and DI features
ADX_FEATURES = [
    'adx', 'plus_di', 'minus_di', 'trending', 'strong_trend',
    'adx_14', 'plus_di_14', 'minus_di_14', 'di_diff_14',
    'adx_20', 'plus_di_20', 'minus_di_20', 'di_diff_20',
]

# Aroon
AROON_FEATURES = [
    'aroon_up_25', 'aroon_down_25', 'aroon_oscillator_25',
]

# Parabolic SAR
SAR_FEATURES = [
    'sar', 'sar_signal',
]

# Trend features
TREND_FEATURES = [
    'trend_strength_10', 'trend_direction_10',
    'trend_strength_20', 'trend_direction_20',
    'trend_strength_50', 'trend_direction_50',
    'trend_strength_score', 'is_trending', 'is_ranging',
]

# Supertrend
SUPERTREND_FEATURES = [
    'supertrend_10', 'supertrend_signal_10',
    'supertrend_20', 'supertrend_signal_20',
]

# Volume features
VOLUME_FEATURES = [
    'volume_sma', 'volume_ratio', 'volume_spike',
    'obv', 'obv_ema', 'obv_trend',
    'obv_sma_20',
    'mfi', 'mfi_overbought', 'mfi_oversold',
    'mfi_14', 'mfi_14_oversold', 'mfi_14_overbought',
    'vpt', 'vpt_sma_20', 'vpt_signal',
    'cmf_20',
    'vwap', 'price_to_vwap', 'vwap_signal',
    'vroc_10', 'vroc_20',
    'ad_line', 'ad_line_sma_20', 'ad_line_trend',
    'force_index_13', 'force_index_ema_13',
    'force_index_20', 'force_index_ema_20',
    'ease_of_movement',
    'volume_sma_20', 'volume_sma_50',
    'volume_trend', 'high_volume', 'low_volume',
    'bullish_volume_confirm', 'bearish_volume_confirm',
    'accumulation', 'distribution',
    'price_volume_divergence',
    'price_volume_corr_20', 'price_volume_corr_50',
    'volume_score',
]

# MACD features
MACD_FEATURES = [
    'macd', 'macd_signal', 'macd_hist',
    'macd_cross_up', 'macd_cross_down', 'macd_divergence',
]

# Ultimate Oscillator
ULTIMATE_OSC_FEATURES = [
    'ultimate_oscillator',
]

# Candlestick patterns - basic
CANDLESTICK_BASIC = [
    'body_size', 'body_pct', 'upper_shadow', 'lower_shadow',
    'is_green', 'is_red',
    'high_low_range', 'high_low_pct',
    'total_range', 'body_relative', 'upper_shadow_relative',
    'lower_shadow_relative', 'body_center', 'body_position',
    'close_position', 'gap_pct',
    'gap_up', 'gap_down', 'gap_size',
]

# Candlestick patterns - single candle
CANDLESTICK_SINGLE = [
    'doji', 'dragonfly_doji', 'gravestone_doji',
    'hammer', 'inverted_hammer', 'shooting_star',
    'marubozu_bullish', 'marubozu_bearish',
    'spinning_top', 'long_legged_doji',
]

# Candlestick patterns - multi-candle
CANDLESTICK_MULTI = [
    'engulfing_bullish', 'engulfing_bearish',
    'harami_bullish', 'harami_bearish',
    'piercing_pattern', 'dark_cloud_cover',
    'tweezer_top', 'tweezer_bottom',
    'morning_star', 'evening_star',
    'three_white_soldiers', 'three_black_crows',
    'three_inside_up', 'three_inside_down',
    'abandoned_baby_bullish', 'abandoned_baby_bearish',
]

# Candlestick patterns - advanced
CANDLESTICK_ADVANCED = [
    'inside_bar', 'outside_bar',
    'pin_bar_bullish', 'pin_bar_bearish',
    'strong_bullish_candle', 'strong_bearish_candle',
    'rejection_zone_top', 'rejection_zone_bottom',
    'key_reversal_bullish', 'key_reversal_bearish',
]

# Pattern analysis
PATTERN_FEATURES = [
    'bullish_score', 'bearish_score', 'pattern_signal',
    'pattern_density_3', 'pattern_density_5',
    'strong_bullish_reversal', 'strong_bearish_reversal',
    'bullish_continuation', 'bearish_continuation',
]

# Market structure
MARKET_STRUCTURE = [
    'higher_high', 'lower_low', 'higher_low', 'lower_high',
    'consecutive_green', 'consecutive_red',
    'candles_above_ma',
    'range_expansion', 'range_contraction',
]

# Alignment features
ALIGNMENT_FEATURES = [
    'bullish_alignment', 'bearish_alignment',
]

# Support and Resistance
SUPPORT_RESISTANCE_FEATURES = [
    'support_level', 'resistance_level',
    'dist_to_support', 'dist_to_resistance',
    'near_support', 'near_resistance',
    'resistance_20', 'support_20', 'sr_range_20', 'sr_position_20',
    'resistance_50', 'support_50', 'sr_range_50', 'sr_position_50',
]

# Pivot points
PIVOT_FEATURES = [
    'pivot', 'r1', 's1', 'r2', 's2', 'pivot_signal',
]

# Fibonacci levels
FIBONACCI_FEATURES = [
    'fib_0_50', 'fib_236_50', 'fib_382_50', 'fib_500_50', 'fib_618_50', 'fib_1000_50',
    'fib_0_100', 'fib_236_100', 'fib_382_100', 'fib_500_100', 'fib_618_100', 'fib_1000_100',
    'near_fib_382_20', 'near_fib_618_20',
    'near_fib_382_50', 'near_fib_618_50',
]

# Swing features
# SWING_FEATURES = [
#     'swing_high', 'swing_low',
#     'near_swing_high', 'near_swing_low',
# ]

# Statistical features
STATISTICAL_FEATURES = [
    'price_zscore', 'volume_zscore',
    'price_percentile', 'volume_percentile',
    'zscore_20', 'zscore_extreme_20',
    'zscore_50', 'zscore_extreme_50',
    'skew_20', 'kurtosis_20',
    'autocorr_1', 'autocorr_5', 'autocorr_10',
]

# Ichimoku
ICHIMOKU_FEATURES = [
    'ichimoku_conversion', 'ichimoku_base',
    'ichimoku_span_a', 'ichimoku_span_b',
    'ichimoku_cloud_top', 'ichimoku_cloud_bottom',
    'price_above_cloud', 'price_in_cloud',
]

# Elliott Wave
ELLIOTT_WAVE_FEATURES = [
    'potential_wave_3', 'potential_wave_5',
]

# Market phase
MARKET_PHASE_FEATURES = [
    'market_phase', 'tech_rating',
]

# Signal strength
SIGNAL_FEATURES = [
    'buy_signal_strength', 'sell_signal_strength', 'risk_score',
]

# ============================================================================
# COMBINED FEATURE LIST - ALL FEATURES
# ============================================================================

FEATURE_COLUMNS = (
    BASIC_PRICE_FEATURES +
    MOVING_AVERAGE_FEATURES +
    RSI_FEATURES +
    STOCHASTIC_FEATURES +
    WILLIAMS_FEATURES +
    CCI_FEATURES +
    MOMENTUM_FEATURES +
    VOLATILITY_FEATURES +
    BOLLINGER_FEATURES +
    KELTNER_FEATURES +
    DONCHIAN_FEATURES +
    ADX_FEATURES +
    AROON_FEATURES +
    SAR_FEATURES +
    TREND_FEATURES +
    SUPERTREND_FEATURES +
    VOLUME_FEATURES +
    MACD_FEATURES +
    ULTIMATE_OSC_FEATURES +
    CANDLESTICK_BASIC +
    CANDLESTICK_SINGLE +
    CANDLESTICK_MULTI +
    CANDLESTICK_ADVANCED +
    PATTERN_FEATURES +
    MARKET_STRUCTURE +
    ALIGNMENT_FEATURES +
    SUPPORT_RESISTANCE_FEATURES +
    PIVOT_FEATURES +
    FIBONACCI_FEATURES +
    # SWING_FEATURES +
    STATISTICAL_FEATURES +
    ICHIMOKU_FEATURES +
    ELLIOTT_WAVE_FEATURES +
    MARKET_PHASE_FEATURES +
    SIGNAL_FEATURES
)

print(f"✅ Total features configured: {len(FEATURE_COLUMNS)}")

# Export all
__all__ = [
    'IndicatorConfig',
    'FilterConfig',
    'SwingConfig',
    'INDICATOR_CONFIG',
    'FILTER_CONFIG',
    'SWING_CONFIG',
    'FEATURE_COLUMNS',
]
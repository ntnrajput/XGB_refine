# features/calculators/support_resistance.py

import pandas as pd
import numpy as np
from typing import Optional

from config.features import IndicatorConfig, INDICATOR_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class SupportResistanceCalculator:
    """Calculate support and resistance levels"""
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or INDICATOR_CONFIG
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add support/resistance features"""
        df = df.copy()
        
        lookback = self.config.sr_lookback
        
        # ==================== BASIC SUPPORT/RESISTANCE ====================
        # Simple support/resistance based on rolling min/max
        df['support_level'] = df['low'].rolling(lookback, min_periods=1).min()
        df['resistance_level'] = df['high'].rolling(lookback, min_periods=1).max()
        
        # Distance to levels
        df['dist_to_support'] = (df['close'] - df['support_level']) / df['close'] * 100
        df['dist_to_resistance'] = (df['resistance_level'] - df['close']) / df['close'] * 100
        
        # Near level flags (within 2%)
        df['near_support'] = (df['dist_to_support'] < 2).astype(int)
        df['near_resistance'] = (df['dist_to_resistance'] < 2).astype(int)
        
        # # ==================== SWING HIGHS/LOWS ====================
        # df = self._add_swing_points(df, window=5)
        
        # # ==================== PIVOT POINTS ====================
        # df = self._add_pivot_points(df)
        
        # # ==================== FIBONACCI RETRACEMENT ====================
        # df = self._add_fibonacci_levels(df, lookback)
        
        # # ==================== SUPPORT/RESISTANCE STRENGTH ====================
        # df = self._add_level_strength(df, lookback)
        
        # # ==================== VOLUME-WEIGHTED LEVELS ====================
        # df = self._add_volume_weighted_levels(df, lookback)
        
        # # ==================== BREAKOUT DETECTION ====================
        # df = self._add_breakout_signals(df)
        
        # # ==================== ZONE-BASED SUPPORT/RESISTANCE ====================
        # df = self._add_support_resistance_zones(df, lookback)
        
        # # ==================== PSYCHOLOGICAL LEVELS ====================
        # df = self._add_psychological_levels(df)
        
        # # ==================== DYNAMIC SUPPORT/RESISTANCE ====================
        # df = self._add_dynamic_levels(df)
        
        # # ==================== SUPPORT/RESISTANCE FLIP ====================
        # df = self._add_sr_flip_detection(df)
        
        # # ==================== MULTIPLE TIMEFRAME LEVELS ====================
        # df = self._add_multi_timeframe_levels(df, lookback)
        
        return df
    
    def _add_swing_points(self, df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
        """
        Detect swing highs and lows WITHOUT data leakage
        Uses only historical data - checks if current bar is highest/lowest in last N bars
        """
        # Initialize columns
        df['swing_high'] = 0.0
        df['is_swing_high'] = 0
        df['swing_low'] = 0.0
        df['is_swing_low'] = 0
        
        # Swing high: current high equals the rolling max of last 'window' bars (including current)
        rolling_high = df['high'].rolling(window=window, min_periods=window).max()
        is_swing_high = (df['high'] == rolling_high) & (df['high'] == df['high'].rolling(window, min_periods=window).max())
        
        # Additional confirmation: ensure it's not just the start of the window
        # Check that high is greater than previous bar's high for more robust detection
        is_swing_high = is_swing_high & (df['high'] >= df['high'].shift(1))
        
        df.loc[is_swing_high, 'swing_high'] = df.loc[is_swing_high, 'high']
        df.loc[is_swing_high, 'is_swing_high'] = 1
        
        # Swing low: current low equals the rolling min of last 'window' bars (including current)
        rolling_low = df['low'].rolling(window=window, min_periods=window).min()
        is_swing_low = (df['low'] == rolling_low) & (df['low'] == df['low'].rolling(window, min_periods=window).min())
        
        # Additional confirmation: ensure it's not just the start of the window
        is_swing_low = is_swing_low & (df['low'] <= df['low'].shift(1))
        
        df.loc[is_swing_low, 'swing_low'] = df.loc[is_swing_low, 'low']
        df.loc[is_swing_low, 'is_swing_low'] = 1
        
        # Forward fill swing points to get "last known" swing point
        # This is safe because we're filling with PAST data only
        df['last_swing_high'] = df['swing_high'].replace(0, np.nan).fillna(method='ffill')
        df['last_swing_low'] = df['swing_low'].replace(0, np.nan).fillna(method='ffill')
        
        # Distance to last swing points
        df['dist_to_swing_high'] = ((df['last_swing_high'] - df['close']) / df['close'] * 100).fillna(0)
        df['dist_to_swing_low'] = ((df['close'] - df['last_swing_low']) / df['close'] * 100).fillna(0)
        
        return df
    
    def _add_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate classic, Fibonacci, and Camarilla pivot points using PREVIOUS period data only"""
        # Get previous period's high, low, close (shifted by 1 = looking back)
        df['prev_high'] = df['high'].shift(1)
        df['prev_low'] = df['low'].shift(1)
        df['prev_close'] = df['close'].shift(1)
        
        # Classic Pivot Points
        df['pivot'] = (df['prev_high'] + df['prev_low'] + df['prev_close']) / 3
        df['pivot_r1'] = 2 * df['pivot'] - df['prev_low']
        df['pivot_r2'] = df['pivot'] + (df['prev_high'] - df['prev_low'])
        df['pivot_r3'] = df['prev_high'] + 2 * (df['pivot'] - df['prev_low'])
        df['pivot_s1'] = 2 * df['pivot'] - df['prev_high']
        df['pivot_s2'] = df['pivot'] - (df['prev_high'] - df['prev_low'])
        df['pivot_s3'] = df['prev_low'] - 2 * (df['prev_high'] - df['pivot'])
        
        # Fibonacci Pivot Points
        df['fib_r1'] = df['pivot'] + 0.382 * (df['prev_high'] - df['prev_low'])
        df['fib_r2'] = df['pivot'] + 0.618 * (df['prev_high'] - df['prev_low'])
        df['fib_r3'] = df['pivot'] + 1.000 * (df['prev_high'] - df['prev_low'])
        df['fib_s1'] = df['pivot'] - 0.382 * (df['prev_high'] - df['prev_low'])
        df['fib_s2'] = df['pivot'] - 0.618 * (df['prev_high'] - df['prev_low'])
        df['fib_s3'] = df['pivot'] - 1.000 * (df['prev_high'] - df['prev_low'])
        
        # Camarilla Pivot Points
        range_val = df['prev_high'] - df['prev_low']
        df['cam_r1'] = df['prev_close'] + range_val * 1.1 / 12
        df['cam_r2'] = df['prev_close'] + range_val * 1.1 / 6
        df['cam_r3'] = df['prev_close'] + range_val * 1.1 / 4
        df['cam_r4'] = df['prev_close'] + range_val * 1.1 / 2
        df['cam_s1'] = df['prev_close'] - range_val * 1.1 / 12
        df['cam_s2'] = df['prev_close'] - range_val * 1.1 / 6
        df['cam_s3'] = df['prev_close'] - range_val * 1.1 / 4
        df['cam_s4'] = df['prev_close'] - range_val * 1.1 / 2
        
        # Distance to pivot points (%)
        df['dist_to_pivot'] = ((df['close'] - df['pivot']) / df['close'] * 100).fillna(0)
        df['dist_to_pivot_r1'] = ((df['pivot_r1'] - df['close']) / df['close'] * 100).fillna(0)
        df['dist_to_pivot_s1'] = ((df['close'] - df['pivot_s1']) / df['close'] * 100).fillna(0)
        
        # Near pivot flags
        df['near_pivot'] = (abs(df['dist_to_pivot']) < 1).astype(int)
        df['near_pivot_r1'] = (abs(df['dist_to_pivot_r1']) < 1).astype(int)
        df['near_pivot_s1'] = (abs(df['dist_to_pivot_s1']) < 1).astype(int)
        
        return df
    
    def _add_fibonacci_levels(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate Fibonacci retracement levels using historical data only"""
        # Get swing high/low over lookback period (looking back only)
        swing_high = df['high'].rolling(lookback, min_periods=1).max()
        swing_low = df['low'].rolling(lookback, min_periods=1).min()
        swing_range = swing_high - swing_low
        
        # Fibonacci retracement levels (from high to low)
        df['fib_0'] = swing_high
        df['fib_236'] = swing_high - 0.236 * swing_range
        df['fib_382'] = swing_high - 0.382 * swing_range
        df['fib_500'] = swing_high - 0.500 * swing_range
        df['fib_618'] = swing_high - 0.618 * swing_range
        df['fib_786'] = swing_high - 0.786 * swing_range
        df['fib_100'] = swing_low
        
        # Fibonacci extension levels
        df['fib_1272'] = swing_low - 0.272 * swing_range
        df['fib_1618'] = swing_low - 0.618 * swing_range
        
        # Distance to key Fibonacci levels
        df['dist_to_fib_382'] = ((df['fib_382'] - df['close']) / df['close'] * 100).fillna(0)
        df['dist_to_fib_500'] = ((df['fib_500'] - df['close']) / df['close'] * 100).fillna(0)
        df['dist_to_fib_618'] = ((df['fib_618'] - df['close']) / df['close'] * 100).fillna(0)
        
        # Near Fibonacci level flags
        df['near_fib_382'] = (abs(df['dist_to_fib_382']) < 1).astype(int)
        df['near_fib_500'] = (abs(df['dist_to_fib_500']) < 1).astype(int)
        df['near_fib_618'] = (abs(df['dist_to_fib_618']) < 1).astype(int)
        
        return df
    
    def _add_level_strength(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """
        Calculate support/resistance strength based on touches
        FIXED: Uses only historical data by looking back from current position
        """
        tolerance = 0.02  # 2% tolerance for level touch
        
        # Initialize columns
        df['support_touches'] = 0
        df['resistance_touches'] = 0
        
        # Vectorized approach to avoid data leakage
        for i in range(len(df)):
            if i < lookback:
                continue
                
            # Only look at data BEFORE and INCLUDING current bar
            support_level = df.iloc[i]['support_level']
            resistance_level = df.iloc[i]['resistance_level']
            
            # Count touches in lookback window (including current bar, all historical)
            window_lows = df.iloc[max(0, i-lookback):i+1]['low']  # i+1 to include current bar
            window_highs = df.iloc[max(0, i-lookback):i+1]['high']
            
            if support_level > 0:
                support_touches = ((abs(window_lows - support_level) / support_level) < tolerance).sum()
                df.iloc[i, df.columns.get_loc('support_touches')] = support_touches
            
            if resistance_level > 0:
                resistance_touches = ((abs(window_highs - resistance_level) / resistance_level) < tolerance).sum()
                df.iloc[i, df.columns.get_loc('resistance_touches')] = resistance_touches
        
        # Normalize strength (0-1 scale)
        df['support_strength'] = df['support_touches'] / lookback
        df['resistance_strength'] = df['resistance_touches'] / lookback
        
        # Strong level flags (>=3 touches)
        df['strong_support'] = (df['support_touches'] >= 3).astype(int)
        df['strong_resistance'] = (df['resistance_touches'] >= 3).astype(int)
        
        return df
    
    def _add_volume_weighted_levels(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate volume-weighted support/resistance levels using historical data"""
        # Volume-weighted average price in lookback window (looking back only)
        df['vwap'] = (
            (df['close'] * df['volume']).rolling(lookback, min_periods=1).sum() / 
            df['volume'].rolling(lookback, min_periods=1).sum()
        )
        
        # Volume-weighted support (weighted by volume at lows)
        df['vol_weighted_support'] = (
            (df['low'] * df['volume']).rolling(lookback, min_periods=1).sum() / 
            df['volume'].rolling(lookback, min_periods=1).sum()
        )
        
        # Volume-weighted resistance (weighted by volume at highs)
        df['vol_weighted_resistance'] = (
            (df['high'] * df['volume']).rolling(lookback, min_periods=1).sum() / 
            df['volume'].rolling(lookback, min_periods=1).sum()
        )
        
        # Distance to volume-weighted levels
        df['dist_to_vwap'] = ((df['close'] - df['vwap']) / df['close'] * 100).fillna(0)
        df['dist_to_vol_support'] = ((df['close'] - df['vol_weighted_support']) / df['close'] * 100).fillna(0)
        df['dist_to_vol_resistance'] = ((df['vol_weighted_resistance'] - df['close']) / df['close'] * 100).fillna(0)
        
        # Price position relative to VWAP
        df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
        df['below_vwap'] = (df['close'] < df['vwap']).astype(int)
        
        return df
    
    def _add_breakout_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect breakouts above resistance and below support
        Uses shift(1) to compare with PREVIOUS bar only
        """
        # Resistance breakout (close above resistance compared to previous bar)
        df['resistance_breakout'] = (
            (df['close'] > df['resistance_level']) & 
            (df['close'].shift(1) <= df['resistance_level'].shift(1))
        ).astype(int)
        
        # Support breakdown (close below support compared to previous bar)
        df['support_breakdown'] = (
            (df['close'] < df['support_level']) & 
            (df['close'].shift(1) >= df['support_level'].shift(1))
        ).astype(int)
        
        # Volume confirmation (breakout with above-average volume)
        # Average volume calculated from historical data only
        avg_volume = df['volume'].rolling(20, min_periods=1).mean()
        df['resistance_breakout_volume'] = (
            df['resistance_breakout'] & 
            (df['volume'] > avg_volume)
        ).astype(int)
        df['support_breakdown_volume'] = (
            df['support_breakdown'] & 
            (df['volume'] > avg_volume)
        ).astype(int)
        
        # False breakout detection (price returns within level on NEXT bar)
        # We can only detect this on the bar AFTER the breakout
        df['false_breakout_up'] = (
            (df['resistance_breakout'].shift(1) == 1) & 
            (df['close'] < df['resistance_level'])
        ).astype(int)
        df['false_breakout_down'] = (
            (df['support_breakdown'].shift(1) == 1) & 
            (df['close'] > df['support_level'])
        ).astype(int)
        
        # Breakout strength (distance moved after breakout)
        df['breakout_strength'] = 0.0
        df.loc[df['resistance_breakout'] == 1, 'breakout_strength'] = df['dist_to_resistance']
        df.loc[df['support_breakdown'] == 1, 'breakout_strength'] = -df['dist_to_support']
        
        return df
    
    def _add_support_resistance_zones(self, df: pd.DataFrame, lookback: int) -> pd.DataFrame:
        """Calculate support/resistance zones instead of exact levels"""
        # Support zone (band around support level)
        zone_width = 0.015  # 1.5% zone width
        df['support_zone_upper'] = df['support_level'] * (1 + zone_width)
        df['support_zone_lower'] = df['support_level'] * (1 - zone_width)
        
        # Resistance zone (band around resistance level)
        df['resistance_zone_upper'] = df['resistance_level'] * (1 + zone_width)
        df['resistance_zone_lower'] = df['resistance_level'] * (1 - zone_width)
        
        # Price in zone flags (current bar only)
        df['in_support_zone'] = (
            (df['close'] >= df['support_zone_lower']) & 
            (df['close'] <= df['support_zone_upper'])
        ).astype(int)
        df['in_resistance_zone'] = (
            (df['close'] >= df['resistance_zone_lower']) & 
            (df['close'] <= df['resistance_zone_upper'])
        ).astype(int)
        
        # Rejection from zone (was in zone on previous bar, exited in opposite direction on current bar)
        df['support_zone_bounce'] = (
            (df['in_support_zone'].shift(1) == 1) & 
            (df['close'] > df['support_zone_upper']) &
            (df['close'].shift(1) <= df['support_zone_upper'].shift(1))
        ).astype(int)
        
        df['resistance_zone_rejection'] = (
            (df['in_resistance_zone'].shift(1) == 1) & 
            (df['close'] < df['resistance_zone_lower']) &
            (df['close'].shift(1) >= df['resistance_zone_lower'].shift(1))
        ).astype(int)
        
        return df
    
    def _add_psychological_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Identify proximity to psychological/round number levels"""
        # Find nearest round numbers (10, 50, 100, etc.)
        df['nearest_round_10'] = (df['close'] / 10).round() * 10
        df['nearest_round_50'] = (df['close'] / 50).round() * 50
        df['nearest_round_100'] = (df['close'] / 100).round() * 100
        
        # Distance to psychological levels
        df['dist_to_round_10'] = abs((df['close'] - df['nearest_round_10']) / df['close'] * 100)
        df['dist_to_round_50'] = abs((df['close'] - df['nearest_round_50']) / df['close'] * 100)
        df['dist_to_round_100'] = abs((df['close'] - df['nearest_round_100']) / df['close'] * 100)
        
        # Near psychological level flags (within 0.5%)
        df['near_psychological_10'] = (df['dist_to_round_10'] < 0.5).astype(int)
        df['near_psychological_50'] = (df['dist_to_round_50'] < 0.5).astype(int)
        df['near_psychological_100'] = (df['dist_to_round_100'] < 0.5).astype(int)
        
        return df
    
    def _add_dynamic_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate dynamic support/resistance using moving averages"""
        # Common MA periods used as dynamic S/R
        periods = [20, 50, 200]
        
        for period in periods:
            ma_col = f'sma_{period}'
            # Rolling mean looks backward only
            df[ma_col] = df['close'].rolling(period, min_periods=1).mean()
            
            # Distance to MA
            df[f'dist_to_sma_{period}'] = ((df['close'] - df[ma_col]) / df['close'] * 100).fillna(0)
            
            # Price above/below MA
            df[f'above_sma_{period}'] = (df['close'] > df[ma_col]).astype(int)
            
            # MA acting as support/resistance (on current bar)
            df[f'sma_{period}_support'] = (
                (df['close'] > df[ma_col]) & 
                (df['low'] <= df[ma_col])
            ).astype(int)
            df[f'sma_{period}_resistance'] = (
                (df['close'] < df[ma_col]) & 
                (df['high'] >= df[ma_col])
            ).astype(int)
        
        # Golden cross / Death cross signals (comparing previous bar to avoid lookahead)
        df['golden_cross'] = (
            (df['sma_50'] > df['sma_200']) & 
            (df['sma_50'].shift(1) <= df['sma_200'].shift(1))
        ).astype(int)
        df['death_cross'] = (
            (df['sma_50'] < df['sma_200']) & 
            (df['sma_50'].shift(1) >= df['sma_200'].shift(1))
        ).astype(int)
        
        return df
    
    def _add_sr_flip_detection(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect when support becomes resistance and vice versa
        Uses shifts to ensure we're only looking at past data
        """
        # Resistance becoming support (price broke above old resistance 5+ bars ago and is holding)
        df['resistance_to_support'] = (
            (df['close'] > df['resistance_level'].shift(5)) & 
            (df['close'].shift(1) > df['resistance_level'].shift(6)) &
            (df['low'] >= df['resistance_level'].shift(5) * 0.98)
        ).astype(int)
        
        # Support becoming resistance (price broke below old support 5+ bars ago and is staying below)
        df['support_to_resistance'] = (
            (df['close'] < df['support_level'].shift(5)) & 
            (df['close'].shift(1) < df['support_level'].shift(6)) &
            (df['high'] <= df['support_level'].shift(5) * 1.02)
        ).astype(int)
        
        return df
    
    def _add_multi_timeframe_levels(self, df: pd.DataFrame, base_lookback: int) -> pd.DataFrame:
        """Calculate support/resistance on multiple timeframes using historical data only"""
        timeframes = {
            'short': max(base_lookback // 2, 5),  # Ensure minimum lookback
            'medium': base_lookback,
            'long': base_lookback * 2
        }
        
        for tf_name, lookback in timeframes.items():
            # Support/Resistance for each timeframe (rolling looks back only)
            df[f'support_{tf_name}'] = df['low'].rolling(lookback, min_periods=1).min()
            df[f'resistance_{tf_name}'] = df['high'].rolling(lookback, min_periods=1).max()
            
            # Distance to levels
            df[f'dist_to_support_{tf_name}'] = (
                (df['close'] - df[f'support_{tf_name}']) / df['close'] * 100
            ).fillna(0)
            df[f'dist_to_resistance_{tf_name}'] = (
                (df[f'resistance_{tf_name}'] - df['close']) / df['close'] * 100
            ).fillna(0)
        
        # Confluence detection (multiple timeframes agreeing on current bar)
        support_confluence_threshold = 0.02  # 2% difference
        resistance_confluence_threshold = 0.02
        
        df['support_confluence'] = (
            (abs(df['support_short'] - df['support_medium']) / df['support_medium'] < support_confluence_threshold) &
            (abs(df['support_medium'] - df['support_long']) / df['support_long'] < support_confluence_threshold)
        ).astype(int)
        
        df['resistance_confluence'] = (
            (abs(df['resistance_short'] - df['resistance_medium']) / df['resistance_medium'] < resistance_confluence_threshold) &
            (abs(df['resistance_medium'] - df['resistance_long']) / df['resistance_long'] < resistance_confluence_threshold)
        ).astype(int)
        
        return df
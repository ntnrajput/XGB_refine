# features/calculators/candlestick.py - ENHANCED VERSION

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class CandlestickCalculator:
    """Calculate candlestick patterns with proper NaN handling - Enhanced for swing trading"""
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick patterns and advanced features for swing trading"""
        df = df.copy()
        
        # ==========================
        # BASIC CANDLE PROPERTIES
        # ==========================
        df['body_size'] = abs(df['close'] - df['open'])
        df['upper_shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['low']
        df['is_green'] = (df['close'] > df['open']).fillna(False).astype(int)
        df['is_red'] = (df['close'] < df['open']).fillna(False).astype(int)
        df['total_range'] = df['high'] - df['low']
        
        # Relative measurements (normalized)
        avg_range = df['total_range'].rolling(20, min_periods=1).mean()
        df['body_relative'] = (df['body_size'] / avg_range).fillna(0)
        df['upper_shadow_relative'] = (df['upper_shadow'] / avg_range).fillna(0)
        df['lower_shadow_relative'] = (df['lower_shadow'] / avg_range).fillna(0)
        
        # Body position within candle
        df['body_center'] = (df['open'] + df['close']) / 2
        df['body_position'] = ((df['body_center'] - df['low']) / df['total_range']).fillna(0.5)
        
        # ==========================
        # CLASSIC PATTERNS
        # ==========================
        
        # Doji variations
        df['doji'] = (df['body_size'] / df['close'] < 0.001).fillna(False).astype(int)
        df['dragonfly_doji'] = (
            df['doji'] & 
            (df['lower_shadow'] > df['upper_shadow'] * 2)
        ).fillna(False).astype(int)
        df['gravestone_doji'] = (
            df['doji'] & 
            (df['upper_shadow'] > df['lower_shadow'] * 2)
        ).fillna(False).astype(int)
        
        # Hammer and Hanging Man (context dependent)
        hammer_shape = (
            (df['lower_shadow'] > df['body_size'] * 2) & 
            (df['upper_shadow'] < df['body_size'] * 0.3) &
            (df['body_position'] > 0.65)
        )
        df['hammer'] = hammer_shape.fillna(False).astype(int)
        df['inverted_hammer'] = (
            (df['upper_shadow'] > df['body_size'] * 2) & 
            (df['lower_shadow'] < df['body_size'] * 0.3) &
            (df['body_position'] < 0.35)
        ).fillna(False).astype(int)
        
        # Shooting star
        df['shooting_star'] = (
            (df['upper_shadow'] > df['body_size'] * 2) & 
            (df['lower_shadow'] < df['body_size']) &
            (df['body_position'] < 0.35)
        ).fillna(False).astype(int)
        
        # Marubozu (no shadows)
        df['marubozu_bullish'] = (
            (df['is_green'] == 1) &
            (df['upper_shadow'] < df['body_size'] * 0.03) &
            (df['lower_shadow'] < df['body_size'] * 0.03) &
            (df['body_relative'] > 0.5)
        ).fillna(False).astype(int)
        
        df['marubozu_bearish'] = (
            (df['is_red'] == 1) &
            (df['upper_shadow'] < df['body_size'] * 0.03) &
            (df['lower_shadow'] < df['body_size'] * 0.03) &
            (df['body_relative'] > 0.5)
        ).fillna(False).astype(int)
        
        # Spinning tops
        df['spinning_top'] = (
            (df['body_relative'] < 0.3) &
            (df['upper_shadow'] > df['body_size']) &
            (df['lower_shadow'] > df['body_size'])
        ).fillna(False).astype(int)
        
        # Long-legged doji
        df['long_legged_doji'] = (
            (df['body_relative'] < 0.1) &
            (df['upper_shadow_relative'] > 0.5) &
            (df['lower_shadow_relative'] > 0.5)
        ).fillna(False).astype(int)
        
        # ==========================
        # TWO-CANDLE PATTERNS
        # ==========================
        
        # Engulfing patterns with volume confirmation (if available)
        df['engulfing_bullish'] = (
            (df['is_green'] == 1) & 
            (df['is_red'].shift(1) == 1) & 
            (df['open'] <= df['close'].shift(1)) & 
            (df['close'] >= df['open'].shift(1)) &
            (df['body_size'] > df['body_size'].shift(1))
        ).fillna(False).astype(int)
        
        df['engulfing_bearish'] = (
            (df['is_red'] == 1) & 
            (df['is_green'].shift(1) == 1) & 
            (df['open'] >= df['close'].shift(1)) & 
            (df['close'] <= df['open'].shift(1)) &
            (df['body_size'] > df['body_size'].shift(1))
        ).fillna(False).astype(int)
        
        # Harami patterns
        df['harami_bullish'] = (
            (df['is_green'] == 1) &
            (df['is_red'].shift(1) == 1) &
            (df['open'] > df['close'].shift(1)) &
            (df['close'] < df['open'].shift(1)) &
            (df['body_size'] < df['body_size'].shift(1) * 0.5)
        ).fillna(False).astype(int)
        
        df['harami_bearish'] = (
            (df['is_red'] == 1) &
            (df['is_green'].shift(1) == 1) &
            (df['open'] < df['close'].shift(1)) &
            (df['close'] > df['open'].shift(1)) &
            (df['body_size'] < df['body_size'].shift(1) * 0.5)
        ).fillna(False).astype(int)
        
        # Piercing pattern and Dark cloud cover
        df['piercing_pattern'] = (
            (df['is_green'] == 1) &
            (df['is_red'].shift(1) == 1) &
            (df['open'] < df['low'].shift(1)) &
            (df['close'] > (df['open'].shift(1) + df['close'].shift(1)) / 2) &
            (df['close'] < df['open'].shift(1))
        ).fillna(False).astype(int)
        
        df['dark_cloud_cover'] = (
            (df['is_red'] == 1) &
            (df['is_green'].shift(1) == 1) &
            (df['open'] > df['high'].shift(1)) &
            (df['close'] < (df['open'].shift(1) + df['close'].shift(1)) / 2) &
            (df['close'] > df['open'].shift(1))
        ).fillna(False).astype(int)
        
        # Tweezer patterns
        df['tweezer_top'] = (
            (abs(df['high'] - df['high'].shift(1)) < df['close'] * 0.001) &
            (df['is_green'].shift(1) == 1) &
            (df['is_red'] == 1)
        ).fillna(False).astype(int)
        
        df['tweezer_bottom'] = (
            (abs(df['low'] - df['low'].shift(1)) < df['close'] * 0.001) &
            (df['is_red'].shift(1) == 1) &
            (df['is_green'] == 1)
        ).fillna(False).astype(int)
        
        # ==========================
        # THREE-CANDLE PATTERNS
        # ==========================
        
        # Morning/Evening star (enhanced)
        df['morning_star'] = (
            (df['is_green'] == 1) & 
            (df['body_size'] > df['body_size'].shift(2) * 0.5) &
            ((df['spinning_top'].shift(1) == 1) | (df['doji'].shift(1) == 1)) & 
            (df['is_red'].shift(2) == 1) &
            (df['close'] > (df['open'].shift(2) + df['close'].shift(2)) / 2)
        ).fillna(False).astype(int)
        
        df['evening_star'] = (
            (df['is_red'] == 1) & 
            (df['body_size'] > df['body_size'].shift(2) * 0.5) &
            ((df['spinning_top'].shift(1) == 1) | (df['doji'].shift(1) == 1)) & 
            (df['is_green'].shift(2) == 1) &
            (df['close'] < (df['open'].shift(2) + df['close'].shift(2)) / 2)
        ).fillna(False).astype(int)
        
        # Three soldiers/crows with strength
        three_green = (
            (df['is_green'] == 1) & 
            (df['is_green'].shift(1) == 1) & 
            (df['is_green'].shift(2) == 1)
        )
        progressive_highs = (
            (df['close'] > df['close'].shift(1)) &
            (df['close'].shift(1) > df['close'].shift(2))
        )
        df['three_white_soldiers'] = (
            three_green & progressive_highs &
            (df['body_relative'] > 0.3) &
            (df['body_relative'].shift(1) > 0.3) &
            (df['body_relative'].shift(2) > 0.3)
        ).fillna(False).astype(int)
        
        three_red = (
            (df['is_red'] == 1) & 
            (df['is_red'].shift(1) == 1) & 
            (df['is_red'].shift(2) == 1)
        )
        progressive_lows = (
            (df['close'] < df['close'].shift(1)) &
            (df['close'].shift(1) < df['close'].shift(2))
        )
        df['three_black_crows'] = (
            three_red & progressive_lows &
            (df['body_relative'] > 0.3) &
            (df['body_relative'].shift(1) > 0.3) &
            (df['body_relative'].shift(2) > 0.3)
        ).fillna(False).astype(int)
        
        # Three inside up/down
        df['three_inside_up'] = (
            df['harami_bullish'].shift(1) == 1
        ) & (df['close'] > df['high'].shift(1))
        df['three_inside_up'] = df['three_inside_up'].fillna(False).astype(int)
        
        df['three_inside_down'] = (
            df['harami_bearish'].shift(1) == 1
        ) & (df['close'] < df['low'].shift(1))
        df['three_inside_down'] = df['three_inside_down'].fillna(False).astype(int)
        
        # Abandoned baby
        gap_up = df['low'] > df['high'].shift(1)
        gap_down = df['high'] < df['low'].shift(1)
        
        df['abandoned_baby_bullish'] = (
            (df['is_green'] == 1) &
            (df['doji'].shift(1) == 1) &
            (df['is_red'].shift(2) == 1) &
            gap_down.shift(1) &
            gap_up
        ).fillna(False).astype(int)
        
        df['abandoned_baby_bearish'] = (
            (df['is_red'] == 1) &
            (df['doji'].shift(1) == 1) &
            (df['is_green'].shift(2) == 1) &
            gap_up.shift(1) &
            gap_down
        ).fillna(False).astype(int)
        
        # ==========================
        # SWING TRADING FEATURES
        # ==========================
        
        # Price action features
        df['higher_high'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['high'].shift(1) > df['high'].shift(2))
        ).fillna(False).astype(int)
        
        df['lower_low'] = (
            (df['low'] < df['low'].shift(1)) &
            (df['low'].shift(1) < df['low'].shift(2))
        ).fillna(False).astype(int)
        
        df['inside_bar'] = (
            (df['high'] <= df['high'].shift(1)) &
            (df['low'] >= df['low'].shift(1))
        ).fillna(False).astype(int)
        
        df['outside_bar'] = (
            (df['high'] > df['high'].shift(1)) &
            (df['low'] < df['low'].shift(1))
        ).fillna(False).astype(int)
        
        # Pin bars (rejection candles)
        df['pin_bar_bullish'] = (
            (df['lower_shadow'] > df['body_size'] * 3) &
            (df['lower_shadow'] > df['upper_shadow'] * 2) &
            (df['body_position'] > 0.7)
        ).fillna(False).astype(int)
        
        df['pin_bar_bearish'] = (
            (df['upper_shadow'] > df['body_size'] * 3) &
            (df['upper_shadow'] > df['lower_shadow'] * 2) &
            (df['body_position'] < 0.3)
        ).fillna(False).astype(int)
        
        # Gap patterns
        df['gap_up'] = (df['low'] > df['high'].shift(1)).fillna(False).astype(int)
        df['gap_down'] = (df['high'] < df['low'].shift(1)).fillna(False).astype(int)
        df['gap_size'] = np.where(
            df['gap_up'], 
            (df['low'] - df['high'].shift(1)) / df['close'].shift(1),
            np.where(
                df['gap_down'],
                (df['low'].shift(1) - df['high']) / df['close'].shift(1),
                0
            )
        )
        
        # Momentum candles
        avg_body = df['body_size'].rolling(20, min_periods=1).mean()
        df['strong_bullish_candle'] = (
            (df['is_green'] == 1) &
            (df['body_size'] > avg_body * 1.5) &
            (df['upper_shadow'] < df['body_size'] * 0.25)
        ).fillna(False).astype(int)
        
        df['strong_bearish_candle'] = (
            (df['is_red'] == 1) &
            (df['body_size'] > avg_body * 1.5) &
            (df['lower_shadow'] < df['body_size'] * 0.25)
        ).fillna(False).astype(int)
        
        # Reversal zones (multiple rejections)
        df['rejection_zone_top'] = (
            (df['shooting_star'] | df['pin_bar_bearish'] | df['gravestone_doji']) &
            ((df['shooting_star'].shift(1) == 1) | 
             (df['pin_bar_bearish'].shift(1) == 1) |
             (df['shooting_star'].shift(2) == 1))
        ).fillna(False).astype(int)
        
        df['rejection_zone_bottom'] = (
            (df['hammer'] | df['pin_bar_bullish'] | df['dragonfly_doji']) &
            ((df['hammer'].shift(1) == 1) | 
             (df['pin_bar_bullish'].shift(1) == 1) |
             (df['hammer'].shift(2) == 1))
        ).fillna(False).astype(int)
        
        # ==========================
        # PATTERN STRENGTH SCORES
        # ==========================
        
        # Bullish strength score
        df['bullish_score'] = (
            df['hammer'] * 2 +
            df['engulfing_bullish'] * 3 +
            df['morning_star'] * 3 +
            df['three_white_soldiers'] * 4 +
            df['piercing_pattern'] * 2 +
            df['harami_bullish'] * 2 +
            df['tweezer_bottom'] * 2 +
            df['pin_bar_bullish'] * 2 +
            df['dragonfly_doji'] * 1 +
            df['marubozu_bullish'] * 3 +
            df['three_inside_up'] * 3 +
            df['abandoned_baby_bullish'] * 4 +
            df['strong_bullish_candle'] * 1 +
            df['rejection_zone_bottom'] * 3
        )
        
        # Bearish strength score
        df['bearish_score'] = (
            df['shooting_star'] * 2 +
            df['engulfing_bearish'] * 3 +
            df['evening_star'] * 3 +
            df['three_black_crows'] * 4 +
            df['dark_cloud_cover'] * 2 +
            df['harami_bearish'] * 2 +
            df['tweezer_top'] * 2 +
            df['pin_bar_bearish'] * 2 +
            df['gravestone_doji'] * 1 +
            df['marubozu_bearish'] * 3 +
            df['three_inside_down'] * 3 +
            df['abandoned_baby_bearish'] * 4 +
            df['strong_bearish_candle'] * 1 +
            df['rejection_zone_top'] * 3
        )
        
        # Net pattern signal
        df['pattern_signal'] = df['bullish_score'] - df['bearish_score']
        
        # ==========================
        # TREND CONTEXT FEATURES
        # ==========================
        
        # Recent price action context
        df['candles_above_ma'] = (df['close'] > df['close'].rolling(20, min_periods=1).mean()).astype(int)
        df['consecutive_green'] = (df['is_green'].rolling(3, min_periods=1).sum() == 3).astype(int)
        df['consecutive_red'] = (df['is_red'].rolling(3, min_periods=1).sum() == 3).astype(int)
        
        # Volatility context
        df['range_expansion'] = (
            df['total_range'] > df['total_range'].rolling(20, min_periods=1).mean() * 1.5
        ).fillna(False).astype(int)
        
        df['range_contraction'] = (
            df['total_range'] < df['total_range'].rolling(20, min_periods=1).mean() * 0.5
        ).fillna(False).astype(int)
        
        # ==========================
        # VOLUME-BASED PATTERNS (if volume column exists)
        # ==========================
        if 'volume' in df.columns:
            avg_volume = df['volume'].rolling(20, min_periods=1).mean()
            
            # Volume confirmation
            df['volume_spike'] = (df['volume'] > avg_volume * 1.5).fillna(False).astype(int)
            df['low_volume'] = (df['volume'] < avg_volume * 0.5).fillna(False).astype(int)
            
            # Pattern with volume confirmation
            df['bullish_volume_confirm'] = (
                (df['bullish_score'] > 0) & 
                (df['volume_spike'] == 1)
            ).fillna(False).astype(int)
            
            df['bearish_volume_confirm'] = (
                (df['bearish_score'] > 0) & 
                (df['volume_spike'] == 1)
            ).fillna(False).astype(int)
            
            # Accumulation/Distribution patterns
            df['accumulation'] = (
                (df['is_green'] == 1) &
                (df['volume'] > avg_volume) &
                (df['close'] > (df['high'] + df['low']) / 2)
            ).fillna(False).astype(int)
            
            df['distribution'] = (
                (df['is_red'] == 1) &
                (df['volume'] > avg_volume) &
                (df['close'] < (df['high'] + df['low']) / 2)
            ).fillna(False).astype(int)
        
        # ==========================
        # MULTI-TIMEFRAME PATTERNS
        # ==========================
        
        # Key reversal
        df['key_reversal_bullish'] = (
            (df['low'] < df['low'].rolling(20, min_periods=1).min()) &
            (df['close'] > df['high'].shift(1)) &
            (df['is_green'] == 1)
        ).fillna(False).astype(int)
        
        df['key_reversal_bearish'] = (
            (df['high'] > df['high'].rolling(20, min_periods=1).max()) &
            (df['close'] < df['low'].shift(1)) &
            (df['is_red'] == 1)
        ).fillna(False).astype(int)
        
        # ==========================
        # PATTERN CLUSTERS
        # ==========================
        
        # Count patterns in recent window
        pattern_cols = [col for col in df.columns if any(x in col for x in 
                       ['doji', 'hammer', 'star', 'engulfing', 'harami', 'pin_bar'])]
        
      
        rolling_sum = df[pattern_cols].rolling(3, min_periods=1).sum()
        df['pattern_density_3'] = rolling_sum.sum(axis=1)
        rolling_sum_5 = df[pattern_cols].rolling(5, min_periods=1).sum()
        df['pattern_density_5'] = rolling_sum_5.sum(axis=1)
                
        # ==========================
        # FINAL COMPOSITE SIGNALS
        # ==========================
        
        # Strong reversal signals
        df['strong_bullish_reversal'] = (
            (df['bullish_score'] >= 5) |
            ((df['bullish_score'] >= 3) & (df.get('volume_spike', 0) == 1)) |
            (df['key_reversal_bullish'] == 1)
        ).fillna(False).astype(int)
        
        df['strong_bearish_reversal'] = (
            (df['bearish_score'] >= 5) |
            ((df['bearish_score'] >= 3) & (df.get('volume_spike', 0) == 1)) |
            (df['key_reversal_bearish'] == 1)
        ).fillna(False).astype(int)
        
        # Continuation patterns
        df['bullish_continuation'] = (
            (df['candles_above_ma'] == 1) &
            (df['is_green'] == 1) &
            (df['inside_bar'].shift(1) == 1)
        ).fillna(False).astype(int)
        
        df['bearish_continuation'] = (
            (df['candles_above_ma'] == 0) &
            (df['is_red'] == 1) &
            (df['inside_bar'].shift(1) == 1)
        ).fillna(False).astype(int)
        
        # logger.info(f"Calculated {len([c for c in df.columns if c not in ['open', 'high', 'low', 'close', 'volume']])} candlestick features")
        
        return df


__all__ = ['CandlestickCalculator']
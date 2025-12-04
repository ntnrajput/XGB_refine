# features/calculators/enhanced.py - FIXED

import pandas as pd
import numpy as np

from utils.logger import get_logger

logger = get_logger(__name__)


class EnhancedCalculator:
    """Calculate enhanced features"""
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add enhanced features"""
        df = df.copy()
        
        # Price relationships
        df['high_low_range'] = (df['high'] - df['low']) / df['close'] * 100
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
        df['gap_pct'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        # Handle division by zero
        df['close_position'] = df['close_position'].replace([np.inf, -np.inf], np.nan)
        
        # Multi-timeframe
        if 'ema_50' in df.columns and 'ema_200' in df.columns:
            df['ema50_ema200'] = df['ema_50'] / df['ema_200']
        
        if 'sma_50' in df.columns and 'sma_200' in df.columns:
            df['sma50_sma200'] = df['sma_50'] / df['sma_200']
        
        if 'ema_200' in df.columns:
            df['ema200_price'] = df['ema_200'] / df['close']
        
        if 'sma_200' in df.columns:
            df['sma200_price'] = df['sma_200'] / df['close']
        
        # Statistical features
        window = 20
        
        # Z-scores
        price_mean = df['close'].rolling(window).mean()
        price_std = df['close'].rolling(window).std()
        df['price_zscore'] = (df['close'] - price_mean) / price_std
        
        volume_mean = df['volume'].rolling(window).mean()
        volume_std = df['volume'].rolling(window).std()
        df['volume_zscore'] = (df['volume'] - volume_mean) / volume_std
        
        # Percentile ranks - FIXED: use x[-1] not x.iloc[-1] when raw=True
        window = 100
        
        df['price_percentile'] = df['close'].rolling(window).apply(
            lambda x: (x[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50, 
            raw=True
        )
        
        df['volume_percentile'] = df['volume'].rolling(window).apply(
            lambda x: (x[-1] > x).sum() / len(x) * 100 if len(x) > 0 else 50,
            raw=True
        )
        
        return df


__all__ = ['EnhancedCalculator']

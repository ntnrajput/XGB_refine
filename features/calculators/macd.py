# features/calculators/macd.py

import pandas as pd
from typing import Optional

from config.features import IndicatorConfig, INDICATOR_CONFIG
from utils.helpers import calculate_ema
from utils.logger import get_logger

logger = get_logger(__name__)


class MACDCalculator:
    """Calculate MACD indicators"""
    
    def __init__(self, config: Optional[IndicatorConfig] = None):
        self.config = config or INDICATOR_CONFIG
    
    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD features"""
        df = df.copy()
        
        # MACD line
        ema_fast = calculate_ema(df['close'], self.config.macd_fast)
        ema_slow = calculate_ema(df['close'], self.config.macd_slow)
        df['macd'] = ema_fast - ema_slow
        
        # Signal line
        df['macd_signal'] = calculate_ema(df['macd'], self.config.macd_signal)
        
        # Histogram
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Crossovers
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                              (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
        
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
        
        # Divergence (simplified)
        df['macd_divergence'] = ((df['macd_hist'] > 0) & (df['macd_hist'].shift(1) < 0) |
                                (df['macd_hist'] < 0) & (df['macd_hist'].shift(1) > 0)).astype(int)
        
        return df

# data/filters.py - NEW FILE

import pandas as pd
from typing import List, Dict
from dataclasses import dataclass
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class StockFilterConfig:
    min_avg_volume: float = 0.5   # ↓ from 3.0
    min_price: float = 20.0       # ↓ if midcaps included
    min_historical_days: int = 50
    max_price: float = 100000.0


class StockFilter:
    """Filter stocks based on liquidity and quality criteria"""
    
    def __init__(self, config: StockFilterConfig):
        self.config = config
    
    def filter_symbols(self, df: pd.DataFrame) -> List[str]:
        """
        Filter symbols based on volume and price criteria.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            List of symbols that pass filters
        """
        logger.info("[filters.py] 🔍 Applying stock filters...")
        
        filtered_symbols = []
        filter_stats = {
            'total': 0,
            'insufficient_history': 0,
            'low_volume': 0,
            'low_price': 0,
            'high_price': 0,
            'passed': 0
        }
        
        for symbol, group in df.groupby('symbol'):
            filter_stats['total'] += 1
            
            # Sort by date
            group = group.sort_values('date')
            
            # Check 1: Sufficient history
            if len(group) < self.config.min_historical_days:
                filter_stats['insufficient_history'] += 1
                continue
            
            # Check 2: Average volume (last 50 days)
            avg_volume = group['volume'].tail(50).mean()
            if avg_volume < self.config.min_avg_volume:
                filter_stats['low_volume'] += 1
                continue
            
            # Check 3: Latest price
            latest_close = group.iloc[-1]['close']
            
            if latest_close < self.config.min_price:
                filter_stats['low_price'] += 1
                continue
            
            if latest_close > self.config.max_price:
                filter_stats['high_price'] += 1
                continue
            
            # Passed all filters
            filtered_symbols.append(symbol)
            filter_stats['passed'] += 1
        
        # Log statistics
        self._log_filter_stats(filter_stats)
        
        return filtered_symbols
    
    def _log_filter_stats(self, stats: Dict[str, int]):
        """Log filtering statistics"""
        logger.info("[filters.py] 📊 Filter Results:")
        logger.info(f"[filters.py]   Total symbols: {stats['total']}")
        logger.info(f"[filters.py]   ✅ Passed: {stats['passed']}")
        logger.info(f"[filters.py]   ❌ Insufficient history: {stats['insufficient_history']}")
        logger.info(f"[filters.py]   ❌ Low volume: {stats['low_volume']}")
        logger.info(f"[filters.py]   ❌ Low price: {stats['low_price']}")
        logger.info(f"[filters.py]   ❌ High price: {stats['high_price']}")
        logger.info(f"[filters.py]   Pass rate: {stats['passed']/stats['total']*100:.1f}%")
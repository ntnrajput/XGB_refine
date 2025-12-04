# strategy/screener.py - Stock screening for trading opportunities (with date option)

import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from models.trainer import ModelLoader
from data.fetchers.historical import HistoricalDataFetcher
from data.fetchers.symbols import load_symbols
from data.processors.filters import StockFilter
from config.settings import OUTPUT_DIR
from config.models import MODEL_CONFIG
from config.features import FILTER_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


class Screener:
    """Screen stocks for trading opportunities"""
    
    def __init__(self):
        pass
    
    def filter_trend_df(self,df):
        # df['ema_50/ema_200'] = df['ema_50'] / df['ema_200']
        # df = df[df['ema_50/ema_200'] > 0.9]
        # df = df[df['close']> df['ema_200']]
        return df

    def screen(
        self,
        model_path: Path,
        feature_pipeline,
        stock_filter: StockFilter,
        min_probability: float = 0.77,
        screen_date: Optional[str] = None
    ) -> List[Dict]:
        """
        Screen stocks for opportunities.
        
        Args:
            model_path: Path to trained model
            feature_pipeline: Feature pipeline
            stock_filter: Stock filter
            min_probability: Minimum probability threshold
            screen_date: Optional date to screen (YYYY-MM-DD). If None, uses latest date.
            
        Returns:
            List of opportunities
        """
        logger.info("=" * 80)
        logger.info("🔍 STOCK SCREENING")
        logger.info("=" * 80)
        
        # ✅ CRITICAL FIX: Determine target date BEFORE loading data
        if screen_date:
            target_date = pd.to_datetime(screen_date)
            logger.info(f"\n[screener.py] 📅 Screening for date: {target_date.date()}")
        else:
            target_date = None  # Will use latest date from data
            logger.info(f"\n[screener.py] 📅 Screening for latest available date")
        
        # ✅ CRITICAL FIX: Load data with proper historical context
        df = self._load_latest_data(target_date)  # ← Pass target_date here!
        
        if df.empty:
            logger.warning("[screener.py] ❌ No data available for screening")
            return []
        
        logger.info(f"\n[screener.py] 📂 Loaded data: {len(df):,} rows")
        logger.info(f"[screener.py] 📅 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
        logger.info(f"[screener.py] 📊 Symbols: {df['symbol'].nunique()}")
        
        # Filter stocks
        logger.info("\n[screener.py] 🔍 Applying stock filters...")
        filtered_symbols = stock_filter.filter_symbols(df)
        df = df[df['symbol'].isin(filtered_symbols)]
        print(df)
        
        logger.info(f"[screener.py] ✅ Passed filter: {len(filtered_symbols)} symbols")
        
        # Add features
        logger.info("\n[screener.py] 🔧 Calculating features...")
        df = feature_pipeline.transform(df)
        logger.info(f"[screener.py] ✅ Features calculated: {len(df.columns)} columns")

        
        
        # Finalize target date (if not specified, use latest from data)
        if target_date is None:
            target_date = df['date'].max()
            logger.info(f"\n[screener.py] 📅 Using latest date: {target_date.date()}")
        else:
            # Verify target date exists in data
            if target_date not in df['date'].values:
                logger.warning(f"[screener.py] ⚠️  Date {target_date.date()} not found in data")
                logger.info(f"[screener.py] 📅 Available date range: {df['date'].min().date()} to {df['date'].max().date()}")
                
                # Find closest date
                closest_date = df['date'].iloc[(df['date'] - target_date).abs().argsort()[0]]
                logger.info(f"[screener.py] 📅 Using closest available date: {closest_date.date()}")
                target_date = closest_date
            else:
                logger.info(f"\n[screener.py] 📅 Confirmed screening date: {target_date.date()}")
        
        # Filter to target date
        df = df[df['date'] == target_date]
        df = self.filter_trend_df(df)
        logger.info(f"[screener.py] 📊 Stocks on {target_date.date()}: {len(df)} rows")
        
        if df.empty:
            logger.warning(f"[screener.py] ❌ No data available for {target_date.date()}")
            return []
        
        # Load model and predict
        logger.info("\n[screener.py] 🤖 Running model predictions...")
        model_bundle = ModelLoader.load(model_path)
        predictions, probabilities = ModelLoader.predict(model_bundle, df)
        
        if df.columns.duplicated().any():
            duplicate_cols = df.columns[df.columns.duplicated()].unique().tolist()
            logger.warning(f"[screener.py] ⚠️  Found {len(duplicate_cols)} duplicate column names: {duplicate_cols[:10]}")
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            logger.info(f"[screener.py] ✅ De-duplicated: now {len(df.columns)} unique columns")

        # Count signals
        buy_signals = predictions.sum()
        logger.info(f"[screener.py] 📊 Total signals: {buy_signals} / {len(predictions)} ({buy_signals/len(predictions)*100:.1f}%)")
        
        # Find opportunities
        opportunities = []
        
        for i in range(len(df)):
            if predictions[i] == 1 and probabilities[i][1] >= min_probability:
                row = df.iloc[i]
                
                opportunity = {
                    'symbol': row['symbol'],
                    'date': row['date'],
                    'price': float(row['close']),
                    'probability': float(probabilities[i][1]),
                    'direction': 'LONG',
                    'prediction': 'BUY'
                }
                try:
                    if 'rsi_14' in df.columns:
                        opportunity['rsi'] = float(row['rsi_14'])
                    if 'atr_pct' in df.columns:
                        opportunity['atr_pct'] = float(row['atr_pct'])
                    if 'volume_ratio' in df.columns:
                        opportunity['volume_ratio'] = float(row['volume_ratio'])
                except (TypeError, ValueError) as e:
                    logger.warning(f"[screener.py] ⚠️  Could not extract indicators for {row['symbol']}: {e}")
                
                opportunities.append(opportunity)


        
        # Sort by probability
        opportunities.sort(key=lambda x: x['probability'], reverse=True)
        
        logger.info(f"\n[screener.py] ✅ Found {len(opportunities)} opportunities (probability >= {min_probability:.1%})")
        
        if opportunities:
            logger.info(f"[screener.py] 📈 Top 10 by probability:")
            for i, opp in enumerate(opportunities[:10], 1):
                logger.info(f"   {i:2d}. {opp['symbol']:20s} - {opp['probability']:.1%} @ ₹{opp['price']:.2f}")
        
        # Save results
        self._save_results(opportunities, target_date)
        
        logger.info("=" * 80)
        
        return opportunities
    
    def _load_latest_data(self, screen_date: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Load data with proper historical context"""
        from config.settings import HISTORICAL_DATA_FILE
        
        try:
            df = pd.read_parquet(HISTORICAL_DATA_FILE)
            df['date'] = pd.to_datetime(df['date'])
            
            if screen_date:
                # Load 1 year of data BEFORE the screening date (for features)
                one_year_before = screen_date - timedelta(days=1000)
                df = df[(df['date'] >= one_year_before) & (df['date'] <= screen_date)]
                logger.info(f"[screener.py] 📂 Loaded data from {one_year_before.date()} to {screen_date.date()}")
            else:
                # Default: load last 1 year from today
                one_year_ago = datetime.now() - timedelta(days=1000)
                df = df[df['date'] >= one_year_ago]
                logger.info(f"[screener.py] 📂 Loaded data from last year")
            
            
            return df
        
        except Exception as e:
            logger.error(f"[screener.py] ❌ Failed to load data: {e}")
            return pd.DataFrame()
    
    def _save_results(self, opportunities: List[Dict], screening_date: pd.Timestamp):
        """Save screening results"""
        if not opportunities:
            logger.info("[screener.py] No opportunities to save")
            return
        
        # Create results directory
        results_dir = OUTPUT_DIR / "screening_results"
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = results_dir / f"screening_{timestamp}.csv"
        
        df = pd.DataFrame(opportunities)
        df.to_csv(filename, index=False)
        
        logger.info(f"\n[screener.py] 💾 Results saved: {filename}")
        logger.info(f"[screener.py] 📊 Total opportunities: {len(opportunities)}")
        logger.info(f"[screener.py] 📅 Screening date: {screening_date.date()}")
        
        # Also save a "latest" file (overwrites previous)
        latest_filename = results_dir / "latest_screening.csv"
        df.to_csv(latest_filename, index=False)
        logger.info(f"[screener.py] 💾 Latest screening: {latest_filename}")


def run_screening(pipeline, model_config):
    print('model config ye hai',model_config)
    """Convenience function for screening"""
    screener = Screener()
    return screener


__all__ = ['Screener', 'run_screening']
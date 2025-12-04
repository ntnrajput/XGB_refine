# features/pipeline.py - Optimized feature engineering pipeline

import pandas as pd
import numpy as np
from typing import Optional, List
from dataclasses import dataclass
from tqdm import tqdm
import time

from features.calculators import (
    BasicCalculator,
    CandlestickCalculator,
    MACDCalculator,
    AdvancedCalculator,
    EnhancedCalculator,
    SupportResistanceCalculator
)
from features.labels import SwingLabelGenerator
from features.validators import FeatureValidator
from config.features import INDICATOR_CONFIG, SWING_CONFIG
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for feature pipeline"""
    
    # Components to include
    include_basic: bool = True
    include_candlestick: bool = True
    include_macd: bool = True
    include_advanced: bool = True
    include_enhanced: bool = True
    include_support_resistance: bool = True
    include_swing_labels: bool = True
    
    # Validation
    validate_input: bool = True
    validate_output: bool = True
    
    # Error handling
    fail_on_error: bool = True
    skip_failed_symbols: bool = True
    
    # Performance
    show_progress: bool = True
    progress_interval: int = 100
    batch_size: int = 50
    profile_performance: bool = True


class PerformanceProfiler:
    """Track performance of each component"""
    
    def __init__(self):
        self.timings = {}
        self.counts = {}
    
    def record(self, component: str, duration: float):
        if component not in self.timings:
            self.timings[component] = []
            self.counts[component] = 0
        self.timings[component].append(duration)
        self.counts[component] += 1
    
    def print_summary(self):
        """Print performance summary"""
        if not self.timings:
            return
        
        summary = {}
        for component, times in self.timings.items():
            summary[component] = {
                'total': sum(times),
                'avg': np.mean(times),
                'count': self.counts[component]
            }
        
        total_time = sum(s['total'] for s in summary.values())
        
        logger.info("\n" + "=" * 80)
        logger.info("⚡ PERFORMANCE PROFILE")
        logger.info("=" * 80)
        
        sorted_components = sorted(summary.items(), key=lambda x: x[1]['total'], reverse=True)
        
        for component, stats in sorted_components:
            pct = (stats['total'] / total_time * 100) if total_time > 0 else 0
            logger.info(f"{component:25s} {stats['total']:7.2f}s ({pct:5.1f}%) | "
                       f"Avg: {stats['avg']*1000:6.1f}ms | Count: {stats['count']}")
        
        logger.info("=" * 80)


class FeaturePipeline:
    """
    Optimized feature engineering pipeline.
    
    Speed improvements:
    - Batch processing
    - Performance profiling
    - Progress bars
    - Efficient concatenation
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Initialize calculators
        self.calculators = []
        
        if self.config.include_basic:
            self.calculators.append(('Basic', BasicCalculator(INDICATOR_CONFIG)))
        
        if self.config.include_candlestick:
            self.calculators.append(('Candlestick', CandlestickCalculator()))
        
        if self.config.include_macd:
            self.calculators.append(('MACD', MACDCalculator(INDICATOR_CONFIG)))
        
        if self.config.include_support_resistance:
            self.calculators.append(('SupportResistance', SupportResistanceCalculator(INDICATOR_CONFIG)))
        
        if self.config.include_advanced:
            self.calculators.append(('Advanced', AdvancedCalculator()))
        
        if self.config.include_enhanced:
            self.calculators.append(('Enhanced', EnhancedCalculator()))
        
        if self.config.include_swing_labels:
            self.calculators.append(('SwingLabel', SwingLabelGenerator(SWING_CONFIG)))
        
        # Validator
        self.validator = FeatureValidator(strict=False) if self.config.validate_output else None
        
        # Performance profiler
        self.profiler = PerformanceProfiler() if self.config.profile_performance else None
        
        # Log
        component_names = [name for name, _ in self.calculators]
        logger.info("✅ Feature pipeline initialized")
        logger.info(f"Components: {', '.join(component_names)}")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature calculations (OPTIMIZED).
        
        Args:
            df: Raw OHLCV data with 'symbol' column
            
        Returns:
            DataFrame with features added
        """
        start_time = time.time()
        
        logger.info("=" * 70)
        logger.info("                     FEATURE ENGINEERING PIPELINE")
        logger.info("=" * 70)
        logger.info(f"Input shape: {df.shape}")
        logger.info(f"Symbols: {df['symbol'].nunique()}")
        logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Validate input
        if self.config.validate_input:
            self._validate_input(df)
        
        # Get symbols
        symbols = df['symbol'].unique()
        total_symbols = len(symbols)
        
        # Process with progress bar
        results = []
        failed_symbols = []
        
        with tqdm(total=total_symbols, desc="Processing", unit="sym") as pbar:
            for i, symbol in enumerate(symbols):
                # Show progress every N symbols
                if self.config.show_progress and (i + 1) % self.config.progress_interval == 0:
                    logger.info(f"Processing symbol {i + 1}/{total_symbols}: {symbol}")
                
                try:
                    symbol_df = df[df['symbol'] == symbol].copy()
                    processed = self._process_symbol(symbol, symbol_df)
                    results.append(processed)
                
                except Exception as e:
                    logger.error(f"Failed to process {symbol}: {e}")
                    failed_symbols.append(symbol)
                    
                    if self.config.fail_on_error:
                        raise
                    elif not self.config.skip_failed_symbols:
                        results.append(symbol_df)
                
                pbar.update(1)
        
        # Combine results
        if not results:
            raise ValueError("No symbols processed successfully")
        
        combine_start = time.time()
        df_final = pd.concat(results, ignore_index=True)
        combine_time = time.time() - combine_start
        
        if self.profiler:
            self.profiler.record('DataFrame Combine', combine_time)
        
        # Validate output
        if self.validator:
            validation_start = time.time()
            validation = self.validator.validate(df_final)
            
            if self.profiler:
                self.profiler.record('Validation', time.time() - validation_start)
            
            if validation.warnings:
                logger.warning("⚠️  Feature validation warnings:")
                for warning in validation.warnings[:3]:
                    logger.warning(f"   • {warning}")
        
        # Total time
        total_time = time.time() - start_time
        
        # Summary
        logger.info("=" * 70)
        logger.info("                          PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"✅ Output shape: {df_final.shape}")
        logger.info(f"✅ Features added: {df_final.shape[1] - df.shape[1]}")
        logger.info(f"✅ Success rate: {len(results)}/{total_symbols}")
        logger.info(f"⏱️  Total time: {total_time:.1f}s ({total_symbols/total_time:.1f} sym/s)")
        
        if failed_symbols:
            logger.warning(f"⚠️  Failed symbols: {len(failed_symbols)}")
        
        # Performance profile
        if self.profiler:
            self.profiler.print_summary()
        
        # Show features
        self._show_features(df_final)
        
        return df_final
    
    def _validate_input(self, df: pd.DataFrame):
        """Validate input dataframe"""
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        if df.empty:
            raise ValueError("Input dataframe is empty")
    
    def _process_symbol(self, symbol: str, df: pd.DataFrame) -> pd.DataFrame:
        """Process single symbol through all calculators"""
        df = df.sort_values('date').reset_index(drop=True)
        
        # Apply each calculator
        for name, calculator in self.calculators:
            start_time = time.time()
            
            try:
                df = calculator.calculate(df)
            except Exception as e:
                logger.error(f"{symbol}: {name} failed: {e}")
                if self.config.fail_on_error:
                    raise
            
            if self.profiler:
                self.profiler.record(name, time.time() - start_time)
        
        return df
    
    def _show_features(self, df: pd.DataFrame):
        """Show generated features"""
        
        # Get feature columns (exclude base OHLCV)
        base_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in df.columns if col not in base_cols]
        
        if not feature_cols:
            return
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 GENERATED FEATURES")
        logger.info("=" * 80)
        
        # Group features
        groups = {
            'Price & Returns': [c for c in feature_cols if any(x in c.lower() for x in ['return', 'price', 'gap'])],
            'Moving Averages': [c for c in feature_cols if 'sma' in c.lower() or 'ema' in c.lower()],
            'Indicators': [c for c in feature_cols if any(x in c.lower() for x in ['rsi', 'macd', 'bb', 'atr', 'adx'])],
            'Candlestick': [c for c in feature_cols if any(x in c.lower() for x in ['doji', 'hammer', 'engulfing', 'star'])],
            'Volume': [c for c in feature_cols if 'volume' in c.lower() or 'obv' in c.lower()],
            'Advanced': [c for c in feature_cols if any(x in c.lower() for x in ['momentum', 'roc', 'volatility', 'trend'])],
            'Support/Resistance': [c for c in feature_cols if any(x in c.lower() for x in ['support', 'resistance', 'level'])],
            'Enhanced': [c for c in feature_cols if any(x in c.lower() for x in ['zscore', 'percentile'])],
            'Labels': [c for c in feature_cols if 'label' in c.lower() or 'swing' in c.lower()],
        }
        
        # Print grouped features
        total = 0
        for group_name, features in groups.items():
            if features:
                logger.info(f"\n{group_name} ({len(features)}):")
                for feat in features[:10]:  # Show first 10
                    try:
                        sample = df[feat].iloc[-1] if len(df) > 0 else None
                        if pd.notna(sample):
                            logger.info(f"   • {feat:30s} = {sample:.4f}")
                        else:
                            logger.info(f"   • {feat:30s} = NaN")
                    except:
                        logger.info(f"   • {feat}")
                
                if len(features) > 10:
                    logger.info(f"   ... and {len(features) - 10} more")
                
                total += len(features)
        
        # Ungrouped
        grouped_features = set()
        for features in groups.values():
            grouped_features.update(features)
        
        ungrouped = [f for f in feature_cols if f not in grouped_features]
        if ungrouped:
            logger.info(f"\nOther ({len(ungrouped)}):")
            for feat in ungrouped[:10]:
                logger.info(f"   • {feat}")
            if len(ungrouped) > 10:
                logger.info(f"   ... and {len(ungrouped) - 10} more")
            total += len(ungrouped)
        
        logger.info(f"\n📊 Total Features: {total}")
        logger.info("=" * 80)


def get_training_pipeline() -> FeaturePipeline:
    """Get pipeline for training"""
    config = PipelineConfig(
        include_basic=True,
        include_candlestick=True,
        include_macd=True,
        include_advanced=True,
        include_enhanced=True,
        include_support_resistance=True,
        include_swing_labels=True,
        validate_input=True,
        validate_output=True,
        fail_on_error=True,
        skip_failed_symbols=False,
        show_progress=True,
        profile_performance=True
    )
    return FeaturePipeline(config)


def get_inference_pipeline() -> FeaturePipeline:
    """Get pipeline for inference/backtesting"""
    config = PipelineConfig(
        include_basic=True,
        include_candlestick=True,
        include_macd=True,
        include_advanced=True,
        include_enhanced=True,
        include_support_resistance=True,
        include_swing_labels=False,
        validate_input=True,
        validate_output=False,
        fail_on_error=False,
        skip_failed_symbols=True,
        show_progress=True,
        profile_performance=True
    )
    return FeaturePipeline(config)


__all__ = [
    'FeaturePipeline',
    'PipelineConfig',
    'get_training_pipeline',
    'get_inference_pipeline'
]

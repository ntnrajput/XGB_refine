# main.py - Fixed imports

import argparse
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Config imports - split across multiple files
from config.settings import (
    HISTORICAL_DATA_FILE,
    OUTPUT_DIR,
    TOKEN_PATH
)
from config.models import MODEL_CONFIG
from config.features import (
    FILTER_CONFIG,
    SWING_CONFIG,
    FEATURE_COLUMNS
)

# Data imports
from data.api.auth import authenticate, generate_token
from data.fetchers.historical import HistoricalDataFetcher
from data.fetchers.symbols import load_symbols
from data.processors.filters import StockFilter

# Feature imports
from features.pipeline import FeaturePipeline, PipelineConfig

# Model imports
from models.trainer import ModelTrainer
from models.backtester import Backtester

# Strategy imports
from strategy.screener import Screener

# Utils
from utils.logger import get_logger
from utils.exceptions import DataError, ModelError

logger = get_logger("Main")


class SwingTradingCLI:
    """Command-line interface for swing trading system"""
    
    def __init__(self):
        self.stock_filter = StockFilter(FILTER_CONFIG)
        
        # Use simpler pipeline configuration for backtest/screening
        backtest_config = PipelineConfig(
            include_basic=True,
            include_candlestick=True,
            include_macd=True,
            include_advanced=True,
            include_enhanced=True,
            include_support_resistance=True,
            include_swing_labels=False,  # No labels for backtest/screening
            validate_input=True,
            validate_output=False,
            fail_on_error=False,
            skip_failed_symbols=True,
            show_progress=True
        )
        
        self.feature_pipeline = FeaturePipeline(backtest_config)
        self.model_trainer = ModelTrainer()
        self.backtester = Backtester()
        self.screener = Screener()
    
    def filter_df(self, df):
        # df['ema_50/ema_200'] = df['ema_50'] / df['ema_200']
        # df = df[df['ema_50/ema_200'] > 0.9]
        df = df[df['close']> df['ema_200']]
        return df

    def authenticate_fyers(self):
        """Step 1: Authenticate with FYERS"""
        logger.info("[main.py] 🔐 Authenticating with FYERS...")
        authenticate()
    
    def generate_access_token(self, auth_code: str):
        """Step 2: Generate access token from auth code"""
        if generate_token(auth_code):
            logger.info("[main.py] ✅ Token generated successfully")
        else:
            logger.error("[main.py] ❌ Token generation failed")
            sys.exit(1)
    
    def fetch_historical_data(self, years: int = 15):
        """Step 3: Fetch historical data for all symbols"""
        logger.info(f"[main.py] 📊 Fetching {years} years of historical data...")
        
        try:
            # Load symbols
            symbols = load_symbols()
            if not symbols:
                logger.error("[main.py] ❌ No symbols found. Check symbols.csv")
                sys.exit(1)
            
            logger.info(f"[main.py] Found {len(symbols)} symbols")
            
            # Fetch data
            fetcher = HistoricalDataFetcher()
            fetcher.fetch_and_store_all(symbols, years)
            
            logger.info("[main.py] ✅ Historical data fetched")
            print(f"\n📋 Next step: python main.py --train --version <model_name>")
            
        except Exception as e:
            logger.error(f"[main.py] ❌ Data fetch failed: {e}")
            sys.exit(1)
    
    def train_model(
        self, 
        version: str, 
        end_date: Optional[str] = None,
        fast_mode: bool = True,
        n_features: int = 100,
        feature_selection_method: str = 'correlation'
    ):
        """
        Step 4: Train the model with optimization options
        
        Args:
            version: Model version name
            end_date: End date for training data
            fast_mode: Use fast training (HistGradientBoosting, fewer features)
            n_features: Number of features to select (if fast_mode=True)
            feature_selection_method: 'correlation', 'mutual_info', 'variance', 'random_forest'
        """
        logger.info("=" * 80)
        logger.info(f"[main.py] 🤖 Training model version: {version}")
        logger.info("=" * 80)
        
        if fast_mode:
            logger.info(f"[main.py] ⚡ FAST MODE ENABLED")
            logger.info(f"[main.py]    • Feature selection: {feature_selection_method}")
            logger.info(f"[main.py]    • Top features: {n_features}")
            logger.info(f"[main.py]    • Algorithm: HistGradientBoosting")
        else:
            logger.info(f"[main.py] 🐢 STANDARD MODE (all features, slower)")
        
        try:
            MODEL_CONFIG.set_version(version)
            
            # Load and prepare data
            logger.info(f"[main.py] 📂 Loading training data...")
            df = self._load_and_prepare_training_data(end_date)
            min_date = df['date'].min()
            cutoff_date = min_date + timedelta(days=365)
            df = df[df['date'] > cutoff_date].reset_index(drop=True)

            print('filter stocks in up trend')

            df = self.filter_df(df)

            logger.info(f"[main.py] ✅ Training data ready: {df.shape}")
            logger.info(f"[main.py]    Rows: {len(df):,}")
            logger.info(f"[main.py]    Features: {df.shape[1] - 1}")  # -1 for label column
            
            # Memory optimization: Sample if too large
            # if len(df) > 1_500_000:
            #     logger.warning(f"[main.py] ⚠️  Very large dataset ({len(df):,} rows)")
            #     logger.info(f"[main.py] 📊 Sampling 50% for faster training...")
            #     df = df.sample(frac=0.5, random_state=42)
            #     logger.info(f"[main.py]    Sampled to: {len(df):,} rows")
            
            # Feature selection (if fast mode)
            if fast_mode:
                logger.info(f"\n[main.py] 🔍 Performing feature selection...")
                df = self._select_features(
                    df, 
                    n_features=n_features,
                    method=feature_selection_method
                )
                logger.info(f"[main.py] ✅ Features reduced to: {df.shape[1] - 1}")
            
            # Train with optimized settings
            logger.info(f"\n[main.py] 🏋️  Starting model training...")
            import time
            start_time = time.time()
            
            # Pass fast_mode to trainer
            self.model_trainer.train(
                df, 
                MODEL_CONFIG,
                fast_mode=fast_mode
            )
            
            training_time = time.time() - start_time
            logger.info(f"\n[main.py] ✅ Model trained successfully!")
            logger.info(f"[main.py] ⏱️  Total training time: {training_time:.1f}s ({training_time/60:.1f}min)")
            
            # Show model info
            model_size = MODEL_CONFIG.model_path.stat().st_size / (1024 * 1024)
            logger.info(f"[main.py] 💾 Model saved: {MODEL_CONFIG.model_path}")
            logger.info(f"[main.py]    Size: {model_size:.2f} MB")
            
            print(f"\n{'='*80}")
            print(f"✅ TRAINING COMPLETE")
            print(f"{'='*80}")
            print(f"📊 Model: {version}")
            print(f"⏱️  Time: {training_time/60:.1f} minutes")
            print(f"💾 Saved: {MODEL_CONFIG.model_path}")
            print(f"\n📋 Next steps:")
            print(f"   • Backtest:  python main.py --backtest --version {version}")
            print(f"   • Screen:    python main.py --screener --version {version}")
            print(f"{'='*80}\n")
            
        except Exception as e:
            logger.error(f"[main.py] ❌ Training failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

    def _select_features(
        self, 
        df: pd.DataFrame, 
        n_features: int = 100,
        method: str = 'correlation'
    ) -> pd.DataFrame:
        """
        Select top N features for faster training.
        
        Args:
            df: Training dataframe
            n_features: Number of features to keep
            method: Selection method
            
        Returns:
            DataFrame with selected features + label
        """
        logger.info(f"[main.py] 🔍 Feature Selection ({method} method)")
        
        # Get target column
        target_col = 'label' if 'label' in df.columns else 'target_hit'
        
        # Separate features and target
        feature_cols = [col for col in df.columns if col not in ['symbol', 'date', target_col]]
        
        logger.info(f"[main.py]    Input features: {len(feature_cols)}")
        logger.info(f"[main.py]    Target features: {n_features}")
        
        if len(feature_cols) <= n_features:
            logger.info(f"[main.py] ✅ Already <= {n_features} features, skipping selection")
            return df
        
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        # Select features based on method
        if method == 'correlation':
            # Fastest: Correlation with target
            logger.info(f"[main.py]    Computing correlations...")
            correlations = X.corrwith(y).abs()
            selected = correlations.nlargest(n_features).index.tolist()
            
        elif method == 'variance':
            # Fast: High variance features
            logger.info(f"[main.py]    Computing variances...")
            variances = X.var()
            selected = variances.nlargest(n_features).index.tolist()
            
        elif method == 'mutual_info':
            # Medium speed: Mutual information
            from sklearn.feature_selection import mutual_info_classif
            logger.info(f"[main.py]    Computing mutual information...")
            
            # Sample for speed
            if len(X) > 100000:
                sample_idx = np.random.choice(len(X), 100000, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
            else:
                X_sample = X
                y_sample = y
            
            mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
            mi_df = pd.DataFrame({'feature': feature_cols, 'score': mi_scores})
            selected = mi_df.nlargest(n_features, 'score')['feature'].tolist()
            
        elif method == 'random_forest':
            # Slower but accurate: Random Forest importance
            from sklearn.ensemble import RandomForestClassifier
            logger.info(f"[main.py]    Training quick RF for feature importance...")
            
            # Sample for speed
            if len(X) > 100000:
                sample_idx = np.random.choice(len(X), 100000, replace=False)
                X_sample = X.iloc[sample_idx]
                y_sample = y.iloc[sample_idx]
            else:
                X_sample = X
                y_sample = y
            
            rf = RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                min_samples_split=100,
                n_jobs=-1,
                random_state=42
            )
            rf.fit(X_sample, y_sample)
            
            importance_df = pd.DataFrame({
                'feature': feature_cols,
                'importance': rf.feature_importances_
            }).sort_values('importance', ascending=False)
            
            selected = importance_df.head(n_features)['feature'].tolist()
        
        else:
            raise ValueError(f"Unknown selection method: {method}")
        
        # Log selected features
        logger.info(f"[main.py] ✅ Selected {len(selected)} features")
        logger.info(f"[main.py]    Top 10: {selected[:10]}")
        
        # Save selected features for reference
        selected_features_path = MODEL_CONFIG.model_dir / "selected_features.txt"
        with open(selected_features_path, 'w') as f:
            f.write(f"Feature Selection Method: {method}\n")
            f.write(f"Total Features Selected: {len(selected)}\n\n")
            f.write("Selected Features:\n")
            for i, feat in enumerate(selected, 1):
                f.write(f"{i:3d}. {feat}\n")
        
        logger.info(f"[main.py] 💾 Feature list saved: {selected_features_path}")
        
        # Return dataframe with selected features + target
        base_cols = ['symbol', 'date'] if 'symbol' in df.columns else []
        if 'date' not in df.columns:
            base_cols = []
        
        result_cols = base_cols + selected + [target_col]
        return df[result_cols]
    
    def run_backtest(
        self,
        version: str,
        capital: float = 500000,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        save_charts: bool = True
    ):
        """Step 5: Backtest the model"""
        logger.info("[main.py] 📈 Starting backtest...")
        
        try:
            MODEL_CONFIG.set_version(version)
            
            if isinstance(start_date, str):
                start_date = datetime.strptime(start_date, "%Y-%m-%d")
            if isinstance(end_date, str):
                end_date = datetime.strptime(end_date, "%Y-%m-%d")

            # now safe to subtract timedelta
            warmup_start_date = start_date - timedelta(days=365)

            # load extended data
            df = self._load_and_prepare_backtest_data(warmup_start_date, end_date)

            # after generating indicators/features
            df = df[df['date'] >= start_date].reset_index(drop=True)
            df = self.filter_df(df)

            df.to_csv('featured_df.csv')


            # Run backtest
            results = self.backtester.run(
                df=df,
                model_path=MODEL_CONFIG.model_path,
                initial_capital=capital,
                swing_config=SWING_CONFIG,
                save_charts=save_charts
            )
            
            # Display results
            self._display_backtest_results(results)
            
            logger.info("[main.py] ✅ Backtest complete")
            print(f"\n📋 Next step: python main.py --screener --version {version}")
            
        except Exception as e:
            logger.error(f"[main.py] ❌ Backtest failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def run_screener(
        self, 
        version: str, 
        min_probability: float = 0.7,
        screen_date: Optional[str] = None  # ← Added parameter!
    ):
        """Step 6: Screen stocks for trading opportunities"""
        logger.info("[main.py] 🔍 Screening stocks...")
        
        try:
            MODEL_CONFIG.set_version(version)
            
            # Run screener
            opportunities = self.screener.screen(
                model_path=MODEL_CONFIG.model_path,
                feature_pipeline=self.feature_pipeline,
                stock_filter=self.stock_filter,
                min_probability=min_probability,
                screen_date=screen_date  # ← Pass it through!
            )
            
            # Display results
            self._display_screening_results(opportunities)
            
            logger.info("[main.py] ✅ Screening complete")
            
        except Exception as e:
            logger.error(f"[main.py] ❌ Screening failed: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    
    def _load_and_prepare_training_data(
        self,
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Load and prepare data for training"""
        logger.info("[main.py] 📂 Loading training data...")
        
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by end date if specified
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        logger.info(f"[main.py] Date range: {df['date'].min()} to {df['date'].max()}")
        
        # Filter stocks
        filtered_symbols = self.stock_filter.filter_symbols(df)
        df = df[df['symbol'].isin(filtered_symbols)]
        
        # Add features with labels
        training_config = PipelineConfig(
            include_basic=True,
            include_candlestick=True,
            include_macd=True,
            include_advanced=True,
            include_enhanced=True,
            include_support_resistance=True,
            include_swing_labels=True,  # Include labels for training
            validate_input=True,
            validate_output=True,
            fail_on_error=True,
            skip_failed_symbols=False,
            show_progress=True
        )
        training_pipeline = FeaturePipeline(training_config)
        df = training_pipeline.transform(df)
        
        logger.info(f"[main.py] ✅ Training data ready: {df.shape}")
        return df
    
    def _load_and_prepare_backtest_data(
        self,
        start_date: Optional[str],
        end_date: Optional[str]
    ) -> pd.DataFrame:
        """Load and prepare data for backtesting"""
        logger.info("[main.py] 📂 Loading backtest data...")
        
        df = pd.read_parquet(HISTORICAL_DATA_FILE)
        df['date'] = pd.to_datetime(df['date'])
        
        # Filter by date range
        if start_date:
            df = df[df['date'] > pd.to_datetime(start_date)]
        else:
            # Default: need 250+ days buffer for features
            min_date = pd.Timestamp.now() - timedelta(days=365 + 250)
            df = df[df['date'] > min_date]
        
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        logger.info(f"[main.py] Date range: {df['date'].min()} to {df['date'].max()}")
        logger.info(f"[main.py] Total rows: {len(df)}")
        
        # Filter stocks
        filtered_symbols = self.stock_filter.filter_symbols(df)
        df = df[df['symbol'].isin(filtered_symbols)]
        
        # Add features (no labels)
        df = self.feature_pipeline.transform(df)
        
        # Get available features
        available_feature_cols = [col for col in FEATURE_COLUMNS if col in df.columns]
        
        if available_feature_cols:
            # Calculate completeness
            feature_completeness = df[available_feature_cols].notna().sum(axis=1) / len(available_feature_cols)
            rows_before = len(df)
            
            # Keep rows with at least 70% features present
            df = df[feature_completeness > 0.7].copy()  # ← Make sure .copy() is here
            rows_removed = rows_before - len(df)
            
            if rows_removed > 0:
                logger.info(f"[main.py] Removed {rows_removed} rows with >30% missing features")
            
            # Fill remaining NaN values - NOW OPERATING ON THE FILTERED DF
            logger.info(f"[main.py] Filling NaN values in {len(available_feature_cols)} features...")
            
            # FIX: Use .loc to ensure we're modifying the filtered DataFrame
            for col in available_feature_cols:
                df[col] = df[col].fillna(0)
            
            # Replace inf values
            for col in available_feature_cols:
                df[col] = df[col].replace([np.inf, -np.inf], [1e10, -1e10])
            
            # Verify no NaN remains
            remaining_nans = df[available_feature_cols].isnull().sum().sum()
            if remaining_nans > 0:
                logger.warning(f"[main.py] ⚠️  Still have {remaining_nans} NaN values after filling")
                for col in available_feature_cols:
                    df[col] = df[col].fillna(0)
            else:
                logger.info("[main.py] ✅ No NaN values in features")
        
        logger.info(f"[main.py] After filtering incomplete rows: {len(df)} rows")
        logger.info(f"[main.py] ✅ Backtest data ready: {df.shape}")
        return df
    
    def _display_backtest_results(self, results: dict):
        """Display backtest results"""
        print("\n" + "=" * 70)
        print("📊 BACKTEST RESULTS")
        print("=" * 70)
        
        metrics = results.get('metrics', {})
        
        print(f"\n💰 Performance:")
        print(f"  Initial Capital:    ₹{metrics.get('initial_capital', 0):,.0f}")
        print(f"  Final Capital:      ₹{metrics.get('final_capital', 0):,.0f}")
        print(f"  Total Return:       {metrics.get('total_return_pct', 0):.2f}%")
        print(f"  CAGR:              {metrics.get('cagr', 0):.2f}%")
        print(f"  Sharpe Ratio:      {metrics.get('sharpe_ratio', 0):.2f}")
        
        print(f"\n📈 Trades:")
        print(f"  Total Trades:      {metrics.get('total_trades', 0)}")
        print(f"  Winning Trades:    {metrics.get('winning_trades', 0)}")
        print(f"  Losing Trades:     {metrics.get('losing_trades', 0)}")
        print(f"  Win Rate:          {metrics.get('win_rate', 0):.1f}%")
        
        print(f"\n💵 P&L:")
        print(f"  Average Win:       ₹{metrics.get('avg_win', 0):,.0f}")
        print(f"  Average Loss:      ₹{metrics.get('avg_loss', 0):,.0f}")
        print(f"  Profit Factor:     {metrics.get('profit_factor', 0):.2f}")
        
        print("=" * 70)
    
    def _display_screening_results(self, opportunities: list):
        """Display screening results"""
        print("\n" + "=" * 70)
        print("🎯 SCREENING RESULTS")
        print("=" * 70)
        
        if not opportunities:
            print("\n❌ No opportunities found")
            return
        
        print(f"\n✅ Found {len(opportunities)} opportunities:\n")
        
        for i, opp in enumerate(opportunities, 1):
            print(f"{i}. {opp['symbol']}")
            print(f"   Price: ₹{opp['price']:.2f}")
            print(f"   Probability: {opp['probability']:.1%}")
            print(f"   Direction: {opp['direction']}")
            print()
        
        print("=" * 70)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='swing-trader',
        description='AI-based Swing Trading System'
    )
    
    # Commands
    parser.add_argument('--auth', action='store_true', help='Authenticate with FYERS')
    parser.add_argument('--token', help='Generate access token from auth code')
    parser.add_argument('--fetch-history', action='store_true', help='Fetch historical data')
    parser.add_argument('--train', action='store_true', help='Train model')
    parser.add_argument('--backtest', action='store_true', help='Run backtest')
    parser.add_argument('--screener', action='store_true', help='Screen for opportunities')
    

    # Options
    parser.add_argument('--version', help='Model version name')
    parser.add_argument('--years', type=int, default=15, help='Years of historical data')
    parser.add_argument('--capital', type=float, default=500000, help='Initial capital')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--screen-date', help='Date to screen stocks (YYYY-MM-DD). If not specified, uses latest date.')
    parser.add_argument('--min-probability', type=float, default=0.7, help='Minimum probability')
    
    args = parser.parse_args()
    
    # Initialize CLI
    try:
        cli = SwingTradingCLI()
    except Exception as e:
        logger.error(f"[main.py] ❌ Initialization failed: {e}")
        sys.exit(1)
    
    # Route commands
    try:
        if args.auth:
            cli.authenticate_fyers()
        
        elif args.token:
            cli.generate_access_token(args.token)
        
        elif args.fetch_history:
            cli.fetch_historical_data(args.years)
        
        elif args.train:
            if not args.version:
                parser.error("--train requires --version")
            cli.train_model(args.version, args.end_date)
        
        elif args.backtest:
            if not args.version:
                parser.error("--backtest requires --version")
            cli.run_backtest(
                args.version,
                args.capital,
                args.start_date,
                args.end_date
            )
        
        
        elif args.screener:
            if not args.version:
                parser.error("--screener requires --version")
            cli.run_screener(args.version, args.min_probability, args.screen_date)  # ← Added!
                
        else:
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.info("[main.py] \n⚠️ Cancelled by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"[main.py] ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
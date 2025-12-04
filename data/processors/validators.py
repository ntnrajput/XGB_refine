# data/validators.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from utils.logger import get_logger
from utils.exceptions import DataError

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Result of data validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, any]


class DataValidator:
    """
    Comprehensive data validation for OHLCV data.
    Checks for data quality issues, inconsistencies, and anomalies.
    """
    
    def __init__(self, strict_mode: bool = False):
        """
        Initialize validator.
        
        Args:
            strict_mode: If True, warnings are treated as errors
        """
        self.strict_mode = strict_mode
    
    def validate_ohlcv(self, df: pd.DataFrame) -> ValidationResult:
        """
        Validate OHLCV DataFrame.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            ValidationResult with errors, warnings, and stats
        """
        logger.info("[validators.py] [validators.py] 🔍 Validating OHLCV data...")
        
        errors = []
        warnings = []
        stats = {}
        
        # 1. Schema validation
        schema_errors = self._validate_schema(df)
        errors.extend(schema_errors)
        
        if errors and self.strict_mode:
            return ValidationResult(False, errors, warnings, stats)
        
        # 2. Data type validation
        dtype_errors, dtype_warnings = self._validate_dtypes(df)
        errors.extend(dtype_errors)
        warnings.extend(dtype_warnings)
        
        # 3. OHLC relationships
        ohlc_errors, ohlc_warnings = self._validate_ohlc_relationships(df)
        errors.extend(ohlc_errors)
        warnings.extend(ohlc_warnings)
        
        # 4. Price validation
        price_errors, price_warnings = self._validate_prices(df)
        errors.extend(price_errors)
        warnings.extend(price_warnings)
        
        # 5. Volume validation
        volume_warnings = self._validate_volume(df)
        warnings.extend(volume_warnings)
        
        # 6. Date validation
        date_errors, date_warnings = self._validate_dates(df)
        errors.extend(date_errors)
        warnings.extend(date_warnings)
        
        # 7. Duplicate validation
        dup_errors, dup_warnings = self._validate_duplicates(df)
        errors.extend(dup_errors)
        warnings.extend(dup_warnings)
        
        # 8. Missing data validation
        missing_warnings = self._validate_missing_data(df)
        warnings.extend(missing_warnings)
        
        # 9. Statistical anomalies
        anomaly_warnings = self._detect_anomalies(df)
        warnings.extend(anomaly_warnings)
        
        # 10. Calculate statistics
        stats = self._calculate_stats(df)
        
        # Determine if valid
        is_valid = len(errors) == 0
        if self.strict_mode:
            is_valid = is_valid and len(warnings) == 0
        
        # Log results
        self._log_validation_results(is_valid, errors, warnings, stats)
        
        return ValidationResult(is_valid, errors, warnings, stats)
    
    def _validate_schema(self, df: pd.DataFrame) -> List[str]:
        """Validate required columns exist"""
        errors = []
        
        required_columns = ['symbol', 'date', 'open', 'high', 'low', 'close']
        optional_columns = ['volume']
        
        missing_required = [col for col in required_columns if col not in df.columns]
        
        if missing_required:
            errors.append(f"Missing required columns: {missing_required}")
        
        if df.empty:
            errors.append("DataFrame is empty")
        
        return errors
    
    def _validate_dtypes(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate data types of columns"""
        errors = []
        warnings = []
        
        # Check numeric columns
        numeric_cols = ['open', 'high', 'low', 'close']
        for col in numeric_cols:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    errors.append(f"Column '{col}' must be numeric, got {df[col].dtype}")
        
        # Check volume if present
        if 'volume' in df.columns:
            if not pd.api.types.is_numeric_dtype(df['volume']):
                warnings.append(f"Column 'volume' should be numeric, got {df['volume'].dtype}")
        
        # Check date column
        if 'date' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['date']):
                try:
                    pd.to_datetime(df['date'])
                except Exception as e:
                    errors.append(f"Column 'date' cannot be converted to datetime: {e}")
        
        # Check symbol column
        if 'symbol' in df.columns:
            if not (pd.api.types.is_string_dtype(df['symbol']) or pd.api.types.is_object_dtype(df['symbol'])):
                warnings.append(f"Column 'symbol' should be string, got {df['symbol'].dtype}")
        
        return errors, warnings
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate OHLC logical relationships"""
        errors = []
        warnings = []
        
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return errors, warnings
        
        # High should be >= all others
        high_violations = (
            (df['high'] < df['open']) |
            (df['high'] < df['low']) |
            (df['high'] < df['close'])
        ).sum()
        
        if high_violations > 0:
            pct = high_violations / len(df) * 100
            msg = f"High price violations: {high_violations} rows ({pct:.2f}%)"
            if pct > 1:
                errors.append(msg)
            else:
                warnings.append(msg)
        
        # Low should be <= all others
        low_violations = (
            (df['low'] > df['open']) |
            (df['low'] > df['high']) |
            (df['low'] > df['close'])
        ).sum()
        
        if low_violations > 0:
            pct = low_violations / len(df) * 100
            msg = f"Low price violations: {low_violations} rows ({pct:.2f}%)"
            if pct > 1:
                errors.append(msg)
            else:
                warnings.append(msg)
        
        # Open and Close should be between High and Low
        range_violations = (
            (df['open'] > df['high']) |
            (df['open'] < df['low']) |
            (df['close'] > df['high']) |
            (df['close'] < df['low'])
        ).sum()
        
        if range_violations > 0:
            pct = range_violations / len(df) * 100
            msg = f"Open/Close range violations: {range_violations} rows ({pct:.2f}%)"
            if pct > 1:
                errors.append(msg)
            else:
                warnings.append(msg)
        
        return errors, warnings
    
    def _validate_prices(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate price values"""
        errors = []
        warnings = []
        
        price_cols = ['open', 'high', 'low', 'close']
        available_price_cols = [col for col in price_cols if col in df.columns]
        
        for col in available_price_cols:
            # Check for negative prices
            negative_count = (df[col] <= 0).sum()
            if negative_count > 0:
                errors.append(f"Column '{col}' has {negative_count} non-positive values")
            
            # Check for extremely high prices (possible data errors)
            extreme_count = (df[col] > 1000000).sum()
            if extreme_count > 0:
                warnings.append(f"Column '{col}' has {extreme_count} extremely high values (>1M)")
            
            # Check for suspiciously low prices
            low_count = (df[col] < 0.01).sum()
            if low_count > 0:
                warnings.append(f"Column '{col}' has {low_count} very low values (<0.01)")
        
        return errors, warnings
    
    def _validate_volume(self, df: pd.DataFrame) -> List[str]:
        """Validate volume data"""
        warnings = []
        
        if 'volume' not in df.columns:
            warnings.append("Volume column not present")
            return warnings
        
        # Check for negative volume
        negative_count = (df['volume'] < 0).sum()
        if negative_count > 0:
            warnings.append(f"Volume has {negative_count} negative values")
        
        # Check for zero volume
        zero_count = (df['volume'] == 0).sum()
        if zero_count > 0:
            pct = zero_count / len(df) * 100
            if pct > 10:
                warnings.append(f"Volume has {zero_count} zero values ({pct:.1f}%)")
        
        # Check for suspiciously high volume spikes
        if len(df) > 20:
            volume_median = df['volume'].median()
            extreme_volume = (df['volume'] > volume_median * 100).sum()
            if extreme_volume > 0:
                warnings.append(f"Found {extreme_volume} extreme volume spikes (>100x median)")
        
        return warnings
    
    def _validate_dates(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Validate date column"""
        errors = []
        warnings = []
        
        if 'date' not in df.columns:
            errors.append("Date column not present")
            return errors, warnings
        
        # Convert to datetime if needed
        dates = pd.to_datetime(df['date'])
        
        # Check for future dates
        future_dates = (dates > pd.Timestamp.now()).sum()
        if future_dates > 0:
            warnings.append(f"Found {future_dates} future dates")
        
        # Check for very old dates (before 1990)
        old_dates = (dates < pd.Timestamp('1990-01-01')).sum()
        if old_dates > 0:
            warnings.append(f"Found {old_dates} dates before 1990")
        
        # Check date ordering within each symbol
        if 'symbol' in df.columns:
            for symbol, group in df.groupby('symbol'):
                group_dates = pd.to_datetime(group['date'])
                if not group_dates.is_monotonic_increasing:
                    warnings.append(f"Dates not sorted for symbol: {symbol}")
                    break  # Only report once
        
        return errors, warnings
    
    def _validate_duplicates(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """Check for duplicate records"""
        errors = []
        warnings = []
        
        if 'symbol' in df.columns and 'date' in df.columns:
            duplicates = df.duplicated(subset=['symbol', 'date'], keep=False).sum()
            
            if duplicates > 0:
                pct = duplicates / len(df) * 100
                msg = f"Found {duplicates} duplicate symbol-date combinations ({pct:.2f}%)"
                
                if pct > 5:
                    errors.append(msg)
                else:
                    warnings.append(msg)
        
        return errors, warnings
    
    def _validate_missing_data(self, df: pd.DataFrame) -> List[str]:
        """Check for missing data"""
        warnings = []
        
        # Check each column for missing values
        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                pct = missing_count / len(df) * 100
                
                if pct > 50:
                    warnings.append(f"Column '{col}' has {missing_count} missing values ({pct:.1f}%)")
                elif pct > 10:
                    warnings.append(f"Column '{col}' has {missing_count} missing values ({pct:.1f}%)")
        
        return warnings
    
    def _detect_anomalies(self, df: pd.DataFrame) -> List[str]:
        """Detect statistical anomalies"""
        warnings = []
        
        price_cols = ['open', 'high', 'low', 'close']
        available_cols = [col for col in price_cols if col in df.columns]
        
        if not available_cols:
            return warnings
        
        # Check for suspicious price jumps within each symbol
        if 'symbol' in df.columns and 'date' in df.columns:
            for symbol, group in df.groupby('symbol'):
                if len(group) < 2:
                    continue
                
                group = group.sort_values('date')
                
                for col in available_cols:
                    # Calculate daily returns
                    returns = group[col].pct_change().abs()
                    
                    # Flag returns > 50% in a single day
                    extreme_moves = (returns > 0.5).sum()
                    if extreme_moves > 0:
                        warnings.append(
                            f"{symbol}: {extreme_moves} extreme price moves (>50%) in {col}"
                        )
                        break  # Only report once per symbol
        
        return warnings
    
    def _calculate_stats(self, df: pd.DataFrame) -> Dict:
        """Calculate summary statistics"""
        stats = {
            'total_rows': len(df),
            'total_symbols': df['symbol'].nunique() if 'symbol' in df.columns else 1,
            'date_range': None,
            'missing_data_pct': (df.isnull().sum().sum() / (len(df) * len(df.columns))) * 100,
        }
        
        if 'date' in df.columns:
            dates = pd.to_datetime(df['date'])
            stats['date_range'] = (dates.min(), dates.max())
            stats['total_days'] = (dates.max() - dates.min()).days
        
        if 'volume' in df.columns:
            stats['avg_volume'] = df['volume'].mean()
            stats['zero_volume_pct'] = (df['volume'] == 0).sum() / len(df) * 100
        
        # Price statistics
        if 'close' in df.columns:
            stats['avg_price'] = df['close'].mean()
            stats['min_price'] = df['close'].min()
            stats['max_price'] = df['close'].max()
        
        return stats
    
    def _log_validation_results(
        self,
        is_valid: bool,
        errors: List[str],
        warnings: List[str],
        stats: Dict
    ):
        """Log validation results"""
        
        if is_valid:
            logger.info("[validators.py] [validators.py] ✅ Data validation PASSED")
        else:
            logger.error("[validators.py] [validators.py] ❌ Data validation FAILED")
        
        if errors:
            logger.error(f"[validators.py] [validators.py] \n🚨 Errors ({len(errors)}):")
            for i, error in enumerate(errors, 1):
                logger.error(f"[validators.py] [validators.py]   {i}. {error}")
        
        if warnings:
            logger.warning(f"[validators.py] [validators.py] \n⚠️  Warnings ({len(warnings)}):")
            for i, warning in enumerate(warnings[:10], 1):  # Limit to first 10
                logger.warning(f"[validators.py] [validators.py]   {i}. {warning}")
            
            if len(warnings) > 10:
                logger.warning(f"[validators.py] [validators.py]   ... and {len(warnings) - 10} more warnings")
        
        logger.info(f"[validators.py] [validators.py] \n📊 Statistics:")
        for key, value in stats.items():
            if isinstance(value, tuple):
                logger.info(f"[validators.py] [validators.py]   {key}: {value[0]} to {value[1]}")
            elif isinstance(value, float):
                logger.info(f"[validators.py] [validators.py]   {key}: {value:.2f}")
            else:
                logger.info(f"[validators.py] [validators.py]   {key}: {value}")


class SymbolValidator:
    """Validate symbol list and format"""
    
    @staticmethod
    def validate_symbol_format(symbol: str) -> Tuple[bool, Optional[str]]:
        """
        Validate individual symbol format.
        
        Returns:
            (is_valid, error_message)
        """
        if not symbol or not isinstance(symbol, str):
            return False, "Symbol must be a non-empty string"
        
        symbol = symbol.strip()
        
        if len(symbol) == 0:
            return False, "Symbol is empty"
        
        if len(symbol) > 50:
            return False, "Symbol too long (>50 characters)"
        
        # Check for valid characters (alphanumeric, dash, colon)
        if not all(c.isalnum() or c in ['-', ':', '_'] for c in symbol):
            return False, "Symbol contains invalid characters"
        
        return True, None
    
    @staticmethod
    def validate_symbol_list(symbols: List[str]) -> ValidationResult:
        """Validate list of symbols"""
        errors = []
        warnings = []
        
        if not symbols:
            errors.append("Symbol list is empty")
            return ValidationResult(False, errors, warnings, {})
        
        valid_symbols = []
        invalid_symbols = []
        
        for symbol in symbols:
            is_valid, error = SymbolValidator.validate_symbol_format(symbol)
            if is_valid:
                valid_symbols.append(symbol)
            else:
                invalid_symbols.append((symbol, error))
        
        if invalid_symbols:
            warnings.append(f"Found {len(invalid_symbols)} invalid symbols")
            for symbol, error in invalid_symbols[:5]:  # Show first 5
                warnings.append(f"  {symbol}: {error}")
        
        # Check for duplicates
        duplicates = len(symbols) - len(set(symbols))
        if duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate symbols")
        
        stats = {
            'total_symbols': len(symbols),
            'valid_symbols': len(valid_symbols),
            'invalid_symbols': len(invalid_symbols),
            'duplicates': duplicates
        }
        
        is_valid = len(valid_symbols) > 0 and len(errors) == 0
        
        return ValidationResult(is_valid, errors, warnings, stats)


class FeatureDataValidator:
    """Validate feature data before model training"""
    
    @staticmethod
    def validate_features(df: pd.DataFrame, feature_columns: List[str]) -> ValidationResult:
        """
        Validate feature data for model training.
        
        Args:
            df: DataFrame with features
            feature_columns: Expected feature column names
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        stats = {}
        
        # Check if all required features exist
        missing_features = [f for f in feature_columns if f not in df.columns]
        if missing_features:
            if len(missing_features) > 20:
                errors.append(f"Missing {len(missing_features)} features (showing first 20): {missing_features[:20]}")
            else:
                errors.append(f"Missing features: {missing_features}")
        
        available_features = [f for f in feature_columns if f in df.columns]
        
        # Check for infinite values
        inf_counts = {}
        for col in available_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                inf_count = np.isinf(df[col]).sum()
                if inf_count > 0:
                    inf_counts[col] = inf_count
        
        if inf_counts:
            total_inf = sum(inf_counts.values())
            warnings.append(f"Found {total_inf} infinite values across {len(inf_counts)} features")
        
        # Check for NaN values
        nan_counts = df[available_features].isnull().sum()
        high_nan_features = nan_counts[nan_counts > len(df) * 0.5].index.tolist()
        
        if high_nan_features:
            errors.append(f"Features with >50% missing values: {high_nan_features}")
        
        # Check for zero variance
        zero_var_features = []
        for col in available_features:
            if pd.api.types.is_numeric_dtype(df[col]):
                if df[col].var() < 1e-10:
                    zero_var_features.append(col)
        
        if zero_var_features:
            warnings.append(f"Found {len(zero_var_features)} zero-variance features")
        
        # Calculate statistics
        stats = {
            'total_features': len(feature_columns),
            'available_features': len(available_features),
            'missing_features': len(missing_features),
            'features_with_inf': len(inf_counts),
            'features_with_high_nan': len(high_nan_features),
            'zero_variance_features': len(zero_var_features)
        }
        
        is_valid = len(errors) == 0 and len(available_features) > 0
        
        return ValidationResult(is_valid, errors, warnings, stats)
# features/validators.py - Feature validation

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass

from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class ValidationResult:
    """Validation result"""
    is_valid: bool
    warnings: List[str]
    errors: List[str]
    info: Dict


class FeatureValidator:
    """Validate feature data quality"""

    def __init__(self, strict: bool = True):
        """
        Initialize validator.

        Args:
            strict: If True, raise errors. If False, only warn.
        """
        self.strict = strict
        self.validation_results = []

    def validate(self, df: pd.DataFrame, raise_on_error: bool = None) -> ValidationResult:
        """
        Validate feature dataframe.

        Args:
            df: DataFrame to validate
            raise_on_error: Whether to raise exception on errors (overrides self.strict if provided)

        Returns:
            ValidationResult with validation details
        """

        # Use provided raise_on_error or fall back to strict setting
        should_raise = raise_on_error if raise_on_error is not None else self.strict

        warnings = []
        errors = []
        info = {}

        # Check 1: Empty dataframe
        if df.empty:
            errors.append("DataFrame is empty")
            if should_raise:
                raise ValueError("Input dataframe is empty")

        # Check 2: Required columns
        required_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            if should_raise:
                raise ValueError(f"Missing columns: {missing_cols}")

        if not errors:  # Only check further if basic structure is valid

            # Check 3: Null values
            null_counts = df.isnull().sum()
            null_cols = null_counts[null_counts > 0]

            if len(null_cols) > 0:
                total_nulls = null_counts.sum()
                pct = (total_nulls / (len(df) * len(df.columns))) * 100
                warnings.append(f"Found {total_nulls:,} null values ({pct:.2f}%)")
                info['null_counts'] = null_cols.to_dict()

            # ✅ Check 4: Infinite values (fixed)
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            inf_counts = {}

            for col in numeric_cols:
                # Convert to NumPy float array for consistent scalar results
                try:
                    inf_count = np.isinf(df[col].to_numpy(dtype=float)).sum()
                except Exception:
                    inf_count = 0  # if conversion fails (non-numeric), skip

                if inf_count > 0:
                    inf_counts[col] = int(inf_count)

            if len(inf_counts) > 0:
                total_infs = sum(inf_counts.values())
                warnings.append(f"Found {total_infs:,} infinite values across {len(inf_counts)} columns")
                info['inf_counts'] = inf_counts

            # Check 5: Zero variance features
            zero_var_features = []
            for col in numeric_cols:
                try:
                    # Ensure Series and scalar comparison
                    unique_count = pd.Series(df[col]).nunique(dropna=False)
                    if int(unique_count) == 1:
                        zero_var_features.append(col)
                except Exception as e:
                    logger.warning(f"⚠️  Could not evaluate zero variance for {col}: {e}")

            if zero_var_features:
                warnings.append(f"Found {len(zero_var_features)} zero-variance features")
                info['zero_variance_features'] = zero_var_features

            # Check 6: Data types
            expected_types = {
                'symbol': 'object',
                'date': 'datetime64[ns]',
            }

            type_issues = []
            for col, expected_type in expected_types.items():
                if col in df.columns:
                    actual_type = str(df[col].dtype)
                    if actual_type != expected_type:
                        type_issues.append(f"{col}: expected {expected_type}, got {actual_type}")

            if type_issues:
                warnings.append(f"Data type mismatches: {type_issues}")

            # Info
            info['shape'] = df.shape
            info['symbols'] = df['symbol'].nunique() if 'symbol' in df.columns else 0
            info['date_range'] = (
                str(df['date'].min().date()) + ' to ' + str(df['date'].max().date())
                if 'date' in df.columns and len(df) > 0 else 'N/A'
            )

        # Create result
        is_valid = len(errors) == 0

        result = ValidationResult(
            is_valid=is_valid,
            warnings=warnings,
            errors=errors,
            info=info
        )

        # Log results
        if errors:
            logger.error(f"❌ Validation failed: {len(errors)} errors")
            for error in errors:
                logger.error(f"   • {error}")

        if warnings:
            logger.warning(f"⚠️  Validation warning: {warnings[0]}")
            if len(warnings) > 1:
                logger.warning(f"   (+ {len(warnings) - 1} more warnings)")

        if is_valid and not warnings:
            logger.info(
                f"✅ Validation passed: {info.get('shape', (0, 0))[0]} rows, "
                f"{info.get('shape', (0, 0))[1] - len(required_cols)} features"
            )

        if warnings and not self.strict:
            logger.warning("⚠️  Feature validation warnings:")
            for warning in warnings:
                logger.warning(f"   • {warning}")

        return result

    def validate_feature_names(self, feature_names: List[str]) -> bool:
        """Validate feature names"""

        invalid = []

        for name in feature_names:
            # Check for valid characters
            if not name.replace('_', '').replace('-', '').isalnum():
                invalid.append(name)

        if invalid:
            logger.warning(f"⚠️  Invalid feature names: {invalid}")
            return False

        return True

    def check_feature_availability(self, df: pd.DataFrame, required_features: List[str]) -> Dict:
        """Check which required features are available"""

        available = [f for f in required_features if f in df.columns]
        missing = [f for f in required_features if f not in df.columns]

        return {
            'available': available,
            'missing': missing,
            'availability_pct': len(available) / len(required_features) * 100 if required_features else 0
        }


__all__ = ['FeatureValidator', 'ValidationResult']

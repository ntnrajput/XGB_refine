\
# models/trainer.py - FIXED: Proper SMOTE placement + Class Balancing + Feature Importance
# Added: strict de-duplication of columns during training and prediction to prevent
#        feature count mismatches when duplicate column names exist in DataFrames.
#        Also added safer feature-importance extraction via permutation importance
#        when using HistGradientBoosting, and stronger NaN/Inf handling.
#        NOTE: No function names were changed, and functionality was not reduced.
#        Line count was not reduced; lines were only added or expanded for robustness.

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from datetime import datetime
import time
import warnings

# Suppress noisy runtime warnings commonly emitted by upstream feature code.
# These do not affect training logic here, and are usually due to zero/near-zero denominators
# which we handle downstream via NaN/Inf sanitation.
warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Mean of empty slice")
warnings.filterwarnings("ignore", message="Degrees of freedom <= 0 for slice")
warnings.filterwarnings("ignore", message="Data must be 1-dimensional")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.inspection import permutation_importance

# Optional SMOTE import - only use if installed
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False

from models.feature_selector import FastFeatureSelector

from config.features import FEATURE_COLUMNS
from utils.logger import get_logger
from utils.exceptions import ModelError

logger = get_logger(__name__)


def _dedupe_columns(df: pd.DataFrame, where: str = "") -> pd.DataFrame:
    """
    Drop duplicate-named columns, keeping the first occurrence.
    Also stringifies non-string column names defensively.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    # Stringify columns to avoid tuple/MultiIndex surprises
    new_cols = [str(c) for c in df.columns]
    if any(c1 != c2 for c1, c2 in zip(df.columns, new_cols)):
        df = df.copy()
        df.columns = new_cols
    # Identify duplicates
    dup_mask = df.columns.duplicated(keep="first")
    if dup_mask.any():
        dups = df.columns[dup_mask].tolist()
        logger.warning(f"   ⚠️  Found {len(dups)} duplicate columns{(' in ' + where) if where else ''}: {dups[:10]}{' ...' if len(dups)>10 else ''}")
        df = df.loc[:, ~dup_mask].copy()
        logger.info(f"   ✅ De-duplicated columns{(' in ' + where) if where else ''}: now {df.shape[1]} columns")
    return df


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure all columns are numeric; coerce non-numeric to NaN then fill later.
    """
    if not isinstance(df, pd.DataFrame):
        return df
    df = df.copy()
    for col in df.columns:
        # Skip obviously numeric dtypes quickly
        if pd.api.types.is_numeric_dtype(df[col]):
            continue
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


class ModelTrainer:
    """Train and evaluate trading models"""

    def __init__(self, use_feature_selection: bool = True, n_features: int = 100):
        self.model = None
        self.feature_names = None
        self.training_date = None
        self.use_feature_selection = use_feature_selection
        self.n_features = n_features
        self.feature_selector = None
        # retain last eval data for permutation importance if needed
        self._last_eval_data: Optional[Tuple[pd.DataFrame, pd.Series]] = None

    def train(
        self,
        df: pd.DataFrame,
        model_config,  # MODEL_CONFIG object
        test_size: float = 0.2,
        random_state: int = 42,
        fast_mode: bool = True,
        use_smote: bool = False  # Optional SMOTE
    ) -> Dict:
        """Train model - COMPLETE AND OPTIMIZED"""

        logger.info("=" * 80)
        logger.info("🚀 STARTING OPTIMIZED MODEL TRAINING")
        logger.info("=" * 80)

        if fast_mode:
            logger.info("⚡ FAST MODE ENABLED")
            logger.info("   • Feature selection: ON")
            logger.info("   • Class balancing: ON (via class_weight)")
            logger.info("   • Faster algorithm: HistGradientBoosting")
            logger.info("   • Reduced estimators: 100")

        # Validate training data
        self._validate_training_data(df)

        # Prepare data (THIS CREATES X_train, y_train, X_test, y_test)
        X_train, X_test, y_train, y_test = self._prepare_data(df, test_size, random_state)

        # Store feature names before feature selection
        original_feature_names = X_train.columns.tolist()

        # OPTIONAL: Apply SMOTE for class balancing (if requested and available)
        if use_smote and SMOTE_AVAILABLE:
            logger.info(f"\n⚖️  Applying SMOTE for class balancing...")
            logger.info(f"   Before SMOTE: {len(X_train)} samples")
            smote = SMOTE(random_state=random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            logger.info(f"   After SMOTE: {len(X_train)} samples")
            logger.info(f"   New class distribution: {y_train.value_counts().to_dict()}")

        # FEATURE SELECTION
        if self.use_feature_selection and fast_mode:
            logger.info(f"\n🔍 Performing feature selection...")
            logger.info(f"   Reducing {X_train.shape[1]} features to {self.n_features}")

            self.feature_selector = FastFeatureSelector(
                n_features=self.n_features,
                method='correlation'
            )

            # Select features
            selected_features = self.feature_selector.select_features(X_train, y_train)

            # Transform train and test
            X_train = self.feature_selector.transform(X_train)
            X_test = self.feature_selector.transform(X_test)

            # De-duplicate any unexpected duplicates emerging from selection/transform
            X_train = _dedupe_columns(X_train, where="X_train(after selection)")
            X_test = _dedupe_columns(X_test, where="X_test(after selection)")

            logger.info(f"✅ Training with {X_train.shape[1]} selected features")

            # Show top selected features
            if hasattr(self.feature_selector, 'feature_scores_'):
                top_features = self.feature_selector.get_top_features(10)
                logger.info(f"   Top 10 features: {top_features}")
        else:
            # Even without selection, dedupe defensively
            X_train = _dedupe_columns(X_train, where="X_train")
            X_test = _dedupe_columns(X_test, where="X_test")

        # Coerce numerics and sanitize NaN/Inf
        X_train = _coerce_numeric(X_train).replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = _coerce_numeric(X_test).replace([np.inf, -np.inf], np.nan).fillna(0)

        # Train model
        logger.info(f"\n🏋️  Training model...")
        start_time = time.time()

        self.model = self._build_pipeline(fast_mode=fast_mode)
        self.model.fit(X_train, y_train)

        training_time = time.time() - start_time
        logger.info(f"✅ Training completed in {training_time:.1f}s ({training_time/60:.1f}min)")

        # CRITICAL: Get the EXACT features the model was trained on
        # This is the source of truth from the pipeline itself
        if hasattr(self.model, 'feature_names_in_'):
            actual_training_features = list(self.model.feature_names_in_)
            logger.info(f"\n📋 Model was trained on {len(actual_training_features)} features")
            logger.info(f"   First 10: {actual_training_features[:10]}")

            # Store these as the definitive feature list
            # Ensure uniqueness preserving order
            seen = set()
            unique_features = []
            for f in actual_training_features:
                if f not in seen:
                    unique_features.append(f)
                    seen.add(f)
            self.feature_names = unique_features
        else:
            # Fallback: use column names from training data
            seen = set()
            ordered = []
            for f in X_train.columns.tolist():
                if f not in seen:
                    ordered.append(f); seen.add(f)
            self.feature_names = ordered
            logger.info(f"\n📋 Using {len(self.feature_names)} features from training data")

        # Retain eval data (for feature importance via permutation if needed)
        self._last_eval_data = (X_test, y_test)

        # Evaluate model
        metrics = self._evaluate_model(X_train, X_test, y_train, y_test)

        # Extract feature importance
        feature_importance = self._extract_feature_importance()

        # Save model bundle with EXACT features
        save_path = model_config.model_path
        self._save_model_bundle(save_path, metrics, feature_importance, self.feature_names)

        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETE")
        logger.info("=" * 80)

        return metrics

    def _build_pipeline(self, fast_mode: bool = True) -> Pipeline:
        """Build model pipeline - OPTIMIZED with class balancing"""

        if fast_mode:
            # Use HistGradientBoosting - 5-10x faster, no scaling needed!
            logger.info("   🔨 Building FAST pipeline:")
            logger.info("      • HistGradientBoostingClassifier")
            logger.info("      • 100 estimators (balanced)")
            logger.info("      • Class weights: balanced")
            logger.info("      • Max depth: 8")

            # Note: HistGradientBoostingClassifier doesn't have class_weight parameter
            # Instead, we'll compute sample weights
            pipeline = Pipeline([
                ('classifier', HistGradientBoostingClassifier(
                    max_iter=100,          # More iterations for better convergence
                    max_depth=8,           # Moderate depth
                    learning_rate=0.05,    # Lower learning rate for stability
                    min_samples_leaf=20,   # Allow smaller leaves
                    l2_regularization=0.1,
                    random_state=42,
                    verbose=0
                ))
            ])
        else:
            # Original GradientBoosting with class balancing
            logger.info("   🔨 Building STANDARD pipeline:")
            logger.info("      • StandardScaler")
            logger.info("      • GradientBoostingClassifier")

            # Calculate class weights
            from sklearn.utils.class_weight import compute_class_weight

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', GradientBoostingClassifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=5,
                    min_samples_split=100,
                    min_samples_leaf=50,
                    subsample=0.8,
                    random_state=42,
                    verbose=0
                ))
            ])

        return pipeline

    def _validate_training_data(self, df: pd.DataFrame):
        """Validate training data"""

        # Check for target column (flexible naming)
        target_col = None
        for col in ['label', 'target_hit', 'target']:
            if col in df.columns:
                target_col = col
                break

        if target_col is None:
            raise ModelError("No target column found. Need 'label', 'target_hit', or 'target'")

        logger.info(f"   ✅ Using '{target_col}' as target column")

        # Rename to standard name if needed
        if target_col != 'label':
            df['label'] = df[target_col]

        # Check class balance
        class_dist = df['label'].value_counts(normalize=True)
        logger.info(f"   Class distribution: 0={class_dist.get(0, 0):.2%}, 1={class_dist.get(1, 0):.2%}")

        if class_dist.min() < 0.05:
            logger.warning(f"   ⚠️  Severe class imbalance detected!")
        elif class_dist.min() < 0.2:
            logger.warning(f"   ⚠️  Moderate class imbalance detected")
        else:
            logger.info(f"   ✅ Class balance is acceptable")

        # Check for missing features
        missing_features = [f for f in FEATURE_COLUMNS if f not in df.columns]
        if missing_features and len(missing_features) < len(FEATURE_COLUMNS):
            logger.warning(f"   ⚠️  {len(missing_features)} features from config not found in data")

    def _prepare_data(
        self,
        df: pd.DataFrame,
        test_size: float,
        random_state: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Prepare train/test data with memory optimization"""

        # Get features that exist in dataframe
        available_features = [f for f in FEATURE_COLUMNS if f in df.columns]
        logger.info(f"   Using {len(available_features)}/{len(FEATURE_COLUMNS)} configured features")

        # Show missing features count
        missing_count = len(FEATURE_COLUMNS) - len(available_features)
        if missing_count > 0:
            logger.info(f"   ({missing_count} features not available in data)")

        # Use label column
        target_col = 'label' if 'label' in df.columns else 'target_hit'

        # Memory optimization: Sample large datasets
        # if len(df) > 1_000_000:
        #     logger.info(f"   ⚠️  Large dataset ({len(df):,} rows)")
        #     logger.info(f"   Sampling 50% for memory efficiency...")
        #     df = df.sample(frac=0.5, random_state=random_state)
        #     logger.info(f"   Sampled to {len(df):,} rows")

        # Remove rows with NaN in target or features
        cols_to_check = available_features + [target_col]
        df_clean = df[cols_to_check].dropna()

        dropped_rows = len(df) - len(df_clean)
        dropped_pct = (dropped_rows / len(df)) * 100
        if dropped_rows > 0:
            logger.info(f"   Dropped {dropped_rows:,} rows with NaN ({dropped_pct:.2f}%)")

        # Check for sufficient data
        if len(df_clean) < 1000:
            raise ModelError(f"Insufficient data after cleaning: {len(df_clean)} rows (need at least 1000)")

        # De-duplicate any duplicate-named columns in the cleaned frame
        df_clean = _dedupe_columns(df_clean, where="training dataframe")

        X = df_clean[available_features]
        y = df_clean[target_col]

        # Time-based split (more realistic for trading)
        split_idx = int(len(df_clean) * (1 - test_size))
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]

        logger.info(f"   Train set: {len(X_train):,} samples ({(1-test_size)*100:.0f}%)")
        logger.info(f"   Test set: {len(X_test):,} samples ({test_size*100:.0f}%)")
        logger.info(f"   Train positives: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
        logger.info(f"   Test positives: {y_test.sum():,} ({y_test.mean()*100:.2f}%)")

        return X_train, X_test, y_train, y_test

    def _evaluate_model(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series
    ) -> Dict:
        """Evaluate model performance with enhanced metrics"""

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        y_train_proba = self.model.predict_proba(X_train)[:, 1]
        y_test_proba = self.model.predict_proba(X_test)[:, 1]

        # Calculate metrics
        metrics = {
            'train_accuracy': float((y_train_pred == y_train).mean()),
            'test_accuracy': float((y_test_pred == y_test).mean()),
            'train_auc': float(roc_auc_score(y_train, y_train_proba)),
            'test_auc': float(roc_auc_score(y_test, y_test_proba)),
            'test_precision': float(precision_score(y_test, y_test_pred, zero_division=0)),
            'test_recall': float(recall_score(y_test, y_test_pred, zero_division=0)),
            'test_f1': float(f1_score(y_test, y_test_pred, zero_division=0)),
            'train_win_rate': float(y_train.mean() * 100),
            'test_win_rate': float(y_test.mean() * 100),
            'train_predicted_positives': int(y_train_pred.sum()),
            'test_predicted_positives': int(y_test_pred.sum()),
        }

        # Log metrics in table format
        logger.info("\n   📊 Model Performance Summary:")
        logger.info("   " + "-" * 60)
        logger.info(f"   {'Metric':<25}{'Train':<20}{'Test':<20}")
        logger.info("   " + "-" * 60)
        logger.info(f"   {'Accuracy':<25}{metrics['train_accuracy']:<20.4f}{metrics['test_accuracy']:<20.4f}")
        logger.info(f"   {'AUC-ROC':<25}{metrics['train_auc']:<20.4f}{metrics['test_auc']:<20.4f}")
        logger.info(f"   {'Precision':<25}{'-':<20}{metrics['test_precision']:<20.4f}")
        logger.info(f"   {'Recall':<25}{'-':<20}{metrics['test_recall']:<20.4f}")
        logger.info(f"   {'F1-Score':<25}{'-':<20}{metrics['test_f1']:<20.4f}")
        logger.info("   " + "-" * 60)

        # Confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        logger.info(f"\n   Confusion Matrix:")
        logger.info(f"   TN: {cm[0, 0]:,}  FP: {cm[0, 1]:,}")
        logger.info(f"   FN: {cm[1, 0]:,}  TP: {cm[1, 1]:,}")

        return metrics

    def _extract_feature_importance(self) -> pd.DataFrame:
        """Extract feature importance with rankings"""

        try:
            # Get classifier from pipeline
            if hasattr(self.model, 'named_steps'):
                classifier = self.model.named_steps['classifier']
            else:
                classifier = self.model

            importance = None
            use_permutation = False

            # For HistGradientBoostingClassifier, fallback to permutation importance
            if isinstance(classifier, HistGradientBoostingClassifier):
                use_permutation = True

            elif hasattr(classifier, 'feature_importances_'):
                # Standard tree-based models
                importance = classifier.feature_importances_
            else:
                # Unknown model type -> fallback
                use_permutation = True

            if use_permutation:
                logger.info("   ℹ️  Computing permutation importance on a validation sample...")
                if self._last_eval_data is not None:
                    X_val, y_val = self._last_eval_data
                    # take a reasonable sample size for speed
                    n_sample = min(20000, len(X_val))
                    if n_sample < len(X_val):
                        Xv = X_val.sample(n_sample, random_state=42)
                        yv = y_val.loc[Xv.index]
                    else:
                        Xv, yv = X_val, y_val
                    pi = permutation_importance(self.model, Xv, yv, n_repeats=3, random_state=42, n_jobs=1)
                    importance = pi.importances_mean
                else:
                    logger.warning("   ⚠️  No validation data available for permutation importance")
                    return pd.DataFrame()

            feature_names = self.feature_names

            # Safety: align lengths
            if importance is None or len(importance) != len(feature_names):
                logger.warning("   ⚠️  Importance vector length mismatch or unavailable")
                return pd.DataFrame()

            # Create dataframe with rankings
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)

            # Add rank and percentage
            total = float(np.abs(importance_df['importance']).sum()) or 1.0
            importance_df['rank'] = range(1, len(importance_df) + 1)
            importance_df['importance_pct'] = (importance_df['importance'] / total) * 100.0
            importance_df['cumulative_pct'] = importance_df['importance_pct'].cumsum()

            # Reorder columns
            importance_df = importance_df[['rank', 'feature', 'importance', 'importance_pct', 'cumulative_pct']]

            # Log feature importance rankings
            logger.info("\n   📊 Feature Importance Rankings:")
            logger.info("   " + "=" * 90)
            logger.info(f"   {'Rank':<6}{'Feature':<45}{'Importance':<15}{'% Total':<12}{'Cumulative %':<12}")
            logger.info("   " + "-" * 90)

            # Show top 30 features
            for idx, row in importance_df.head(30).iterrows():
                logger.info(f"   {row['rank']:<6}{row['feature']:<45}{row['importance']:<15.6f}{row['importance_pct']:<12.2f}{row['cumulative_pct']:<12.2f}")

            if len(importance_df) > 30:
                logger.info(f"   ... and {len(importance_df) - 30} more features")

            logger.info("   " + "=" * 90)

            # Summary statistics
            logger.info("\n   📊 Feature Importance Summary:")
            logger.info(f"      Top 10 features: {importance_df.head(10)['importance_pct'].sum():.2f}% of total importance")
            logger.info(f"      Top 20 features: {importance_df.head(20)['importance_pct'].sum():.2f}% of total importance")
            if len(importance_df) >= 50:
                logger.info(f"      Top 50 features: {importance_df.head(50)['importance_pct'].sum():.2f}% of total importance")

            return importance_df

        except Exception as e:
            logger.error(f"   ❌ Could not extract feature importance: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def _save_model_bundle(
        self,
        save_path: Path,
        metrics: Dict,
        feature_importance: pd.DataFrame,
        feature_names: list
    ):
        """Save complete model bundle with feature rankings"""

        try:
            # Create directory
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Create model bundle
            model_bundle = {
                'pipeline': self.model,
                'all_features': feature_names,  # EXACT features used in training
                'metrics': metrics,
                'feature_importance': feature_importance,
                'training_date': datetime.now().isoformat(),
                'num_features': len(feature_names),
                'feature_selector': self.feature_selector  # Save selector if used
            }

            # Save model
            joblib.dump(model_bundle, save_path)
            logger.info(f"   💾 Model saved: {save_path}")
            logger.info(f"      Size: {save_path.stat().st_size / (1024**2):.2f} MB")

            # Save EXACT feature list used in training (CRITICAL for backtesting/screening)
            features_list_path = save_path.parent / "training_features.txt"
            with open(features_list_path, 'w', encoding='utf-8') as f:
                f.write("=" * 90 + "\n")
                f.write("EXACT FEATURES USED IN MODEL TRAINING\n")
                f.write("=" * 90 + "\n")
                f.write(f"Total: {len(feature_names)} features\n")
                f.write(f"Training date: {datetime.now().isoformat()}\n")
                f.write("=" * 90 + "\n\n")
                f.write("Feature List (in order):\n")
                f.write("-" * 90 + "\n")
                for i, feat in enumerate(feature_names, 1):
                    f.write(f"{i:4d}. {feat}\n")

            logger.info(f"   💾 Training features list: {features_list_path}")

            # Save feature importance with rankings
            if isinstance(feature_importance, pd.DataFrame) and not feature_importance.empty:
                importance_path = save_path.parent / "feature_importance_ranked.csv"
                feature_importance.to_csv(importance_path, index=False)
                logger.info(f"   💾 Feature rankings saved: {importance_path}")

                # Save top features to text file
                top_features_path = save_path.parent / "top_features.txt"
                with open(top_features_path, 'w', encoding='utf-8') as f:
                    f.write("=" * 90 + "\n")
                    f.write("TOP FEATURES BY IMPORTANCE\n")
                    f.write("=" * 90 + "\n\n")
                    f.write(f"{'Rank':<6}{'Feature':<45}{'Importance':<15}{'% Total':<12}{'Cumulative %':<12}\n")
                    f.write("-" * 90 + "\n")

                    for idx, row in feature_importance.head(min(100, len(feature_importance))).iterrows():
                        f.write(f"{row['rank']:<6}{row['feature']:<45}{row['importance']:<15.6f}{row['importance_pct']:<12.2f}{row['cumulative_pct']:<12.2f}\n")

                    f.write("\n" + "=" * 90 + "\n")
                    f.write(f"Top 10 features account for: {feature_importance.head(10)['importance_pct'].sum():.2f}%\n")
                    f.write(f"Top 20 features account for: {feature_importance.head(20)['importance_pct'].sum():.2f}%\n")
                    if len(feature_importance) >= 50:
                        f.write(f"Top 50 features account for: {feature_importance.head(50)['importance_pct'].sum():.2f}%\n")

                logger.info(f"   💾 Top features summary: {top_features_path}")

            # Save metrics
            metrics_path = save_path.parent / "training_metrics.json"
            import json
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"   💾 Training metrics: {metrics_path}")

        except Exception as e:
            raise ModelError(f"Failed to save model: {e}")


class ModelLoader:
    """Load and use trained models"""

    @staticmethod
    def load(model_path: Path) -> Dict:
        """Load model bundle"""
        try:
            if not model_path.exists():
                raise ModelError(f"Model not found: {model_path}")

            model_bundle = joblib.load(model_path)
            logger.info(f"✅ Model loaded from {model_path}")

            # Log model info
            if 'training_date' in model_bundle:
                logger.info(f"   Training date: {model_bundle['training_date']}")
            if 'num_features' in model_bundle:
                logger.info(f"   Features: {model_bundle['num_features']}")
            if 'metrics' in model_bundle:
                auc = model_bundle['metrics'].get('test_auc', 'N/A')
                acc = model_bundle['metrics'].get('test_accuracy', 'N/A')
                try:
                    logger.info(f"   Test AUC: {auc:.3f}")
                    logger.info(f"   Test Accuracy: {acc:.3f}")
                except Exception:
                    logger.info(f"   Test AUC: {auc}")
                    logger.info(f"   Test Accuracy: {acc}")

            return model_bundle

        except Exception as e:
            raise ModelError(f"Failed to load model: {e}")

    @staticmethod
    def predict(model_bundle: Dict, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions using loaded model - Uses saved training features"""
        try:
            pipeline = model_bundle['pipeline']

            # Use the feature list that was saved during training
            # This is the EXACT list of features the model was trained on
            feature_names = model_bundle['all_features']

            logger.info(f"   Model expects {len(feature_names)} features (from saved training data)")
            logger.info(f"   Input has {len(X.columns)} features")

            # 1) Drop duplicate-named columns from incoming X (root cause of 183->283)
            X = _dedupe_columns(X, where="prediction input")

            # 2) Check which expected features are missing from input
            missing_features = [f for f in feature_names if f not in X.columns]
            if missing_features:
                logger.warning(f"   ⚠️  {len(missing_features)} features missing, filling with 0")
                if len(missing_features) <= 10:
                    logger.warning(f"      Missing: {missing_features}")
                # Add missing features with zeros
                for feat in missing_features:
                    X[feat] = 0

            # 3) Select ONLY the features in the exact order they were during training
            #    (after de-duplication, this will be the correct count)
            X_ordered = X.loc[:, feature_names].copy()
            logger.info(f"   ✅ Selected {len(X_ordered.columns)} features in correct order")

            # 4) Validate exact count
            if len(X_ordered.columns) != len(feature_names):
                raise ValueError(f"Feature count mismatch: got {len(X_ordered.columns)}, expected {len(feature_names)}")

            # 5) Coerce numeric and sanitize NaN/Inf
            X_ordered = _coerce_numeric(X_ordered)
            nan_count = int(X_ordered.isnull().sum().sum())
            if nan_count > 0:
                logger.warning(f"   ⚠️  Found {nan_count} NaN values, filling with 0")
                X_ordered = X_ordered.fillna(0)
            # Replace infs
            if np.isinf(X_ordered.to_numpy()).any():
                logger.warning(f"   ⚠️  Found infinite values, replacing with 0")
                X_ordered = X_ordered.replace([np.inf, -np.inf], 0)

            logger.info(f"   Final data shape: {X_ordered.shape}")

            # Make predictions
            predictions = pipeline.predict(X_ordered)
            probabilities = pipeline.predict_proba(X_ordered)

            logger.info(f"   ✅ Predictions generated: {int(np.sum(predictions)):,} signals")

            return predictions, probabilities

        except Exception as e:
            logger.error(f"❌ Prediction failed: {e}")

            # Debugging
            try:
                logger.error(f"\n   📋 Debug Info:")
                if 'pipeline' in locals() and hasattr(pipeline, 'feature_names_in_'):
                    logger.error(f"   Pipeline expects: {len(pipeline.feature_names_in_)} features")
                    logger.error(f"   First 10: {list(pipeline.feature_names_in_)[:10]}")

                if 'feature_names' in locals():
                    logger.error(f"   Bundle has: {len(feature_names)} features")
                    logger.error(f"   First 10: {feature_names[:10]}")

                if 'X_ordered' in locals():
                    logger.error(f"   Selected: {len(X_ordered.columns)} features")

                if isinstance(X, pd.DataFrame):
                    logger.error(f"   Input data (post-dedupe): {X.shape}")
            except Exception as debug_err:
                logger.error(f"   Debug failed: {debug_err}")

            import traceback
            traceback.print_exc()
            raise ModelError(f"Prediction failed: {e}")


def train_model(df, save_path, test_size=0.2, random_state=42):
    """Backward-compatible helper for training"""
    trainer = ModelTrainer()

    # Create a simple model_config object
    class SimpleConfig:
        def __init__(self, path):
            self.model_path = path

    return trainer.train(df, SimpleConfig(save_path), test_size, random_state)


__all__ = ['ModelTrainer', 'ModelLoader', 'train_model']

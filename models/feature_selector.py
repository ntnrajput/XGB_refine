# models/feature_selector.py - NEW FILE

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from pathlib import Path
import joblib

from utils.logger import get_logger

logger = get_logger(__name__)


class FastFeatureSelector:
    """Fast feature selection to reduce training time"""
    
    def __init__(self, n_features: int = 100, method: str = 'mutual_info'):
        """
        Args:
            n_features: Number of features to select
            method: 'mutual_info', 'variance', 'correlation', or 'random_forest'
        """
        self.n_features = n_features
        self.method = method
        self.selected_features = None
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> list:
        """
        Select top N features using fast method.
        
        Returns:
            List of selected feature names
        """
        logger.info(f"🔍 Selecting top {self.n_features} features using {self.method}...")
        logger.info(f"   Input: {X.shape[1]} features")
        
        if self.method == 'variance':
            selected = self._select_by_variance(X)
        elif self.method == 'correlation':
            selected = self._select_by_correlation(X, y)
        elif self.method == 'mutual_info':
            selected = self._select_by_mutual_info(X, y)
        elif self.method == 'random_forest':
            selected = self._select_by_random_forest(X, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        self.selected_features = selected
        logger.info(f"✅ Selected {len(selected)} features")
        
        return selected
    
    def _select_by_variance(self, X: pd.DataFrame) -> list:
        """Select features with highest variance (fastest)"""
        variances = X.var()
        top_features = variances.nlargest(self.n_features).index.tolist()
        return top_features
    
    def _select_by_correlation(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Select features most correlated with target (fast)"""
        correlations = X.corrwith(y).abs()
        top_features = correlations.nlargest(self.n_features).index.tolist()
        return top_features
    
    def _select_by_mutual_info(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Select features by mutual information (medium speed)"""
        # Sample for speed if dataset is large
        if len(X) > 50000:
            sample_idx = np.random.choice(len(X), 50000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
        
        mi_scores = mutual_info_classif(X_sample, y_sample, random_state=42)
        mi_df = pd.DataFrame({'feature': X.columns, 'mi_score': mi_scores})
        top_features = mi_df.nlargest(self.n_features, 'mi_score')['feature'].tolist()
        return top_features
    
    def _select_by_random_forest(self, X: pd.DataFrame, y: pd.Series) -> list:
        """Select features by Random Forest importance (slower but accurate)"""
        # Sample for speed
        if len(X) > 100000:
            sample_idx = np.random.choice(len(X), 100000, replace=False)
            X_sample = X.iloc[sample_idx]
            y_sample = y.iloc[sample_idx]
        else:
            X_sample = X
            y_sample = y
        
        logger.info("   Training quick RF for feature selection...")
        rf = RandomForestClassifier(
            n_estimators=50,  # Fast
            max_depth=10,
            min_samples_split=100,
            n_jobs=-1,
            random_state=42
        )
        rf.fit(X_sample, y_sample)
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        top_features = importance_df.head(self.n_features)['feature'].tolist()
        return top_features
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select only the chosen features"""
        if self.selected_features is None:
            raise ValueError("Must call select_features first!")
        return X[self.selected_features]
    
    def save(self, path: Path):
        """Save selected features"""
        joblib.dump(self.selected_features, path)
        logger.info(f"💾 Saved selected features to {path}")
    
    @staticmethod
    def load(path: Path):
        """Load selected features"""
        selector = FastFeatureSelector()
        selector.selected_features = joblib.load(path)
        logger.info(f"✅ Loaded {len(selector.selected_features)} selected features")
        return selector


__all__ = ['FastFeatureSelector']
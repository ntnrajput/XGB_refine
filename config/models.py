# config/models.py - Model configuration

from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from config.settings import MODEL_DIR


@dataclass
class ModelConfig:
    """Model training and deployment configuration"""
    
    # Model versioning
    version: str = "v1"
    
    # Model parameters
    n_estimators: int = 200
    learning_rate: float = 0.05
    max_depth: int = 5
    min_samples_split: int = 100
    min_samples_leaf: int = 50
    subsample: float = 0.8
    random_state: int = 42
    
    # Training parameters
    test_size: float = 0.2
    validation_split: float = 0.2
    
    # Feature selection
    feature_importance_threshold: float = 0.001
    
    # Class balancing
    use_smote: bool = True
    smote_sampling_strategy: float = 0.5
    
    # Early stopping
    early_stopping_rounds: Optional[int] = 50
    
    # Paths (computed based on version)
    @property
    def model_dir(self) -> Path:
        """Directory for this model version"""
        return MODEL_DIR / self.version
    
    @property
    def model_path(self) -> Path:
        """Path to saved model file"""
        return self.model_dir / "enhanced_model_pipeline.pkl"
    
    @property
    def metrics_path(self) -> Path:
        """Path to metrics file"""
        return self.model_dir / "metrics.json"
    
    @property
    def feature_importance_path(self) -> Path:
        """Path to feature importance file"""
        return self.model_dir / "feature_importance.csv"
    
    def set_version(self, version: str):
        """Set model version"""
        self.version = version
        # Create directory if it doesn't exist
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def get_sklearn_params(self) -> dict:
        """Get parameters for scikit-learn GradientBoostingClassifier"""
        return {
            'n_estimators': self.n_estimators,
            'learning_rate': self.learning_rate,
            'max_depth': self.max_depth,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'subsample': self.subsample,
            'random_state': self.random_state,
            'verbose': 0
        }


# Global model configuration instance
MODEL_CONFIG = ModelConfig()


# Export
__all__ = ['ModelConfig', 'MODEL_CONFIG']
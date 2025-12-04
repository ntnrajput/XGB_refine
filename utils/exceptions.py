# utils/exceptions.py - NEW FILE FOR CUSTOM EXCEPTIONS

class TradingSystemError(Exception):
    """Base exception for trading system"""
    pass


class DataError(TradingSystemError):
    """Data-related errors"""
    pass


class FeatureEngineeringError(TradingSystemError):
    """Feature engineering errors"""
    pass


class FeatureValidationError(TradingSystemError):
    """Feature validation errors"""
    pass


class ModelError(TradingSystemError):
    """Model-related errors"""
    pass


class BacktestError(TradingSystemError):
    """Backtesting errors"""
    pass
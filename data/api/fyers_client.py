# data/api/fyers_client.py - Unified API client

from typing import Dict, Optional
import pandas as pd
from datetime import datetime
from fyers_apiv3 import fyersModel

from config.settings import FYERS_CLIENT_ID, TOKEN_PATH
from utils.logger import get_logger

logger = get_logger(__name__)


class FyersClient:
    """
    Unified FYERS API client.
    Single point of interaction with FYERS API.
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._client = None
        self._initialized = True
    
    @property
    def client(self) -> fyersModel.FyersModel:
        """Get or create FYERS client"""
        if self._client is None:
            from data.api.auth import get_saved_access_token
            token = get_saved_access_token()
            
            if not token:
                raise ValueError("No access token. Please authenticate first.")
            
            self._client = fyersModel.FyersModel(
                token=token, 
                is_async=False, 
                client_id=None
            )
            logger.info("[fyers_client.py] ✅ FYERS client initialized")
        
        return self._client
    
    def get_history(
        self,
        symbol: str,
        from_date: datetime,
        to_date: datetime,
        resolution: str = "1D"
    ) -> Dict:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Trading symbol
            from_date: Start date
            to_date: End date
            resolution: Time resolution
            
        Returns:
            API response dict
        """
        params = {
            "symbol": symbol.strip(),
            "resolution": resolution,
            "date_format": "1",
            "range_from": from_date.strftime("%Y-%m-%d"),
            "range_to": to_date.strftime("%Y-%m-%d"),
            "cont_flag": "1"
        }
        
        return self.client.history(params)
    
    def is_authenticated(self) -> bool:
        """Check if client is authenticated"""
        try:
            return self._client is not None or TOKEN_PATH.exists()
        except:
            return False
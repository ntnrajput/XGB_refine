# data/api/auth.py - Complete authentication module

import webbrowser
from pathlib import Path
from typing import Optional

from fyers_apiv3 import fyersModel
from config.settings import (
    FYERS_CLIENT_ID,
    FYERS_SECRET_ID,
    FYERS_REDIRECT_URI,
    FYERS_APP_ID_HASH,
    TOKEN_PATH
)
from utils.logger import get_logger

logger = get_logger(__name__)


def authenticate():
    """
    Step 1: Launch browser for authentication.
    User logs in and copies the auth code from redirect URL.
    """
    logger.info("[auth.py] 🔐 Starting FYERS authentication...")
    
    if not FYERS_CLIENT_ID or not FYERS_SECRET_ID:
        logger.error("[auth.py] ❌ FYERS credentials not found in .env file")
        logger.error("[auth.py] Please set FYERS_CLIENT_ID and FYERS_SECRET_ID in .env")
        return
    
    try:
        # Create session for authentication
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type="code",
            grant_type="authorization_code"
        )
        
        # Generate auth URL
        auth_url = session.generate_authcode()
        
        logger.info("[auth.py] 🌐 Opening browser for authentication...")
        logger.info(f"[auth.py] Auth URL: {auth_url}")
        
        # Open browser
        webbrowser.open(auth_url)
        
        logger.info("[auth.py] =" * 70)
        logger.info("[auth.py] 📋 AUTHENTICATION STEPS:")
        logger.info("[auth.py] =" * 70)
        logger.info("[auth.py] 1. Log in to FYERS in the browser")
        logger.info("[auth.py] 2. After login, you'[auth.py] ll be redirected to Google")
        logger.info("[auth.py] 3. Copy the FULL redirect URL from browser address bar")
        logger.info("[auth.py] 4. The URL contains: ?auth_code=XXXXX&state=...")
        logger.info("[auth.py] 5. Copy only the auth_code value (after auth_code=)")
        logger.info("[auth.py] ")
        logger.info("[auth.py] Then run: python main.py --token <AUTH_CODE>")
        logger.info("[auth.py] =" * 70)
        
    except Exception as e:
        logger.error(f"[auth.py] ❌ Authentication failed: {e}")
        raise


def generate_token(auth_code: str) -> bool:
    """
    Step 2: Generate access token from auth code.
    
    Args:
        auth_code: Authorization code from redirect URL
        
    Returns:
        True if successful
    """
    logger.info("[auth.py] 🔑 Generating access token...")
    
    if not auth_code:
        logger.error("[auth.py] ❌ Auth code is empty")
        return False
    
    # Remove any whitespace
    auth_code = auth_code.strip()
    
    try:
        # Create session
        session = fyersModel.SessionModel(
            client_id=FYERS_CLIENT_ID,
            secret_key=FYERS_SECRET_ID,
            redirect_uri=FYERS_REDIRECT_URI,
            response_type="code",
            grant_type="authorization_code"
        )
        
        # Set auth code
        session.set_token(auth_code)
        
        # Generate access token
        response = session.generate_token()
        
        if response.get("code") == 200:
            access_token = response["access_token"]
            
            # Save token
            TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(TOKEN_PATH, "w") as f:
                f.write(access_token)
            
            logger.info(f"[auth.py] ✅ Access token saved to {TOKEN_PATH}")
            logger.info("[auth.py] ✅ Authentication complete!")
            logger.info("[auth.py] \n📋 Next step: python main.py --fetch-history")
            
            return True
        else:
            logger.error(f"[auth.py] ❌ Token generation failed: {response}")
            return False
            
    except Exception as e:
        logger.error(f"[auth.py] ❌ Token generation failed: {e}")
        logger.error("[auth.py] Common issues:")
        logger.error("[auth.py]   • Auth code expired (valid for 5 minutes)")
        logger.error("[auth.py]   • Auth code already used")
        logger.error("[auth.py]   • Wrong auth code copied")
        logger.error("[auth.py] \nSolution: Run --auth again to get a new auth code")
        return False


def get_saved_access_token() -> Optional[str]:
    """
    Get saved access token from file.
    
    Returns:
        Access token or None if not found
    """
    if not TOKEN_PATH.exists():
        logger.warning(f"[auth.py] ⚠️  No saved token found at {TOKEN_PATH}")
        logger.warning("[auth.py] Please authenticate first: python main.py --auth")
        return None
    
    try:
        with open(TOKEN_PATH, "r") as f:
            token = f.read().strip()
        
        if not token:
            logger.warning("[auth.py] ⚠️  Token file is empty")
            return None
        
        logger.debug("[auth.py] ✅ Token loaded from file")
        return token
        
    except Exception as e:
        logger.error(f"[auth.py] ❌ Failed to read token: {e}")
        return None


def is_token_valid() -> bool:
    """
    Check if saved token exists and is not empty.
    Note: This doesn't validate with API, just checks file.
    
    Returns:
        True if token file exists and has content
    """
    token = get_saved_access_token()
    return token is not None and len(token) > 0


def clear_token():
    """Clear saved access token"""
    if TOKEN_PATH.exists():
        TOKEN_PATH.unlink()
        logger.info("[auth.py] ✅ Token cleared")
    else:
        logger.info("[auth.py] ℹ️  No token to clear")
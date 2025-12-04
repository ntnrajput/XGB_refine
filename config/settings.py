# config/settings.py - Updated with better .env handling

import os
from pathlib import Path
from dotenv import load_dotenv

# Determine project root
BASE_DIR = Path(__file__).parent.parent

# Try to load .env from multiple locations
env_locations = [
    BASE_DIR / '.env',              # Root directory
    BASE_DIR / 'config' / '.env',   # Config directory
]

env_loaded = False
for env_path in env_locations:
    if env_path.exists():
        load_dotenv(env_path)
        env_loaded = True
        print(f"✅ Loaded .env from {env_path}")
        break

if not env_loaded:
    print(f"Warning: .env file not found at {BASE_DIR / '.env'}")
    print("Create a .env file with your FYERS credentials")

# Paths
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = OUTPUT_DIR / "models"
DATA_DIR = OUTPUT_DIR / "data"
LOG_DIR = OUTPUT_DIR / "logs"

# Create directories
for dir_path in [OUTPUT_DIR, MODEL_DIR, DATA_DIR, LOG_DIR]:
    dir_path.mkdir(exist_ok=True, parents=True)

# API Credentials (with defaults)
FYERS_CLIENT_ID = os.getenv("FYERS_CLIENT_ID", "")
FYERS_SECRET_ID = os.getenv("FYERS_SECRET_ID", "")
FYERS_REDIRECT_URI = os.getenv("FYERS_REDIRECT_URI", "https://www.google.com")
FYERS_APP_ID_HASH = os.getenv("FYERS_APP_ID_HASH", "")

# File paths
TOKEN_PATH = OUTPUT_DIR / "fyers_access_token.txt"
SYMBOLS_FILE = BASE_DIR / "symbols.csv"
HISTORICAL_DATA_FILE = DATA_DIR / "all_symbols_history.parquet"
LATEST_DATA_FILE = DATA_DIR / "latest_full_data.parquet"
LOG_FILE = LOG_DIR / "system.log"

# Export all
__all__ = [
    'BASE_DIR', 'OUTPUT_DIR', 'MODEL_DIR', 'DATA_DIR', 'LOG_DIR',
    'FYERS_CLIENT_ID', 'FYERS_SECRET_ID', 'FYERS_REDIRECT_URI', 'FYERS_APP_ID_HASH',
    'TOKEN_PATH', 'SYMBOLS_FILE', 'HISTORICAL_DATA_FILE', 'LATEST_DATA_FILE', 'LOG_FILE'
]
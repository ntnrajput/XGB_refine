# data/fetchers/historical.py - SMART FETCHER (skips invalid symbols)

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass
from tqdm import tqdm

from data.api.fyers_client import FyersClient
from config.settings import HISTORICAL_DATA_FILE, OUTPUT_DIR
from utils.logger import get_logger
from utils.exceptions import DataError

logger = get_logger(__name__)


@dataclass
class FetchConfig:
    """Optimized fetch configuration"""
    api_sleep_seconds: float = 0.25
    max_workers: int = 10
    max_retries: int = 1  # Only 1 retry for speed
    retry_delay: float = 0.5
    chunk_size_days: int = 365


class RateLimiter:
    """Thread-safe rate limiter"""
    
    def __init__(self, calls_per_second: float = 5):
        self.min_interval = 1.0 / calls_per_second
        self.last_call = 0
        self.lock = Lock()
    
    def wait(self):
        with self.lock:
            elapsed = time.time() - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.time()


class HistoricalDataFetcher:
    """
    Smart historical data fetcher.
    - Skips invalid symbols immediately (no retries)
    - 10 parallel workers
    - Saves blacklist of invalid symbols
    """
    
    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self.client = FyersClient()
        self.rate_limiter = RateLimiter(calls_per_second=5)
        self.stats = {
            'total_symbols': 0,
            'updated_symbols': 0,
            'skipped_symbols': 0,
            'failed_symbols': 0,
            'invalid_symbols': 0,
            'api_calls': 0
        }
        self.invalid_symbols: Set[str] = self._load_invalid_symbols()
    
    def _load_invalid_symbols(self) -> Set[str]:
        """Load previously identified invalid symbols"""
        blacklist_file = OUTPUT_DIR / "invalid_symbols.txt"
        if blacklist_file.exists():
            with open(blacklist_file, 'r') as f:
                invalid = set(line.strip() for line in f if line.strip())
            logger.info(f"[historical.py] 📋 Loaded {len(invalid)} invalid symbols from blacklist")
            return invalid
        return set()
    
    def _save_invalid_symbols(self):
        """Save invalid symbols to blacklist"""
        blacklist_file = OUTPUT_DIR / "invalid_symbols.txt"
        with open(blacklist_file, 'w') as f:
            for symbol in sorted(self.invalid_symbols):
                f.write(f"{symbol}\n")
        logger.info(f"[historical.py] 💾 Saved {len(self.invalid_symbols)} invalid symbols to blacklist")
    
    def fetch_and_store_all(self, symbols: List[str], years: int = 15):
        """Main fetch method"""
        logger.info(f"[historical.py] 🚀 SMART FETCH: {len(symbols)} symbols, {years} years")
        
        start_time = time.time()
        today = pd.Timestamp.now().normalize()
        start_date = today - timedelta(days=365 * years)
        
        # Filter out known invalid symbols
        valid_symbols = [s for s in symbols if s not in self.invalid_symbols]
        if len(valid_symbols) < len(symbols):
            skipped = len(symbols) - len(valid_symbols)
            logger.info(f"[historical.py] ⏭️  Skipping {skipped} known invalid symbols")
        
        self.stats['total_symbols'] = len(valid_symbols)
        
        # Load existing
        existing_data = self._load_existing_data()
        
        # Categorize
        categorized = self._categorize_symbols(valid_symbols, existing_data, today)
        
        logger.info(f"[historical.py] 📊 Plan:")
        logger.info(f"[historical.py]   New: {len(categorized['new'])}")
        logger.info(f"[historical.py]   Update: {len(categorized['update'])}")
        logger.info(f"[historical.py]   Skip: {len(categorized['skip'])}")
        
        # Process
        all_data = []
        
        if categorized['skip']:
            skip_data = existing_data[existing_data['symbol'].isin(categorized['skip'])]
            all_data.append(skip_data)
            self.stats['skipped_symbols'] = len(categorized['skip'])
        
        if categorized['new']:
            logger.info(f"[historical.py] \n📥 Fetching {len(categorized['new'])} new symbols...")
            new_data = self._fetch_batch(categorized['new'], start_date, today)
            if new_data:
                all_data.extend(new_data)
        
        if categorized['update']:
            logger.info(f"[historical.py] \n🔄 Updating {len(categorized['update'])} symbols...")
            update_data = self._update_batch(categorized['update'], existing_data, today)
            if update_data:
                all_data.extend(update_data)
        
        # Save
        if all_data:
            self._save_data(all_data)
        
        # Save blacklist
        if self.invalid_symbols:
            self._save_invalid_symbols()
        
        # Stats
        elapsed = time.time() - start_time
        logger.info(f"[historical.py] \n⏱️  Completed in {elapsed/60:.1f} minutes")
        logger.info(f"[historical.py] 📊 Valid: {self.stats['updated_symbols']}, Invalid: {self.stats['invalid_symbols']}, API calls: {self.stats['api_calls']}")
    
    def _categorize_symbols(self, symbols, existing_data, today):
        """Categorize symbols"""
        categorized = {'new': [], 'update': [], 'skip': []}
        
        if existing_data.empty:
            categorized['new'] = symbols
            return categorized
        
        existing_symbols = set(existing_data['symbol'].unique())
        
        for symbol in symbols:
            if symbol not in existing_symbols:
                categorized['new'].append(symbol)
            else:
                latest = existing_data[existing_data['symbol'] == symbol]['date'].max()
        
                if (today - latest).days >= 1:
                    categorized['update'].append(symbol)

                else:
                    categorized['skip'].append(symbol)
        
        return categorized
    
    def _fetch_batch(self, symbols, start_date, end_date):
        """Fetch batch in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol, sym, start_date, end_date): sym
                for sym in symbols
            }
            
            with tqdm(total=len(symbols), desc="Fetching", unit="sym") as pbar:
                for future in as_completed(futures):
                    symbol = futures[future]
                    try:
                        data = future.result()
                        if not data.empty:
                            results.append(data)
                            self.stats['updated_symbols'] += 1
                        pbar.set_postfix({
                            'valid': len(results), 
                            'invalid': self.stats['invalid_symbols']
                        })
                    except Exception:
                        self.stats['failed_symbols'] += 1
                    pbar.update(1)
        
        return results
    
    def _update_batch(self, symbols, existing_data, today):
        """Update batch"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {}
            
            for symbol in symbols:
                symbol_data = existing_data[existing_data['symbol'] == symbol]
                latest = symbol_data['date'].max()
                future = executor.submit(self._fetch_and_merge, symbol, symbol_data, latest, today)
                futures[future] = symbol
            
            with tqdm(total=len(symbols), desc="Updating", unit="sym") as pbar:
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if not data.empty:
                            results.append(data)
                            self.stats['updated_symbols'] += 1
                    except:
                        pass
                    pbar.update(1)
        
        return results
    
    def _fetch_symbol(self, symbol, start_date, end_date):
        """Fetch one symbol with chunking"""
        all_chunks = []
        current = start_date
        
        while current < end_date:
            chunk_end = min(current + timedelta(days=self.config.chunk_size_days), end_date)
            
            chunk = self._fetch_chunk(symbol, current, chunk_end)
            if chunk is None:  # Invalid symbol
                return pd.DataFrame()
            
            if not chunk.empty:
                all_chunks.append(chunk)
            
            current = chunk_end + timedelta(days=1)
            time.sleep(0.05)  # Tiny delay
        
        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True)
            df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df.sort_values('date', inplace=True)
            return df
        
        return pd.DataFrame()
    
    def _fetch_chunk(self, symbol, start_date, end_date):
        """
        Fetch single chunk.
        Returns:
        - DataFrame if successful
        - Empty DataFrame if no data
        - None if invalid symbol (triggers blacklist)
        """
        
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limiter.wait()
                
                data = self.client.get_history(symbol, start_date, end_date)
                self.stats['api_calls'] += 1
                
                if data.get("s") == "ok":
                    if "candles" in data and data["candles"]:
                        df = pd.DataFrame(
                            data["candles"],
                            columns=["timestamp", "open", "high", "low", "close", "volume"]
                        )
                        df["volume"] /= 1e5
                        df["symbol"] = symbol
                        df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                        return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
                    return pd.DataFrame()  # No data but valid symbol
                
                elif data.get("s") == "error":
                    msg = data.get('message', '').lower()
                    
                    # Invalid symbol - blacklist immediately, no retry
                    if 'invalid symbol' in msg or 'symbol' in msg:
                        self.invalid_symbols.add(symbol)
                        self.stats['invalid_symbols'] += 1
                        return None  # Signal invalid
                    
                    # Other errors - may retry
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                        continue
                
                return pd.DataFrame()
            
            except Exception:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        return pd.DataFrame()
    
    def _fetch_and_merge(self, symbol, existing, from_date, to_date):
        """Fetch and merge"""
        new = self._fetch_symbol(symbol, from_date, to_date)
        
        if new.empty:
            return existing
        
        merged = pd.concat([existing, new], ignore_index=True)
        merged.drop_duplicates(subset=['date'], keep='last', inplace=True)
        merged.sort_values('date', inplace=True)
        return merged
    
    def _load_existing_data(self):
        """Load existing data"""
        if HISTORICAL_DATA_FILE.exists():
            try:
                df = pd.read_parquet(HISTORICAL_DATA_FILE)
                df['date'] = pd.to_datetime(df['date'])
                logger.info(f"[historical.py] 📂 Loaded: {len(df)} rows, {df['symbol'].nunique()} symbols")
                return df
            except:
                pass
        
        logger.info("[historical.py] 📂 No existing data")
        return pd.DataFrame()
    
    def _save_data(self, data_list):
        """Save combined data"""
        df = pd.concat(data_list, ignore_index=True)
        df.drop_duplicates(subset=['symbol', 'date'], keep='last', inplace=True)
        df.sort_values(by=['symbol', 'date'], inplace=True)
        
        HISTORICAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(HISTORICAL_DATA_FILE, index=False, compression='snappy')
        
        size_mb = HISTORICAL_DATA_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"[historical.py] 💾 Saved: {len(df)} rows, {df['symbol'].nunique()} symbols ({size_mb:.1f} MB)")


__all__ = ['HistoricalDataFetcher', 'FetchConfig']

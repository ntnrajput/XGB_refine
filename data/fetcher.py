# optimize_fetcher.py - Create ultra-fast fetcher

from pathlib import Path

FAST_FETCHER = '''# data/fetchers/historical.py - OPTIMIZED FOR SPEED

import time
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dataclasses import dataclass
from tqdm import tqdm

from data.api.fyers_client import FyersClient
from config.settings import HISTORICAL_DATA_FILE
from utils.logger import get_logger
from utils.exceptions import DataError

logger = get_logger(__name__)


@dataclass
class FetchConfig:
    """Optimized fetch configuration"""
    api_sleep_seconds: float = 0.25  # Faster: 0.25 vs 0.5
    max_workers: int = 8             # More parallel: 8 vs 3
    max_retries: int = 2             # Less retries: 2 vs 3
    retry_delay: float = 1.0         # Faster retry: 1s vs 2s
    chunk_size_days: int = 365       # Full year chunks


class RateLimiter:
    """Thread-safe rate limiter"""
    
    def __init__(self, calls_per_second: float = 4):  # Faster: 4 vs 2
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
    Ultra-fast historical data fetcher.
    
    Speed improvements:
    - 8 parallel workers (vs 3)
    - 4 calls/second rate limit (vs 2)
    - Batch processing by year
    - Smart caching and incremental updates
    """
    
    def __init__(self, config: Optional[FetchConfig] = None):
        self.config = config or FetchConfig()
        self.client = FyersClient()
        self.rate_limiter = RateLimiter(calls_per_second=4)
        self.stats = {
            'total_symbols': 0,
            'updated_symbols': 0,
            'skipped_symbols': 0,
            'failed_symbols': 0,
            'api_calls': 0,
            'total_chunks': 0
        }
    
    def fetch_and_store_all(self, symbols: List[str], years: int = 15):
        """
        Main fetch method with smart caching.
        
        Performance: ~500 symbols in 20-30 minutes
        """
        logger.info(f"[fetcher.py] 🚀 FAST FETCH: {len(symbols)} symbols, {years} years")
        
        start_time = time.time()
        today = pd.Timestamp.now().normalize()
        start_date = today - timedelta(days=365 * years)
        
        self.stats['total_symbols'] = len(symbols)
        
        # Load existing
        existing_data = self._load_existing_data()
        
        # Categorize
        categorized = self._categorize_symbols(symbols, existing_data, today)
        
        logger.info(f"[fetcher.py] 📊 Plan:")
        logger.info(f"[fetcher.py]   New: {len(categorized['new'])}")
        logger.info(f"[fetcher.py]   Update: {len(categorized['update'])}")
        logger.info(f"[fetcher.py]   Skip: {len(categorized['skip'])}")
        
        # Process
        all_data = []
        
        # 1. Keep current
        if categorized['skip']:
            skip_data = existing_data[existing_data['symbol'].isin(categorized['skip'])]
            all_data.append(skip_data)
            self.stats['skipped_symbols'] = len(categorized['skip'])
        
        # 2. Fetch new (parallel batches)
        if categorized['new']:
            logger.info(f"[fetcher.py] \\n📥 Fetching {len(categorized['new'])} new symbols...")
            new_data = self._fetch_batch(categorized['new'], start_date, today)
            if new_data:
                all_data.extend(new_data)
        
        # 3. Update existing (only recent days)
        if categorized['update']:
            logger.info(f"[fetcher.py] \\n🔄 Updating {len(categorized['update'])} symbols...")
            update_data = self._update_batch(categorized['update'], existing_data, today)
            if update_data:
                all_data.extend(update_data)
        
        # Save
        if all_data:
            self._save_data(all_data)
        
        # Stats
        elapsed = time.time() - start_time
        logger.info(f"[fetcher.py] \\n⏱️  Completed in {elapsed/60:.1f} minutes")
        logger.info(f"[fetcher.py] 📊 Stats: {self.stats['updated_symbols']} updated, {self.stats['api_calls']} API calls")
    
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
                if (today - latest).days > 1:
                    categorized['update'].append(symbol)
                else:
                    categorized['skip'].append(symbol)
        
        return categorized
    
    def _fetch_batch(self, symbols, start_date, end_date):
        """Fetch batch of symbols in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = {
                executor.submit(self._fetch_symbol, sym, start_date, end_date): sym
                for sym in symbols
            }
            
            with tqdm(total=len(symbols), desc="Fetching", unit="sym") as pbar:
                for future in as_completed(futures):
                    try:
                        data = future.result()
                        if not data.empty:
                            results.append(data)
                            self.stats['updated_symbols'] += 1
                        pbar.set_postfix({'success': len(results)})
                    except:
                        self.stats['failed_symbols'] += 1
                    pbar.update(1)
        
        return results
    
    def _update_batch(self, symbols, existing_data, today):
        """Update batch - only fetch last few days"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            futures = []
            
            for symbol in symbols:
                symbol_data = existing_data[existing_data['symbol'] == symbol]
                latest = symbol_data['date'].max()
                
                # Only fetch missing days
                future = executor.submit(
                    self._fetch_and_merge, 
                    symbol, symbol_data, latest, today
                )
                futures.append(future)
            
            with tqdm(total=len(symbols), desc="Updating", unit="sym") as pbar:
                for future in futures:
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
            if not chunk.empty:
                all_chunks.append(chunk)
                self.stats['total_chunks'] += 1
            
            current = chunk_end + timedelta(days=1)
            time.sleep(0.1)  # Small delay between chunks
        
        if all_chunks:
            df = pd.concat(all_chunks, ignore_index=True)
            df.drop_duplicates(subset=['date'], keep='last', inplace=True)
            df.sort_values('date', inplace=True)
            return df
        
        return pd.DataFrame()
    
    def _fetch_chunk(self, symbol, start_date, end_date):
        """Fetch single chunk"""
        
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limiter.wait()
                
                data = self.client.get_history(symbol, start_date, end_date)
                self.stats['api_calls'] += 1
                
                if data.get("s") == "ok" and "candles" in data and data["candles"]:
                    df = pd.DataFrame(
                        data["candles"],
                        columns=["timestamp", "open", "high", "low", "close", "volume"]
                    )
                    df["volume"] /= 1e5
                    df["symbol"] = symbol
                    df["date"] = pd.to_datetime(df["timestamp"], unit="s")
                    return df[["symbol", "date", "open", "high", "low", "close", "volume"]]
                
                return pd.DataFrame()
            
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
        
        return pd.DataFrame()
    
    def _fetch_and_merge(self, symbol, existing, from_date, to_date):
        """Fetch new data and merge"""
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
                logger.info(f"[fetcher.py] 📂 Loaded: {len(df)} rows, {df['symbol'].nunique()} symbols")
                return df
            except:
                pass
        
        logger.info("[fetcher.py] 📂 No existing data")
        return pd.DataFrame()
    
    def _save_data(self, data_list):
        """Save combined data"""
        df = pd.concat(data_list, ignore_index=True)
        df.drop_duplicates(subset=['symbol', 'date'], keep='last', inplace=True)
        df.sort_values(by=['symbol', 'date'], inplace=True)
        
        HISTORICAL_DATA_FILE.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(HISTORICAL_DATA_FILE, index=False, compression='snappy')
        
        size_mb = HISTORICAL_DATA_FILE.stat().st_size / (1024 * 1024)
        logger.info(f"[fetcher.py] 💾 Saved: {len(df)} rows, {df['symbol'].nunique()} symbols ({size_mb:.1f} MB)")


__all__ = ['HistoricalDataFetcher', 'FetchConfig']
'''


def optimize_fetcher():
    """Replace with optimized version"""
    file_path = Path('data/fetchers/historical.py')
    
    print("=" * 70)
    print("⚡ OPTIMIZING FETCHER FOR SPEED")
    print("=" * 70)
    
    # Backup
    if file_path.exists():
        import shutil
        backup = file_path.with_suffix('.py.backup')
        shutil.copy2(file_path, backup)
        print(f"📦 Backup: {backup}")
    
    # Write optimized version
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(FAST_FETCHER)
    
    print("✅ Optimized fetcher installed")
    print("\n⚡ Speed improvements:")
    print("  • 8 parallel workers (was 3)")
    print("  • 4 API calls/sec (was 2)")
    print("  • Faster retries")
    print("  • Smart incremental updates")
    print("\n📊 Expected performance:")
    print("  • 500 symbols: ~20-30 min (was 60+ min)")
    print("  • Daily update: ~5 min")
    print("=" * 70)


if __name__ == "__main__":
    optimize_fetcher()
    print("\n🎯 Test: python main.py --fetch-history")
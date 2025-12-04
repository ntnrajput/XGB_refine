# analyze_historical_data_fast.py - FAST version with progress bars

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


class FastHistoricalAnalyzer:
    """Fast analysis using vectorized operations"""
    
    def __init__(self, data_file: str = "outputs/data/all_symbols_history.parquet"):
        self.data_file = Path(data_file)
        self.df = None
        
        print("=" * 80)
        print("⚡ FAST HISTORICAL DATA ANALYZER")
        print("=" * 80)
    
    def load_data(self):
        """Load data"""
        print(f"\n📂 Loading: {self.data_file}")
        
        if not self.data_file.exists():
            print(f"❌ Not found: {self.data_file}")
            return False
        
        self.df = pd.read_parquet(self.data_file)
        self.df['date'] = pd.to_datetime(self.df['date'])

        print(self.df)
        
        file_size = self.data_file.stat().st_size / (1024 * 1024)
        print(f"✅ Loaded: {self.df.shape[0]:,} rows, {file_size:.1f} MB")
        return True
    
    def analyze_all(self):
        """Run all analyses with progress tracking"""
        print("\n" + "=" * 80)
        print("🔍 RUNNING ANALYSIS")
        print("=" * 80)
        
        analyses = [
            ("Basic Stats", self.basic_stats),
            ("Symbol Analysis", self.symbol_analysis_fast),
            ("Date Coverage", self.date_coverage),
            ("Data Quality", self.data_quality),
            ("Price Analysis", self.price_analysis),
            ("Volume Analysis", self.volume_analysis),
            ("Temporal Patterns", self.temporal_patterns),
        ]
        
        for name, func in tqdm(analyses, desc="Running analyses"):
            func()
        
        self.generate_report()
    
    def basic_stats(self):
        """Basic statistics - FAST"""
        stats = {
            'total_rows': len(self.df),
            'total_symbols': self.df['symbol'].nunique(),
            'date_start': self.df['date'].min(),
            'date_end': self.df['date'].max(),
            'total_days': (self.df['date'].max() - self.df['date'].min()).days,
        }
        
        self.basic = stats
    
    def symbol_analysis_fast(self):
        """Symbol analysis using FAST groupby operations"""
        print("\n⚡ Analyzing symbols (vectorized)...")
        
        # Use groupby with agg - MUCH faster than loops
        symbol_stats = self.df.groupby('symbol').agg({
            'date': ['min', 'max', 'count'],
            'close': ['mean', 'min', 'max', 'std'],
            'volume': ['mean', 'sum'],
            'open': 'count'  # For record count
        }).reset_index()
        
        # Flatten column names
        symbol_stats.columns = [
            'symbol', 
            'date_start', 'date_end', 'date_count',
            'avg_price', 'min_price', 'max_price', 'price_std',
            'avg_volume', 'total_volume',
            'total_records'
        ]
        
        # Calculate derived metrics
        symbol_stats['days_span'] = (symbol_stats['date_end'] - symbol_stats['date_start']).dt.days
        symbol_stats['price_range'] = symbol_stats['max_price'] - symbol_stats['min_price']
        
        # Calculate volatility per symbol (faster with groupby)
        returns = self.df.groupby('symbol')['close'].pct_change()
        vol_by_symbol = returns.groupby(self.df['symbol']).std() * np.sqrt(252) * 100
        symbol_stats['volatility'] = symbol_stats['symbol'].map(vol_by_symbol)
        
        self.symbol_stats = symbol_stats
    
    def date_coverage(self):
        """Date coverage - FAST"""
        date_stats = self.df.groupby('date').size()
        
        self.date_coverage_data = {
            'trading_days': len(date_stats),
            'avg_records_per_day': date_stats.mean(),
            'min_records': date_stats.min(),
            'max_records': date_stats.max(),
        }
    
    def data_quality(self):
        """Data quality - FAST"""
        quality = {}
        
        # Vectorized null check
        quality['nulls'] = self.df.isnull().sum().to_dict()
        
        # Vectorized zero check
        quality['zeros'] = {
            col: (self.df[col] == 0).sum() 
            for col in ['open', 'high', 'low', 'close', 'volume']
        }
        
        # Vectorized negative check
        quality['negatives'] = {
            col: (self.df[col] < 0).sum()
            for col in ['open', 'high', 'low', 'close', 'volume']
        }
        
        # Invalid OHLC - vectorized
        quality['invalid_ohlc'] = (
            (self.df['high'] < self.df['low']) |
            (self.df['high'] < self.df['open']) |
            (self.df['high'] < self.df['close']) |
            (self.df['low'] > self.df['open']) |
            (self.df['low'] > self.df['close'])
        ).sum()
        
        # Duplicates
        quality['duplicates'] = self.df.duplicated(subset=['symbol', 'date']).sum()
        
        self.quality = quality
    
    def price_analysis(self):
        """Price analysis - FAST"""
        price_stats = {
            'mean': self.df['close'].mean(),
            'median': self.df['close'].median(),
            'std': self.df['close'].std(),
            'min': self.df['close'].min(),
            'max': self.df['close'].max(),
        }
        
        # Price distribution - vectorized
        bins = [0, 100, 500, 1000, 5000, 10000, float('inf')]
        labels = ['<100', '100-500', '500-1K', '1K-5K', '5K-10K', '>10K']
        price_dist = pd.cut(self.df['close'], bins=bins, labels=labels).value_counts()
        
        self.price_stats = price_stats
        self.price_dist = price_dist
    
    def volume_analysis(self):
        """Volume analysis - FAST"""
        vol_stats = {
            'mean': self.df['volume'].mean(),
            'median': self.df['volume'].median(),
            'std': self.df['volume'].std(),
            'min': self.df['volume'].min(),
            'max': self.df['volume'].max(),
        }
        
        # Volume distribution
        bins = [0, 1, 5, 10, 50, 100, float('inf')]
        labels = ['<1L', '1-5L', '5-10L', '10-50L', '50-100L', '>100L']
        vol_dist = pd.cut(self.df['volume'], bins=bins, labels=labels).value_counts()
        
        self.vol_stats = vol_stats
        self.vol_dist = vol_dist
    
    def temporal_patterns(self):
        """Temporal patterns - FAST"""
        self.df['year'] = self.df['date'].dt.year
        self.df['day_of_week'] = self.df['date'].dt.day_name()
        
        year_counts = self.df['year'].value_counts().sort_index()
        dow_counts = self.df['day_of_week'].value_counts()
        
        self.year_counts = year_counts
        self.dow_counts = dow_counts
    
    def generate_report(self):
        """Generate comprehensive report"""
        print("\n" + "=" * 80)
        print("📊 ANALYSIS REPORT")
        print("=" * 80)
        
        # Basic Stats
        print(f"\n1️⃣  DATASET OVERVIEW")
        print(f"   • Total Records: {self.basic['total_rows']:,}")
        print(f"   • Total Symbols: {self.basic['total_symbols']:,}")
        print(f"   • Date Range: {self.basic['date_start'].date()} to {self.basic['date_end'].date()}")
        print(f"   • Duration: {self.basic['total_days']:,} days ({self.basic['total_days']/365:.1f} years)")
        
        # Symbol Stats
        print(f"\n2️⃣  SYMBOL STATISTICS")
        print(f"   • Records per Symbol:")
        print(f"      - Mean: {self.symbol_stats['total_records'].mean():.0f}")
        print(f"      - Median: {self.symbol_stats['total_records'].median():.0f}")
        print(f"      - Min: {self.symbol_stats['total_records'].min()}")
        print(f"      - Max: {self.symbol_stats['total_records'].max()}")
        
        print(f"\n   • Top 10 Symbols by Records:")
        top_10 = self.symbol_stats.nlargest(10, 'total_records')
        for _, row in top_10.iterrows():
            print(f"      {row['symbol']:25s} {row['total_records']:6,} records ({row['days_span']:,} days)")
        
        print(f"\n   • Price Summary:")
        print(f"      - Average Price: ₹{self.symbol_stats['avg_price'].mean():,.2f}")
        print(f"      - Highest Price Ever: ₹{self.symbol_stats['max_price'].max():,.2f}")
        print(f"      - Lowest Price Ever: ₹{self.symbol_stats['min_price'].min():.2f}")
        
        print(f"\n   • Volatility (Top 10 Most Volatile):")
        most_volatile = self.symbol_stats.nlargest(10, 'volatility')
        for _, row in most_volatile.iterrows():
            print(f"      {row['symbol']:25s} {row['volatility']:.2f}%")
        
        # Date Coverage
        print(f"\n3️⃣  DATE COVERAGE")
        print(f"   • Trading Days: {self.date_coverage_data['trading_days']:,}")
        print(f"   • Records per Day:")
        print(f"      - Average: {self.date_coverage_data['avg_records_per_day']:.0f}")
        print(f"      - Min: {self.date_coverage_data['min_records']}")
        print(f"      - Max: {self.date_coverage_data['max_records']}")
        
        days_old = (datetime.now() - self.basic['date_end']).days
        print(f"   • Data Freshness: {days_old} days old")
        if days_old > 7:
            print(f"      ⚠️  Data needs updating")
        else:
            print(f"      ✅ Data is recent")
        
        # Data Quality
        print(f"\n4️⃣  DATA QUALITY")
        
        has_issues = False
        
        # Check nulls
        null_issues = {k: v for k, v in self.quality['nulls'].items() if v > 0}
        if null_issues:
            print(f"   ⚠️  Null Values:")
            for col, count in null_issues.items():
                print(f"      - {col}: {count:,}")
            has_issues = True
        
        # Check zeros
        zero_issues = {k: v for k, v in self.quality['zeros'].items() if v > 0}
        if zero_issues:
            print(f"   ⚠️  Zero Values:")
            for col, count in zero_issues.items():
                print(f"      - {col}: {count:,}")
            has_issues = True
        
        # Check negatives
        neg_issues = {k: v for k, v in self.quality['negatives'].items() if v > 0}
        if neg_issues:
            print(f"   ❌ Negative Values:")
            for col, count in neg_issues.items():
                print(f"      - {col}: {count:,}")
            has_issues = True
        
        if self.quality['invalid_ohlc'] > 0:
            print(f"   ❌ Invalid OHLC: {self.quality['invalid_ohlc']:,}")
            has_issues = True
        
        if self.quality['duplicates'] > 0:
            print(f"   ⚠️  Duplicates: {self.quality['duplicates']:,}")
            has_issues = True
        
        if not has_issues:
            print(f"   ✅ Excellent quality - No issues found!")
        
        # Price Analysis
        print(f"\n5️⃣  PRICE DISTRIBUTION")
        print(f"   • Statistics:")
        print(f"      - Mean: ₹{self.price_stats['mean']:,.2f}")
        print(f"      - Median: ₹{self.price_stats['median']:,.2f}")
        print(f"      - Range: ₹{self.price_stats['min']:.2f} - ₹{self.price_stats['max']:,.2f}")
        
        print(f"\n   • Distribution:")
        for label, count in self.price_dist.sort_index().items():
            pct = (count / len(self.df)) * 100
            bar = '█' * int(pct / 2)
            print(f"      ₹{label:10s} {count:8,} ({pct:5.2f}%) {bar}")
        
        # Volume Analysis
        print(f"\n6️⃣  VOLUME DISTRIBUTION")
        print(f"   • Statistics (in lakhs):")
        print(f"      - Mean: {self.vol_stats['mean']:.2f}L")
        print(f"      - Median: {self.vol_stats['median']:.2f}L")
        print(f"      - Range: {self.vol_stats['min']:.2f}L - {self.vol_stats['max']:.2f}L")
        
        print(f"\n   • Distribution:")
        for label, count in self.vol_dist.sort_index().items():
            pct = (count / len(self.df)) * 100
            bar = '█' * int(pct / 2)
            print(f"      {label:10s} {count:8,} ({pct:5.2f}%) {bar}")
        
        # Temporal
        print(f"\n7️⃣  TEMPORAL PATTERNS")
        print(f"   • Records by Year:")
        for year, count in self.year_counts.items():
            symbols_that_year = self.df[self.df['year'] == year]['symbol'].nunique()
            print(f"      {year}: {count:8,} records ({symbols_that_year:4,} symbols)")
        
        print(f"\n   • Records by Day of Week:")
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for day in day_order:
            if day in self.dow_counts.index:
                count = self.dow_counts[day]
                pct = (count / len(self.df)) * 100
                print(f"      {day:10s}: {count:8,} ({pct:5.2f}%)")
        
        # Save report
        self.save_reports()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
    
    def save_reports(self):
        """Save reports to files"""
        output_dir = Path("outputs/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save symbol stats
        csv_file = output_dir / f"symbol_analysis_{timestamp}.csv"
        self.symbol_stats.to_csv(csv_file, index=False)
        print(f"\n💾 Saved: {csv_file}")
        
        # Save summary stats
        summary_file = output_dir / f"summary_{timestamp}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"Historical Data Analysis Summary\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"\n")
            f.write(f"Total Records: {self.basic['total_rows']:,}\n")
            f.write(f"Total Symbols: {self.basic['total_symbols']:,}\n")
            f.write(f"Date Range: {self.basic['date_start'].date()} to {self.basic['date_end'].date()}\n")
        
        print(f"💾 Saved: {summary_file}")


def main():
    analyzer = FastHistoricalAnalyzer()
    
    if analyzer.load_data():
        analyzer.analyze_all()


if __name__ == "__main__":
    main()
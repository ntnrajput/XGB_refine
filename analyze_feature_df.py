# analyze_feature_df.py - Deep analysis of feature_df.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 8)


class FeatureDataFrameAnalyzer:
    """
    Comprehensive analysis of feature DataFrame.
    
    Analyzes:
    - Data quality
    - Feature distributions
    - Correlations
    - Missing values
    - Outliers
    - Feature importance
    - Temporal patterns
    """
    
    def __init__(self, csv_file: str = "featured_df.csv"):
        self.csv_file = Path(csv_file)
        self.df = None
        self.output_dir = Path("outputs/analysis")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self):
        """Load CSV file"""
        print("=" * 80)
        print("📊 FEATURE DATAFRAME ANALYZER")
        print("=" * 80)
        
        if not self.csv_file.exists():
            print(f"❌ File not found: {self.csv_file}")
            return False
        
        print(f"\n📂 Loading: {self.csv_file}")
        
        try:
            self.df = pd.read_csv(self.csv_file)
            
            # Convert date if present
            if 'date' in self.df.columns:
                self.df['date'] = pd.to_datetime(self.df['date'])
            
            file_size = self.csv_file.stat().st_size / (1024 * 1024)
            
            print(f"✅ Loaded successfully!")
            print(f"   Rows: {len(self.df):,}")
            print(f"   Columns: {len(self.df.columns)}")
            print(f"   File size: {file_size:.2f} MB")
            print(f"   Memory: {self.df.memory_usage(deep=True).sum() / (1024**2):.2f} MB")
            
            return True
        
        except Exception as e:
            print(f"❌ Error loading file: {e}")
            return False
    
    def analyze_all(self):
        """Run all analyses"""
        if self.df is None:
            print("❌ No data loaded")
            return
        
        print("\n" + "=" * 80)
        print("🔍 RUNNING COMPREHENSIVE ANALYSIS")
        print("=" * 80)
        
        # Run analyses
        self.basic_info()
        self.data_quality_check()
        self.column_analysis()
        self.feature_statistics()
        self.missing_values_analysis()
        self.distribution_analysis()
        self.correlation_analysis()
        self.outlier_detection()
        
        if 'symbol' in self.df.columns:
            self.symbol_analysis()
        
        if 'date' in self.df.columns:
            self.temporal_analysis()
        
        if 'label' in self.df.columns:
            self.label_analysis()
            self.feature_importance_analysis()
        
        # Generate report
        self.generate_html_report()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE")
        print("=" * 80)
    
    def basic_info(self):
        """Basic dataset information"""
        print("\n" + "=" * 80)
        print("1️⃣  BASIC INFORMATION")
        print("=" * 80)
        
        print(f"\n📊 Shape: {self.df.shape}")
        print(f"   Rows: {self.df.shape[0]:,}")
        print(f"   Columns: {self.df.shape[1]}")
        
        print(f"\n📋 Column Types:")
        dtype_counts = self.df.dtypes.value_counts()
        for dtype, count in dtype_counts.items():
            print(f"   {dtype}: {count} columns")
        
        print(f"\n💾 Memory Usage:")
        memory_per_col = self.df.memory_usage(deep=True)
        print(f"   Total: {memory_per_col.sum() / (1024**2):.2f} MB")
        print(f"   Per column (top 5):")
        for col, mem in memory_per_col.nlargest(5).items():
            print(f"      {col}: {mem / (1024**2):.2f} MB")
    
    def data_quality_check(self):
        """Check data quality"""
        print("\n" + "=" * 80)
        print("2️⃣  DATA QUALITY CHECK")
        print("=" * 80)
        
        # Null values
        null_counts = self.df.isnull().sum()
        null_cols = null_counts[null_counts > 0].sort_values(ascending=False)
        
        if len(null_cols) > 0:
            print(f"\n⚠️  Null Values Found ({len(null_cols)} columns):")
            for col, count in null_cols.head(10).items():
                pct = (count / len(self.df)) * 100
                print(f"   {col:30s}: {count:8,} ({pct:5.2f}%)")
            
            if len(null_cols) > 10:
                print(f"   ... and {len(null_cols) - 10} more columns with nulls")
        else:
            print("✅ No null values")
        
        # Duplicate rows
        duplicates = self.df.duplicated().sum()
        if duplicates > 0:
            print(f"\n⚠️  Duplicate Rows: {duplicates:,} ({duplicates/len(self.df)*100:.2f}%)")
        else:
            print("✅ No duplicate rows")
        
        # Infinite values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        inf_found = False
        
        for col in numeric_cols:
            inf_count = np.isinf(self.df[col]).sum()
            if inf_count > 0:
                if not inf_found:
                    print(f"\n⚠️  Infinite Values:")
                    inf_found = True
                print(f"   {col}: {inf_count:,}")
        
        if not inf_found:
            print("✅ No infinite values")
        
        # Constant columns
        constant_cols = []
        for col in self.df.columns:
            if self.df[col].nunique() == 1:
                constant_cols.append(col)
        
        if constant_cols:
            print(f"\n⚠️  Constant Columns ({len(constant_cols)}):")
            for col in constant_cols[:10]:
                print(f"   {col}")
        else:
            print("✅ No constant columns")
    
    def column_analysis(self):
        """Analyze each column"""
        print("\n" + "=" * 80)
        print("3️⃣  COLUMN ANALYSIS")
        print("=" * 80)
        
        print(f"\n📋 All Columns ({len(self.df.columns)}):")
        
        # Categorize columns
        base_cols = ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']
        feature_cols = [col for col in self.df.columns if col not in base_cols]
        
        print(f"\n   Base Columns ({len([c for c in base_cols if c in self.df.columns])}):")
        for col in base_cols:
            if col in self.df.columns:
                dtype = self.df[col].dtype
                unique = self.df[col].nunique()
                print(f"      • {col:20s} ({dtype}) - {unique:,} unique values")
        
        print(f"\n   Feature Columns ({len(feature_cols)}):")
        
        # Group features
        groups = {
            'Price & Returns': [],
            'Moving Averages': [],
            'Indicators': [],
            'Candlestick': [],
            'Volume': [],
            'Labels': [],
            'Other': []
        }
        
        for col in feature_cols:
            if any(x in col.lower() for x in ['return', 'price', 'gap']):
                groups['Price & Returns'].append(col)
            elif 'sma' in col.lower() or 'ema' in col.lower():
                groups['Moving Averages'].append(col)
            elif any(x in col.lower() for x in ['rsi', 'macd', 'bb', 'atr', 'adx']):
                groups['Indicators'].append(col)
            elif any(x in col.lower() for x in ['doji', 'hammer', 'engulfing', 'star']):
                groups['Candlestick'].append(col)
            elif 'volume' in col.lower() or 'obv' in col.lower():
                groups['Volume'].append(col)
            elif 'label' in col.lower() or 'swing' in col.lower():
                groups['Labels'].append(col)
            else:
                groups['Other'].append(col)
        
        for group_name, cols in groups.items():
            if cols:
                print(f"\n   {group_name} ({len(cols)}):")
                for col in cols[:10]:
                    print(f"      • {col}")
                if len(cols) > 10:
                    print(f"      ... and {len(cols) - 10} more")
    
    def feature_statistics(self):
        """Statistical summary of features"""
        print("\n" + "=" * 80)
        print("4️⃣  FEATURE STATISTICS")
        print("=" * 80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        print(f"\n📊 Numeric Features: {len(numeric_cols)}")
        
        # Get statistics
        stats = self.df[numeric_cols].describe()
        
        print(f"\n   Summary (first 10 features):")
        print(stats.iloc[:, :10].to_string())
        
        # Save full stats
        stats_file = self.output_dir / "feature_statistics.csv"
        stats.to_csv(stats_file)
        print(f"\n💾 Full statistics saved: {stats_file}")
    
    def missing_values_analysis(self):
        """Analyze missing values pattern"""
        print("\n" + "=" * 80)
        print("5️⃣  MISSING VALUES ANALYSIS")
        print("=" * 80)
        
        missing_summary = []
        
        for col in self.df.columns:
            null_count = self.df[col].isnull().sum()
            if null_count > 0:
                missing_summary.append({
                    'column': col,
                    'missing_count': null_count,
                    'missing_pct': (null_count / len(self.df)) * 100,
                    'dtype': str(self.df[col].dtype)
                })
        
        if missing_summary:
            missing_df = pd.DataFrame(missing_summary).sort_values('missing_count', ascending=False)
            
            print(f"\n⚠️  Missing Values Summary:")
            print(f"   Columns with missing: {len(missing_df)}/{len(self.df.columns)}")
            print(f"   Total missing: {missing_df['missing_count'].sum():,}")
            
            print(f"\n   Top 20 columns with missing values:")
            print(missing_df.head(20).to_string(index=False))
            
            # Save
            missing_file = self.output_dir / "missing_values.csv"
            missing_df.to_csv(missing_file, index=False)
            print(f"\n💾 Saved: {missing_file}")
        else:
            print("✅ No missing values!")
    
    def distribution_analysis(self):
        """Analyze feature distributions"""
        print("\n" + "=" * 80)
        print("6️⃣  DISTRIBUTION ANALYSIS")
        print("=" * 80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        print(f"\n📊 Analyzing distributions for {len(numeric_cols)} numeric features...")
        
        dist_summary = []
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) > 0:
                dist_summary.append({
                    'feature': col,
                    'mean': data.mean(),
                    'median': data.median(),
                    'std': data.std(),
                    'skew': data.skew(),
                    'kurtosis': data.kurtosis(),
                    'min': data.min(),
                    'max': data.max(),
                    'q25': data.quantile(0.25),
                    'q75': data.quantile(0.75)
                })
        
        dist_df = pd.DataFrame(dist_summary)
        
        # Most skewed features
        print(f"\n   Most Right-Skewed (top 10):")
        skewed = dist_df.nlargest(10, 'skew')[['feature', 'skew']]
        for _, row in skewed.iterrows():
            print(f"      {row['feature']:30s}: {row['skew']:7.2f}")
        
        # Most left-skewed
        print(f"\n   Most Left-Skewed (top 10):")
        skewed = dist_df.nsmallest(10, 'skew')[['feature', 'skew']]
        for _, row in skewed.iterrows():
            print(f"      {row['feature']:30s}: {row['skew']:7.2f}")
        
        # Save
        dist_file = self.output_dir / "distributions.csv"
        dist_df.to_csv(dist_file, index=False)
        print(f"\n💾 Saved: {dist_file}")
    
    def correlation_analysis(self):
        """Analyze correlations"""
        print("\n" + "=" * 80)
        print("7️⃣  CORRELATION ANALYSIS")
        print("=" * 80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        print(f"\n📊 Computing correlations for {len(numeric_cols)} features...")
        
        # Compute correlation matrix
        corr_matrix = self.df[numeric_cols].corr()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:  # High correlation threshold
                    high_corr_pairs.append({
                        'feature1': corr_matrix.columns[i],
                        'feature2': corr_matrix.columns[j],
                        'correlation': corr_val
                    })
        
        if high_corr_pairs:
            corr_df = pd.DataFrame(high_corr_pairs).sort_values('correlation', key=abs, ascending=False)
            
            print(f"\n⚠️  Highly Correlated Pairs (|corr| > 0.8): {len(corr_df)}")
            print(f"\n   Top 20:")
            for _, row in corr_df.head(20).iterrows():
                print(f"      {row['feature1']:25s} ↔ {row['feature2']:25s}: {row['correlation']:6.3f}")
            
            # Save
            corr_file = self.output_dir / "high_correlations.csv"
            corr_df.to_csv(corr_file, index=False)
            print(f"\n💾 Saved: {corr_file}")
        else:
            print("✅ No highly correlated pairs (|corr| > 0.8)")
        
        # Save full correlation matrix
        corr_file = self.output_dir / "correlation_matrix.csv"
        corr_matrix.to_csv(corr_file)
        print(f"💾 Full correlation matrix: {corr_file}")
    
    def outlier_detection(self):
        """Detect outliers"""
        print("\n" + "=" * 80)
        print("8️⃣  OUTLIER DETECTION")
        print("=" * 80)
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        outlier_summary = []
        
        for col in numeric_cols:
            data = self.df[col].dropna()
            
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = ((data < lower_bound) | (data > upper_bound)).sum()
                
                if outliers > 0:
                    outlier_summary.append({
                        'feature': col,
                        'outlier_count': outliers,
                        'outlier_pct': (outliers / len(data)) * 100,
                        'lower_bound': lower_bound,
                        'upper_bound': upper_bound
                    })
        
        if outlier_summary:
            outlier_df = pd.DataFrame(outlier_summary).sort_values('outlier_count', ascending=False)
            
            print(f"\n⚠️  Features with Outliers: {len(outlier_df)}")
            print(f"\n   Top 20:")
            for _, row in outlier_df.head(20).iterrows():
                print(f"      {row['feature']:30s}: {row['outlier_count']:6,} ({row['outlier_pct']:5.2f}%)")
            
            # Save
            outlier_file = self.output_dir / "outliers.csv"
            outlier_df.to_csv(outlier_file, index=False)
            print(f"\n💾 Saved: {outlier_file}")
        else:
            print("✅ No outliers detected")
    
    def symbol_analysis(self):
        """Analyze by symbol"""
        print("\n" + "=" * 80)
        print("9️⃣  SYMBOL ANALYSIS")
        print("=" * 80)
        
        symbol_counts = self.df['symbol'].value_counts()
        
        print(f"\n📊 Total Symbols: {len(symbol_counts)}")
        print(f"   Records per symbol:")
        print(f"      Mean: {symbol_counts.mean():.0f}")
        print(f"      Median: {symbol_counts.median():.0f}")
        print(f"      Min: {symbol_counts.min()}")
        print(f"      Max: {symbol_counts.max()}")
        
        print(f"\n   Top 10 symbols by record count:")
        for symbol, count in symbol_counts.head(10).items():
            print(f"      {symbol:25s}: {count:,} records")
    
    def temporal_analysis(self):
        """Analyze temporal patterns"""
        print("\n" + "=" * 80)
        print("🔟 TEMPORAL ANALYSIS")
        print("=" * 80)
        
        print(f"\n📅 Date Range:")
        print(f"   Start: {self.df['date'].min().date()}")
        print(f"   End: {self.df['date'].max().date()}")
        print(f"   Duration: {(self.df['date'].max() - self.df['date'].min()).days} days")
        
        # Records by year
        self.df['year'] = self.df['date'].dt.year
        year_counts = self.df['year'].value_counts().sort_index()
        
        print(f"\n   Records by Year:")
        for year, count in year_counts.items():
            print(f"      {year}: {count:,}")
        
        # Records by month
        self.df['month'] = self.df['date'].dt.month
        month_counts = self.df['month'].value_counts().sort_index()
        
        print(f"\n   Records by Month (average):")
        for month, count in month_counts.items():
            avg = count / len(year_counts)
            print(f"      Month {month:2d}: {avg:,.0f} avg")
    
    def label_analysis(self):
        """Analyze labels"""
        print("\n" + "=" * 80)
        print("🏷️  LABEL ANALYSIS")
        print("=" * 80)
        
        label_counts = self.df['label'].value_counts()
        
        print(f"\n📊 Label Distribution:")
        for label, count in label_counts.items():
            pct = (count / len(self.df)) * 100
            print(f"   Label {label}: {count:,} ({pct:.2f}%)")
        
        # Class balance
        if len(label_counts) == 2:
            minority = label_counts.min()
            majority = label_counts.max()
            imbalance_ratio = majority / minority
            
            print(f"\n   Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
            
            if imbalance_ratio > 3:
                print(f"   ⚠️  Highly imbalanced dataset")
            elif imbalance_ratio > 1.5:
                print(f"   ⚠️  Moderately imbalanced")
            else:
                print(f"   ✅ Well balanced")
    
    def feature_importance_analysis(self):
        """Analyze feature importance"""
        print("\n" + "=" * 80)
        print("🎯 FEATURE IMPORTANCE ANALYSIS")
        print("=" * 80)
        
        print("\n📊 Computing feature importance (correlation with label)...")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col != 'label']
        
        importance = []
        
        for col in feature_cols:
            corr = self.df[col].corr(self.df['label'])
            if not np.isnan(corr):
                importance.append({
                    'feature': col,
                    'correlation': corr,
                    'abs_correlation': abs(corr)
                })
        
        imp_df = pd.DataFrame(importance).sort_values('abs_correlation', ascending=False)
        
        print(f"\n   Top 20 Most Important Features:")
        for _, row in imp_df.head(20).iterrows():
            print(f"      {row['feature']:30s}: {row['correlation']:7.4f}")
        
        print(f"\n   Top 20 Least Important Features:")
        for _, row in imp_df.tail(20).iterrows():
            print(f"      {row['feature']:30s}: {row['correlation']:7.4f}")
        
        # Save
        imp_file = self.output_dir / "feature_importance.csv"
        imp_df.to_csv(imp_file, index=False)
        print(f"\n💾 Saved: {imp_file}")
    
    def generate_html_report(self):
        """Generate HTML report"""
        print("\n" + "=" * 80)
        print("📄 GENERATING HTML REPORT")
        print("=" * 80)
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Feature DataFrame Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #2c3e50; }}
                h2 {{ color: #34495e; border-bottom: 2px solid #3498db; }}
                table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #3498db; color: white; }}
                .metric {{ font-size: 24px; font-weight: bold; color: #e74c3c; }}
            </style>
        </head>
        <body>
            <h1>Feature DataFrame Analysis Report</h1>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Summary</h2>
            <p>Total Rows: <span class="metric">{len(self.df):,}</span></p>
            <p>Total Columns: <span class="metric">{len(self.df.columns)}</span></p>
            <p>File: <span class="metric">{self.csv_file}</span></p>
            
            <h2>Files Generated</h2>
            <ul>
                <li>feature_statistics.csv</li>
                <li>missing_values.csv</li>
                <li>distributions.csv</li>
                <li>correlation_matrix.csv</li>
                <li>high_correlations.csv</li>
                <li>outliers.csv</li>
                <li>feature_importance.csv</li>
            </ul>
            
            <p>All files saved to: {self.output_dir}</p>
        </body>
        </html>
        """
        
        html_file = self.output_dir / "analysis_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"💾 HTML Report: {html_file}")


def main():
    """Run analysis"""
    analyzer = FeatureDataFrameAnalyzer("featured_df.csv")
    
    if analyzer.load_data():
        analyzer.analyze_all()
    else:
        print("❌ Failed to load data")


if __name__ == "__main__":
    main()
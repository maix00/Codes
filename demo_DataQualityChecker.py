import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings

# Import our DataQualityChecker
from DataQualityChecker import DataQualityChecker

def create_sample_data():
    """Create sample data with various quality issues for testing"""
    # Create datetime range
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(hours=i) for i in range(100)]
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Create sample data with issues
    data = {
        'datetime': dates,
        'symbol': ['FUTURE1'] * 100,
        'open': np.random.rand(100) * 100 + 50,
        'high': np.random.rand(100) * 100 + 55,
        'low': np.random.rand(100) * 100 + 45,
        'close': np.random.rand(100) * 100 + 50,
        'volume': np.random.randint(1000, 10000, 100),
        'row_id': range(100)  # Add row identifier to track rows
    }
    df = pd.DataFrame(data)

    df.loc[50:99, 'symbol'] = 'FUTURE2'  # Two different symbols
    
    # Introduce some data quality issues:
    # 1. Zero sequences in volume
    df.loc[10:15, 'volume'] = 0  # Short sequence
    df.loc[30:45, 'volume'] = 0  # Long sequence
    
    df.loc[0:1, 'close'] = 0  # Edge case: zeros at start

    # 2. Zero values in prices
    df.loc[25:30, 'close'] = 0
    
    # 3. Outliers
    df.loc[50, 'high'] = 500  # Artificial outlier
    
    # 4. Some string values that should be numeric
    df.loc[70, 'open'] = 'invalid'
    df.loc[71, 'close'] = 'N/A'
    df.loc[72, 'close'] = 'N//A'
    
    return df

def custom_zero_handler(df, col, start_idx, end_idx):
    """Custom zero handling function - replaces zeros with forward fill"""
    print(f"Custom zero handler called for column {col}, indices {start_idx}-{end_idx}")
    # Forward fill strategy
    prev_idx = start_idx - 1
    next_idx = end_idx + 1
    
    if prev_idx >= 0:
        fill_value = df.iloc[prev_idx][col]
        for i in range(start_idx, end_idx + 1):
            df.loc[i, col] = fill_value
    
    return df

def demo_basic_processing():
    """Demonstrate basic data quality checking"""
    print("=== Creating Sample Data ===")
    df = create_sample_data()
    print(f"Original data shape: {df.shape}")
    print("\nRows with string values (indices 70-71):")
    print(df.loc[68:73, ['row_id', 'datetime', 'open', 'close']])
    
    print("\n=== Initializing DataQualityChecker ===")
    # Initialize checker with the dataframe
    checker = DataQualityChecker(df, columns=['open', 'high', 'low', 'close', 'volume'], print_info=True)
    print(f"Minimum time interval: {checker.min_time_interval}")
    print(f"Window size in seconds: {checker.window_size_seconds}")
    
    print("\n=== Processing Data ===")
    processed_df = checker.process_dataframe()
    if processed_df is not None:
        print(f"Processed data shape: {processed_df.shape}")
    else:
        print("Warning: process_dataframe returned None")
        processed_df = df
    
    print("\nComparison of before and after:")
    print("Original Data:")
    pd.set_option('display.max_rows', 100)
    print(df)

    print("\nProcessed Data:")
    print(processed_df)

    # Save processed data using DataQualityChecker's save method
    checker.save_dataframe('../data/demo_DataQualityChecker_processed_data.csv', 'csv', False)
    checker.save_dataframe('../data/demo_DataQualityChecker_issues.csv', 'csv', True)

    return df, processed_df

if __name__ == "__main__":
    print("Data Quality Checker Demo")
    print("========================")
    
    # Run basic demo
    original_df, processed_df = demo_basic_processing()
    
    print("\n=== Summary ===")
    print(f"Original data had {len(original_df)} rows")
    print(f"Processed data has {len(processed_df)} rows")
    print("Notice how rows may have been removed due to long zero sequences")
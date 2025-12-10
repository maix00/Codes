import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DataQualityChecker import DataQualityChecker
from tqdm import tqdm

# Path to data folder
data_folder = '../data/data_mink_product_2025_0'

# List to store all processed dataframes
all_dfs = []

# Iterate through all CSV files
for filename in tqdm(os.listdir(data_folder)):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        
        # Extract symbol from filename (remove .csv extension)
        symbol = filename.replace('.csv', '')
        
        # Read CSV
        df = pd.read_csv(filepath)
        
        # Verify unique_instrument_id matches filename
        if (df['unique_instrument_id'] == symbol).all():
            # Rename column
            df.rename(columns={'unique_instrument_id': 'symbol'}, inplace=True)
            
            # Create DataQualityChecker object
            checker = DataQualityChecker(df, columns=['open_price', 'highest_price', 'lowest_price', 'close_price'])
            
            # Get issues_df
            issues = checker.issues_df
            
            # Add to list
            all_dfs.append(issues)
        else:
            print(f"Warning: {filename} has mismatched unique_instrument_id values")

# Concatenate all dataframes
if all_dfs:
    result_df = pd.concat(all_dfs, ignore_index=True)
    result_df.to_csv('../data/data_mink_product_2025_0_issues.csv', index=False)
    print(result_df)
else:
    print("No valid CSV files found")
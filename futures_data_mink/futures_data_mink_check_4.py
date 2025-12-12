"""
本文件用于批量检查指定文件夹下所有CSV文件的数据质量。
程序会遍历每个CSV文件，验证其unique_instrument_id字段是否与文件名一致，
若一致则重命名为symbol，并利用DataQualityChecker类对指定的价格字段
（open_price, highest_price, lowest_price, close_price）进行质量检查。
所有发现的问题会被收集并合并，最终输出为一个汇总的CSV文件（data_mink_product_2025_0_issues.csv）。
输入为原始数据文件夹（data_mink_product_2025_0）中的CSV文件，输出为包含所有数据质量问题的CSV文件。
"""

import os
import pandas as pd
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from DataQualityChecker import DataQualityChecker
from tqdm import tqdm

# Path to data folder
data_folder = '../data/data_mink_product_2025'

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
            
            # Create DataQualityChecker object
            checker = DataQualityChecker(df, columns=['open_price', 'highest_price', 'lowest_price', 'close_price'], 
                                         column_mapping={'symbol': 'unique_instrument_id', 'time': 'trade_time'})
            
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
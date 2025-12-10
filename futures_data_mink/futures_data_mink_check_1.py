"""
This script performs quality checks and extracts summary information from a futures data parquet file.
It reads the data, analyzes unique values in key columns, and writes the results to a timestamped log file.
Specifically, it:
- Outputs the first 10 rows of the DataFrame.
- Lists unique values in 'exchange_id', 'unique_instrument_id' (with digits removed), and 'product_id'.
- Extracts and checks the values between the first and second '|' in 'unique_instrument_id'.
- Counts unique instrument IDs for each product and provides a total count.
文件说明: 本文件用于对期货数据进行质量检查和信息提取，并将结果输出到日志文件，便于后续数据分析和核查。
"""

import pandas as pd
from datetime import datetime

# Read the parquet file
df = pd.read_parquet('../data/data_mink/data_qc_future_mink_202501.parquet')

unique_product_ids = df['product_id'].unique()
extracted = df['unique_instrument_id'].str.split('|').str[1].unique()

# Redirecting output to a text file
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
filename = f'./futures_data_mink/futures_data_mink_check_log_{timestamp}.txt'

with open(filename, 'w') as f:
    # Write the DataFrame head
    f.write("First 10 lines of the DataFrame:\n")
    f.write(df.head(10).to_string())
    f.write("\n\nUnique values in 'exchange_id':\n")
    f.write(', '.join(map(str, df['exchange_id'].unique())))
    
    f.write("\n\nUnique values in 'unique_instrument_id' (before digits):\n")
    f.write(', '.join(df['unique_instrument_id'].str.replace(r'\d+', '', regex=True).unique()))
    
    f.write("\n\nUnique values in 'product_id':\n")
    f.write(', '.join(map(str, unique_product_ids)))
    
    f.write("\n\nUnique values between first and second '|' in 'unique_instrument_id':\n")
    f.write(', '.join(extracted))
    f.write(f"\nAre all values 'F'? {all(val == 'F' for val in extracted)}")
    
    f.write(f"\n\nTotal number of records: {len(df)}\n")
    total_count = 0
    for product_id in unique_product_ids:
        filtered_df = df[df['product_id'] == product_id]
        unique_instruments = filtered_df['unique_instrument_id'].str.split('|').str[-1].str.extract(r'(\d+)$')[0].unique()
        count = len(unique_instruments)
        total_count += count
        f.write(f"Product ID: {product_id}, Unique instrument IDs after last '|': {', '.join(map(str, unique_instruments))}, Count: {count}\n")
    f.write(f"\nTotal Count of unique instrument IDs: {total_count}\n")
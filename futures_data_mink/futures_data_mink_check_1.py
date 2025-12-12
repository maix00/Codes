"""
该脚本用于分析和检查期货数据明细文件（parquet 格式），并将分析结果输出到带有时间戳的日志文本文件中。
功能说明：
1. 读取指定路径下的 parquet 数据文件，加载为 pandas DataFrame。
2. 提取并输出部分字段的唯一值，包括 'exchange_id'、'unique_instrument_id'（去除数字部分）、'product_id'。
3. 分析 'unique_instrument_id' 字段中 '|' 分隔的第二部分内容，并判断是否全部为 'F'。
4. 统计每个 'product_id' 下，'unique_instrument_id' 最后一个 '|' 后的数字部分的唯一数量，并输出详细信息。
5. 输出数据总行数和所有产品的唯一 instrument id 总数。
6. 所有分析结果均写入带有当前时间戳的日志文件，便于后续追踪和比对。
输入：
- 一个 parquet 格式的期货数据明细文件，路径为 '../data/data_mink/data_qc_future_mink_202501.parquet'。
输出：
- 一个包含数据分析结果的文本日志文件，文件名格式为 'futures_data_mink_check_log_时间戳.txt'，保存在 './futures_data_mink/' 目录下。
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

# End of file futures_data_mink/futures_data_mink_check_1.py
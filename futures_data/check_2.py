"""
期货数据检查模块

该脚本的功能：
1. 从指定目录中扫描所有匹配模式的parquet文件
2. 读取每个parquet文件中的数据
3. 收集所有文件中的唯一商品代码(unique_instrument_id)
4. 对收集到的唯一商品代码进行去重和排序
5. 将排序后的唯一商品代码列表写入到文本文件中
6. 输出统计信息，包括总的唯一商品数量和输出文件路径

输入：
    - 相对路径 '../data/data_mink' 下的所有 'data_qc_future_mink_*.parquet' 文件

输出：
    - 在当前目录生成 'unique_instrument_ids.txt' 文件，每行包含一个唯一商品代码
    - 打印总的唯一商品数量和输出文件路径

用途：用于统计和管理期货数据中的所有唯一商品代码
"""

import pandas as pd
from pathlib import Path
import glob

# Get all parquet files matching the pattern
data_dir = Path('../data/data_mink')
parquet_files = sorted(glob.glob(str(data_dir / 'data_qc_future_mink_*.parquet')))

# Collect all unique instrument IDs
all_unique_ids = set()

for file_path in parquet_files:
    df = pd.read_parquet(file_path)
    unique_ids = df['unique_instrument_id'].unique()
    all_unique_ids.update(unique_ids)

# Convert to sorted list
unique_ids_list = sorted(list(all_unique_ids))

# Write to file in current directory
output_file = Path('./futures_data/unique_instrument_ids.txt')
with open(output_file, 'w') as f:
    for instrument_id in unique_ids_list:
        f.write(f"{instrument_id}\n")

print(f"Total unique instruments: {len(unique_ids_list)}")
print(f"Results saved to {output_file}")

# End of file futures_data/check_2.py
"""
该脚本用于比较两个来源的期货合约代码列表是否一致。

主要流程如下：
1. 从 '../data/wind_mapping.csv' 文件读取期货合约映射数据，并筛选出 END_DATE 年份大于等于 2025 且不为空的合约。
2. 对筛选后的合约代码进行格式化处理，去除后缀（如 '.S', '_S' 等），并获取唯一且排序后的代码列表 codes。
3. 从 './futures_data_mink/unique_instrument_ids.txt' 文件读取所有唯一合约标识，并通过 unique_instrument_id_to_product_id 函数转换为产品代码，得到唯一且排序后的代码列表 codes2。
4. 比较 codes 和 codes2 是否完全一致，若不一致则分别输出两者的差异。

用途：
用于校验 wind_mapping.csv 文件中的合约代码与 unique_instrument_ids.txt 文件中的合约标识转换后是否一致，辅助数据清洗和映射准确性检查。

依赖：
- pandas
- futures_data_mink.DataMinkBasics.unique_instrument_id_to_product_id
"""

import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from futures_data_mink.DataMinkBasics import unique_instrument_id_to_product_id

# 读取CSV文件
df = pd.read_csv('../data/wind_mapping.csv')
# 过滤出ENDDATE年份大于等于2025的行
df = df[df['ENDDATE'].notna() & (df['ENDDATE'].astype(str).str.strip() != '')]  # Remove rows where ENDDATE is NaN or blank
df = df[df['ENDDATE'].astype(str).str[:4].astype(int) >= 2025]
# 提取第一列并转换为字符串类型
codes = df['S_INFO_WINDCODE'].astype(str)
# 提取'.'之前的部分
codes = codes.str.split('.', n=1).str[0]
# 提取'-S'或'_S'之前的部分
codes = codes.str.split('-S|_S', n=1, regex=True).str[0]

# 获取唯一值并排序
codes = sorted(codes.unique().tolist())

# 直接提取./futures_data_mink/unique_instrument_ids中每一行的字符串
with open('./futures_data_mink/unique_instrument_ids.txt', 'r') as f:
    file_names = [line.strip() for line in f if line.strip()]
# 对每个文件名应用 unique_instrument_id_to_product_id
codes2 = [unique_instrument_id_to_product_id(name) for name in file_names]
# 获取唯一值并排序
codes2 = sorted(set(codes2))

if codes == codes2:
    print("codes and codes2 are identical.")
else:
    print("codes not in codes2:", sorted(set(codes) - set(codes2)))
    print("codes2 not in codes:", sorted(set(codes2) - set(codes)))
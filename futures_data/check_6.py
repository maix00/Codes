import pandas as pd

# 读取 parquet 文件
df1 = pd.read_parquet('../data/data_dayk.parquet')
df2 = pd.read_parquet('../data/wind_mapping.parquet')

# 截取 trading_day 截止到 2025-10-01
df1_truncated = df1[df1['trading_day'] <= '2025-10-01']
df2_truncated = df2[df2['STARTDATE'] <= '20251001']

# 保存为新的 parquet 文件
df1_truncated.to_parquet('../data/data_dayk_truncated.parquet', index=False)
df2_truncated.to_parquet('../data/wind_mapping_truncated.parquet', index=False)
df2_truncated.to_csv('../data/wind_mapping_truncated.csv', index=False)

# End of file futures_data/check_6.py
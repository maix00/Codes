import pandas as pd

# 读取Parquet文件的前几行
file_path = '../data/data_dayk.parquet'
df = pd.read_parquet(file_path)

# 打印前几行
pd.set_option('display.max_columns', None)
print(df.head())
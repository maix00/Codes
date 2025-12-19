import pandas as pd

# 读取Parquet文件的前几行
file_path = '../data/data_dayk.parquet'
# file_path = '../data/wind_mapping.parquet'
df = pd.read_parquet(file_path)

# 打印前几行
pd.set_option('display.max_columns', None)
print(df.head())

csv_path = file_path.replace('.parquet', '.csv')
df.to_csv(csv_path, index=False)
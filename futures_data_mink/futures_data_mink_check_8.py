import pandas as pd
import os

# 读取rollover_adjustments.csv
rollover_df = pd.read_csv('rollover_adjustments.csv')

# 假设rollover_adjustments.csv有两列: 'product', 'cutoff_date'
# 例如:
# product,cutoff_date
# IF,2015-01-01
# IC,2016-01-01

# 存储截断后的数据的文件夹
output_dir = 'truncated_products'
os.makedirs(output_dir, exist_ok=True)

for idx, row in rollover_df.iterrows():
    product = row['product']
    cutoff_date = pd.to_datetime(row['cutoff_date'])

    # 假设每个产品的数据文件名为 '{product}.csv'
    product_file = f'{product}.csv'
    if not os.path.exists(product_file):
        print(f"文件不存在: {product_file}")
        continue

    df = pd.read_csv(product_file, parse_dates=['date'])
    # 截断数据
    truncated_df = df[df['date'] >= cutoff_date].copy()
    # 存储截断后的数据
    truncated_file = os.path.join(output_dir, f'{product}_truncated.csv')
    truncated_df.to_csv(truncated_file, index=False)
    print(f"{product} 已截断并存储到 {truncated_file}")
"""
本脚本用于处理期货合约的展期（rollover）调整数据。其主要功能如下：

功能说明：
1. 读取产品与合约的映射关系（CSV文件），筛选出需要处理的产品列表。
2. 对每个产品，利用DataMinkBasics.RolloverDetector检测展期点。
3. 使用PercentageAdjustmentStrategy策略，判断每个展期点是否有效，并计算价格调整值。
4. 将每个展期点的产品ID、老合约和新合约的唯一标识、展期日期及调整值整理为DataFrame。
5. 汇总所有产品的展期调整结果，并输出为CSV文件。

输入：
- 产品与合约的映射关系CSV文件（如：../data/wind_mapping.csv）
- 合约行情数据目录（如：../data/data_mink_product_2025/）

输出：
- 包含所有产品展期调整信息的CSV文件（如：./futures_data_mink/rollover_adjustments.csv）

输出字段包括：
- product_id：产品ID
- old_unique_instrument_id：老合约唯一标识
- new_unique_instrument_id：新合约唯一标识
- rollover_date：展期日期
- adjustment：展期价格调整值
"""

import DataMinkBasics
from StrategySelector import ProductPeriodStrategySelector
from AdjustmentStrategy import PercentageAdjustmentStrategy
import pandas as pd
from tqdm import tqdm
import os

data_mink_path = '../data/data_mink_product_2025/'
data_mink_path_main = '../data/data_mink_product_2025_main/'
os.makedirs(data_mink_path_main, exist_ok=True)

product_contract_start_end_path = '../data/wind_mapping.csv'
product_contract_start_end = pd.read_csv(product_contract_start_end_path)
product_contract_start_end.columns.values[0] = "PRODUCT"
product_contract_start_end.columns.values[1] = "CONTRACT"

product_id_list = product_contract_start_end[~product_contract_start_end['PRODUCT'].str.contains('_S')]['PRODUCT'].str.split('.').str[0].unique().tolist()
# product_id_list = ['A']

all_adjustments = []
all_issues = []
for product_id in tqdm(product_id_list, desc="Processing products"):
    detector = DataMinkBasics.RolloverDetector()
    data = product_contract_start_end[product_contract_start_end['PRODUCT'].str.startswith(product_id + '.')]
    detector.add_data_table('product_contract_start_end', data)
    detector.generate_main_contract_series(path=data_mink_path)
    strategy_selector = ProductPeriodStrategySelector(default_strategy={
        "AdjustmentStrategy": PercentageAdjustmentStrategy(
            old_price_field='close_price', new_price_field='close_price', 
            new_price_old_data_bool=True, use_window=False
        )
    })
    all_adjustments.append(detector.get_adjustment_factor(strategy_selector))
    main_contract_series = detector.data_tables.get('main_tick')
    if main_contract_series is not None:
        output_path = f"{data_mink_path_main}{product_id}.csv"
        main_contract_series.to_csv(output_path, index=False)
    all_issues.append(detector.data_tables.get('main_tick_issues', pd.DataFrame()))

# 拼接所有结果并输出到csv
adjustments_df = pd.concat(all_adjustments, ignore_index=True)
# print(adjustments_df)
adjustments_df.to_csv('../data/rollover_adjustments.csv', index=False)

issues_df = pd.concat(all_issues, ignore_index=True)
issues_df.to_csv('../data/main_tick_issues.csv', index=False)


# End of file futures_data_mink/futures_data_mink_check_7.py
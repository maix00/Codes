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

# data_mink_path = '../data/data_mink_product_2025/'
data_mink_path_main = '../data/data_mink_main_dayk/'
os.makedirs(data_mink_path_main, exist_ok=True)

product_contract_start_end_path = '../data/wind_mapping.parquet'
PCSE = pd.read_parquet(product_contract_start_end_path)
PCSE.columns.values[0] = "PRODUCT"
PCSE.columns.values[1] = "CONTRACT"

contract_dayk_path = '../data/data_dayk.parquet'

product_id_list = PCSE[~PCSE['PRODUCT'].str.contains('_S|-S')]['PRODUCT'].str.split('.').str[0].unique().tolist()
# product_id_list = ['OI']

all_adjustments = []
all_main_ticks = []
all_main_ticks_adjusted = []
all_issues = []
for product_id in tqdm(product_id_list, desc="Processing products"):
    detector = DataMinkBasics.FuturesProcessor()
    detector.add_data_table('product_contract_start_end', PCSE[PCSE['PRODUCT'].str.startswith(product_id + '.')])
    detector.add_data_table('contract_dayk', pd.read_parquet(contract_dayk_path))
    strategy_selector = ProductPeriodStrategySelector(default_strategy={
        "AdjustmentStrategy": PercentageAdjustmentStrategy(
            old_price_field='close_price', new_price_field='close_price', 
            new_price_old_data_bool=True, use_window=False
        )
    })
    detector.generate_main_contract_series_adjusted(strategy_selector=strategy_selector)
    
    adjustment_df = detector.data_tables.get('adjustment_factors')
    if adjustment_df is not None and not adjustment_df.empty and len(adjustment_df) > 0:
        all_adjustments.append(adjustment_df)
    
    main_contract_series = detector.data_tables.get('main_tick')
    if main_contract_series is not None and not main_contract_series.empty:
        # output_path = f"{data_mink_path_main}{product_id}.csv"
        # main_contract_series.to_csv(output_path, index=False)
        all_main_ticks.append(main_contract_series)

    main_contract_series = detector.data_tables.get('main_tick_adjusted')
    if main_contract_series is not None and not main_contract_series.empty:
        all_main_ticks_adjusted.append(main_contract_series)
    
    main_contract_series_issue = detector.data_tables.get('main_tick_issues')
    if main_contract_series_issue is not None and not main_contract_series_issue.empty:
        all_issues.append(main_contract_series_issue)

# 拼接所有结果并输出到csvs
if all_adjustments:
    adjustments_df = pd.concat([df.dropna(axis=1, how='all') for df in all_adjustments], ignore_index=True)
    adjustments_df.to_csv('../data/data_mink_rollover_adjustments.csv', index=False)
else:
    print("No adjustments found.")

if all_main_ticks_adjusted:
    main_ticks_adjusted = pd.concat(all_main_ticks_adjusted, ignore_index=True)
    main_ticks_adjusted.to_csv('../data/data_mink_main_ticks_adjusted.csv', index=False)
else:
    print("No adjusted main ticks found.")

if all_main_ticks:
    main_ticks = pd.concat(all_main_ticks, ignore_index=True)
    main_ticks.to_csv('../data/data_mink_main_dayk.csv', index=False)
else:
    print("No main ticks found.")

if all_issues:
    issues_df = pd.concat(all_issues, ignore_index=True)
    issues_df.to_csv('../data/data_mink_main_dayk_issues.csv', index=False)
else:
    print("No issues found.")

# End of file futures_data_mink/futures_data_mink_check_7.py
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
from AdjustmentStrategy import PercentageAdjustmentStrategy
import pandas as pd
from tqdm import tqdm

data_mink_path = '../data/data_mink_product_2025/'
product_contract_start_end_path = '../data/wind_mapping.csv'
product_contract_start_end = pd.read_csv(product_contract_start_end_path)
product_contract_start_end.columns.values[0] = "PRODUCT"
product_contract_start_end.columns.values[1] = "CONTRACT"

product_id_list = product_contract_start_end[~product_contract_start_end['PRODUCT'].str.contains('_S')]['PRODUCT'].str.split('.').str[0].unique().tolist()
# product_id_list = ['A']

all_results = []

for product_id in tqdm(product_id_list, desc="Processing products"):
    detector = DataMinkBasics.RolloverDetector()
    data = product_contract_start_end[product_contract_start_end['PRODUCT'].str.startswith(product_id + '.')]
    detector.add_data_table('product_contract_start_end', data)
    detector.detect_rollover_points(path=data_mink_path)
    strategy = PercentageAdjustmentStrategy(
        old_price_field='close_price', new_price_field='close_price', 
        new_price_old_data_bool=True, use_window=False
    )
    results = []
    for rollover in detector.rollover_points:
        is_valid, invalid_reason = strategy.is_valid(rollover)
        if is_valid:
            adjustment, _ = strategy.calculate_adjustment(rollover)
            rollover_date = rollover.new_contract_start_datetime
        else:
            adjustment = None
            rollover_date = rollover.new_contract_start_date
        # Convert contract to unique_instrument_id
        old_unique_id = DataMinkBasics.windcode_to_unique_instrument_id(rollover.old_contract)
        new_unique_id = DataMinkBasics.windcode_to_unique_instrument_id(rollover.new_contract)
        
        # 在每次循环时，将之前results中的adjustment都乘以当前adjustment（仅当adjustment不为None时）
        if adjustment is not None and len(results) > 0:
            for r in results:
                if r['adjustment'] is not None:
                    r['adjustment'] *= adjustment

        results.append({
            'product_id': product_id,
            'old_unique_instrument_id': old_unique_id,
            'new_unique_instrument_id': new_unique_id,
            'rollover_date': rollover_date,
            'adjustment': adjustment
        })
    df = pd.DataFrame(results)
    all_results.append(df)

# 拼接所有结果并输出到csv
final_df = pd.concat(all_results, ignore_index=True)
print(final_df)
final_df.to_csv('./futures_data_mink/rollover_adjustments.csv', index=False)

# End of file futures_data_mink/futures_data_mink_check_7.py
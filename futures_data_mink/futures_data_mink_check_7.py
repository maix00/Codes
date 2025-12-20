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
# from tqdm import tqdm
import os
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

data_mink_path = '../data/data_mink_product_2025/'
data_mink_path_main = '../data/data_mink_main_dayk/'
os.makedirs(data_mink_path_main, exist_ok=True)

detector = DataMinkBasics.FuturesProcessor()
detector.set_column_mapping('product_contract_start_end', {'S_INFO_WINDCODE': 'PRODUCT', 'FS_MAPPING_WINDCODE': 'CONTRACT'},)
detector.add_data_table('product_contract_start_end', pd.read_parquet('../data/wind_mapping.parquet'))
detector.add_data_table('contract_dayk', pd.read_parquet('../data/data_dayk.parquet'))
detector.rollover_points_cache_path = '../data/rollover_points_cache.pkl'
detector.rollover_adjustments_cache_path = '../data/rollover_adjustments.csv'
# detector.product_id_list = ['FU.SHF']
# detector.detect_rollover_points()
# detector.calculate_adjustment()

calculate_main_series_dayk = False
calculate_main_series_mink = False
issues_path = '../data/main_mink_issues.csv'
if calculate_main_series_dayk:
    main_series = detector.generate_main_contract_series(source_data_label='dayk', add_adjust_col_bool=True)
    main_series.to_csv('../data/main_dayk.csv', index=False)
    issues_path = '../data/main_dayk_issues.csv'
if calculate_main_series_mink:
    main_series = detector.generate_main_contract_series(source_data_label='mink', 
                                                         source_data_folder_UID_path=data_mink_path,
                                                         add_adjust_col_bool=True)
    main_series.to_csv('../data/main_mink.csv', index=False)

if hasattr(detector, 'all_issues'):
    issues_df = pd.concat(detector.all_issues, ignore_index=True) if len(detector.all_issues) > 0 else None
    if issues_df is not None and not issues_df.empty and len(issues_df) > 0:
        issues_df.to_csv(issues_path, index=False)
    else:
        print("No main tick issues detected.")

profiler.disable()
# # 输出分析结果
# stats = pstats.Stats(profiler)
# stats.sort_stats('cumulative')  # 按累计时间排序
# stats.print_stats(20)  # 显示前20个耗时最多的函数

# End of file futures_data_mink/futures_data_mink_check_7.py
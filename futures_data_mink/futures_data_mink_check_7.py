"""
本脚本用于处理期货主力合约的展期与价格调整，核心流程如下：

主要功能：
1. 读取产品与合约的映射关系（parquet文件），加载合约日行情及分钟行情数据。
2. 配置并初始化FuturesProcessor，设置字段映射及数据表。
3. 检测各产品的主力合约展期点，并根据指定策略计算展期价格调整值。
4. 生成主力合约的分钟级行情序列，并保存展期调整及相关问题报告。

输入数据：
- 产品与合约映射关系（../data/wind_mapping.parquet）
- 合约日行情数据（../data/data_dayk.parquet）
- 合约分钟行情数据目录（../data/data_mink_product_2025/）

输出数据：
- 主力合约分钟行情（../data/main_mink.parquet）
- 展期调整及问题报告（../data/main_mink_issues.csv）

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
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

detector = DataMinkBasics.FuturesProcessor()
detector.set_column_mapping('product_contract_start_end', {'S_INFO_WINDCODE': 'PRODUCT', 'FS_MAPPING_WINDCODE': 'CONTRACT'},)
detector.add_data_table('product_contract_start_end', pd.read_parquet('../data/wind_mapping.parquet'))
detector.add_data_table('contract_dayk', pd.read_parquet('../data/data_dayk.parquet'))
detector.rollover_points_cache_path = '../data/rollover_points_cache.pkl'
detector.rollover_adjustments_cache_path = '../data/rollover_adjustments.csv'
# detector.product_id_list = ['FU.SHF']
# detector.detect_rollover_points()
# detector.calculate_adjustment()
detector.generate_main_contract_series(source_data_label='dayk', add_adjust_col_bool=True,
                                       save_path='../data/main_dayk.parquet',
                                       issues_save_path='../data/main_dayk_issues.csv')
# detector.generate_main_contract_series(source_data_label='mink', 
#                                        source_data_folder_UID_path='../data/data_mink_product_2025/',
#                                        add_adjust_col_bool=True,
#                                        save_path='../data/main_mink.parquet',
#                                        issues_save_path='../data/main_mink_issues.csv')
price_cols = ['open_price', 'highest_price', 'lowest_price', 'close_price', 'twap', 'vwap',
              'settlement_price', 'upper_limit_price', 'lower_limit_price', 'pre_settlement_price']
# detector.generate_main_contract_series_adjusted(data=pd.read_parquet('../data/main_dayk.parquet'),
#                                                 save_path='../data/main_dayk_adjusted.parquet', price_cols=price_cols,
#                                                 report_bool=True, report_save_path='../data/main_dayk_adjusted_report.csv',
#                                                 plot_bool=True, plot_save_path='../data/main_dayk_adjusted_plots/')
# detector.generate_main_contract_series_adjusted(data=pd.read_parquet('../data/main_mink.parquet'),
#                                                 save_path='../data/main_mink_adjusted.parquet', price_cols=price_cols,
#                                                 report_bool=True, report_save_path='../data/main_mink_adjusted_report.csv',
#                                                 plot_bool=True, plot_save_path='../data/main_mink_adjusted_plots/',
#                                                 plot_start_date='2025-01-01', plot_end_date='2026-01-01',)

profiler.disable()
# # 输出分析结果
# stats = pstats.Stats(profiler)
# stats.sort_stats('cumulative')  # 按累计时间排序
# stats.print_stats(20)  # 显示前20个耗时最多的函数

# End of file futures_data_mink/futures_data_mink_check_7.py
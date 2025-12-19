"""
DataMinkBasics.py
本模块提供了期货合约唯一标识符（unique_instrument_id）与产品代码（product_id）、Wind代码（windcode）之间的转换工具函数。
主要内容包括：
- 交易所代码映射表（exchange_map）
- 获取当前年份十位数字的工具
- unique_instrument_id 与 product_id、windcode 之间的相互转换函数
函数说明：
- unique_instrument_id_to_product_id(unique_instrument_id: str) -> str
    将 unique_instrument_id 转换为产品代码（product_id）。如果 unique_instrument_id 以 'F' 结尾，则在 product_id 后加 '_F'。
- unique_instrument_id_to_windcode(unique_instrument_id: str) -> str
    将 unique_instrument_id 转换为 Wind 代码（windcode），格式为“产品代码+月份.交易所代码”。
- unique_instrument_id_to_windcode_simp(unique_instrument_id: str) -> str
    将 unique_instrument_id 转换为简化版 Wind 代码（windcode），月份部分去掉首位字符。
- windcode_to_unique_instrument_id(windcode: str) -> str
    将 Wind 代码（windcode）转换为 unique_instrument_id，自动处理年份十位数字的补全。
注意事项：
- 所有函数均假设 unique_instrument_id 的格式为“交易所|F|产品代码|月份”。
- 若输入格式不正确，函数会抛出 ValueError 异常。
"""

from datetime import datetime
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ContractRollover import ContractRollover
from FuturesProcessor import FuturesProcessorBase 
from StrategySelector import ProductPeriodStrategySelector
from DataQualityChecker import DataQualityChecker, DataIssueLabel, DataIssueSolution
from AdjustmentStrategy import PercentageAdjustmentStrategy, AdjustmentDirection, AdjustmentOperation
import pandas as pd
from tqdm import tqdm


# 交易所代码映射表
exchange_map = {
    "DCE": "DCE",
    "CZCE": "CZC",
    "INE": "INE",
    "SHFE": "SHF",
    "CFFEX": "CFE",
    "GFEX": "GFE"
}

# 获取今年年份的十位数字
year_tens = str(datetime.now().year)[2]

def unique_instrument_id_to_product_id(unique_instrument_id: str) -> str:
    # 第一部分：取出第二到第三个竖线之间的字母
    parts = unique_instrument_id.split('|')
    if len(parts) < 4:
        print(parts)
        raise ValueError("unique_instrument_id格式不正确")
    product_id = parts[2]

    # 第二部分：如果最后一个字母是F，加上'_F'
    if unique_instrument_id.endswith('F'):
        product_id += '_F'

    return product_id

def unique_instrument_id_to_windcode(unique_instrument_id: str) -> str:
    parts = unique_instrument_id.split('|')
    if len(parts) < 4:
        print(parts)
        raise ValueError("unique_instrument_id格式不正确")
    exchange = exchange_map.get(parts[0], parts[0])
    return f"{parts[2]}{parts[3]}.{exchange}"

def unique_instrument_id_to_windcode_simp(unique_instrument_id: str) -> str:
    parts = unique_instrument_id.split('|')
    if len(parts) < 4:
        print(parts)
        raise ValueError("unique_instrument_id格式不正确")
    exchange = exchange_map.get(parts[0], parts[0])
    return f"{parts[2]}{parts[3][1:]}.{exchange}"

def windcode_to_shortened_windcode(windcode: str) -> str:
    # 去掉第一个数字，如果数字有四位的话，否则三位则不动
    # 例如: RB2210F.DCE -> RB210F.DCE, RB210.DCE -> RB210.DCE
    try:
        product_month, exchange_code = windcode.split('.')
    except ValueError:
        raise ValueError("windcode格式不正确")
    i = 0
    while i < len(product_month) and product_month[i].isalpha():
        i += 1
    product_id = product_month[:i]
    month = product_month[i:]
    # 找出month中前面的数字
    digit_seq = ''
    j = 0
    while j < len(month) and month[j].isdigit():
        digit_seq += month[j]
        j += 1
    # 如果数字有4位，去掉第一个数字
    if len(digit_seq) == 4:
        new_month = digit_seq[1:] + month[j:]
    elif len(digit_seq) == 3:
        new_month = month
    else:
        raise ValueError("windcode中的合约月份格式不正确")
    return f"{product_id}{new_month}.{exchange_code}"

def windcode_to_unique_instrument_id(windcode: str, decade_str: str = year_tens) -> str:
    # 示例输入: "RB2210F.DCE"
    try:
        product_month, exchange_code = windcode.split('.')
    except ValueError:
        raise ValueError("windcode格式不正确")
    
    # 反向映射交易所代码
    reverse_exchange_map = {v: k for k, v in exchange_map.items()}
    exchange = reverse_exchange_map.get(exchange_code, exchange_code)

    # 提取产品代码和月份
    i = 0
    while i < len(product_month) and product_month[i].isalpha():
        i += 1
    product_id = product_month[:i]
    month = product_month[i:]
    # 计算month字符串中最开始的数字字符序列的长度
    digit_seq_len = 0
    for ch in month:
        if ch.isdigit():
            digit_seq_len += 1
        else:
            break
    if digit_seq_len == 3:
        month = decade_str + month

    # 构造unique_instrument_id
    unique_instrument_id = f"{exchange}|F|{product_id}|{month}"
    return unique_instrument_id

class FuturesProcessor(FuturesProcessorBase):
    @property
    def EXPECTED_TABLE_NAMES(self) -> List[str]:
        return ['product_contract_start_end', 'old_contract_tick', 'new_contract_tick', 
                'contract_dayk', 'contract_mink', 'main_tick', 'adjustment_factors', 'main_tick_adjusted']
    
    @property
    def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
        return {
            'product_contract_start_end': ['PRODUCT', 'CONTRACT', 'STARTDATE', 'ENDDATE'],
            'old_contract_tick': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'new_contract_tick': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'contract_dayk': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'contract_mink': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id']
        }

    @property
    def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
        return {
            'product_contract_start_end': ['PRODUCT', 'CONTRACT', 'STARTDATE', 'ENDDATE'],
            'old_contract_tick': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'new_contract_tick': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'contract_dayk': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'contract_mink': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
        }
    
    def detect_rollover_points2(self) -> Dict[str, List[ContractRollover]]:

        # 获取所有产品的合约起止日期表
        all_df = self.data_tables.get('product_contract_start_end')
        if all_df is None or all_df.empty:
            raise ValueError("'product_contract_start_end'数据表为空或不存在")
        
        # 检查是否已经有product_id_list属性
        if not hasattr(self, 'product_id_list'):
            self.product_id_list = sorted(all_df['PRODUCT'].unique().tolist())
        if not self.product_id_list:
            raise ValueError("'PRODUCT'列为空")
        
        # 遍历每个产品，检测展期点
        for product_id in tqdm(self.product_id_list, desc="Detecting rollover points"):

            # 获取该产品的合约起止日期数据
            df = all_df[all_df['PRODUCT'] == product_id].reset_index(drop=True)
            
            # 初始化该产品的展期点列表
            if not hasattr(self, 'rollover_points'):
                self.rollover_points = {}
            if product_id not in self.rollover_points:
                self.rollover_points[product_id] = []
            
            # 检查'STARTDATE'和'ENDDATE'是否都有数据
            mask_start_null = df['STARTDATE'].isnull()
            mask_end_null = df['ENDDATE'].isnull()
            both_null = mask_start_null & mask_end_null
            one_null = mask_start_null ^ mask_end_null

            # 删除'STARTDATE'和'ENDDATE'都为空的行
            df = df.drop(df.index[both_null])
            if one_null.any():
                raise ValueError(f"{product_id}: 存在'STARTDATE'或'ENDDATE'有一个为空的行")
            
            # 按'STARTDATE'排序
            df = df.sort_values('STARTDATE').reset_index(drop=True)

            # 将STARTDATE和ENDDATE全部转为date格式（兼容float/int/str）
            def to_date(val):
                if pd.isnull(val):
                    return pd.NaT
                if isinstance(val, (int, float)):
                    val = int(val)
                    return pd.to_datetime(str(val)).date()
                else:
                    return pd.to_datetime(val).date()

            df['STARTDATE'] = df['STARTDATE'].map(to_date)
            df['ENDDATE'] = df['ENDDATE'].map(to_date)

            # 打印所有行
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     if product_id == 'OI.CZC':
            #         print(df)

            # 遍历所有行，检查CONTRACT重复且ENDDATE顺序
            drop_indices = []
            for idx in range(len(df) - 1):
                curr_end = df.iloc[idx]['ENDDATE']
                next_start = df.iloc[idx + 1]['STARTDATE']
                if curr_end >= next_start:
                    raise ValueError(f"{product_id}: 第{idx}行的ENDDATE不在下一行的STARTDATE之前: {curr_end} >= {next_start}")
                curr_contract = windcode_to_shortened_windcode(df.iloc[idx]['CONTRACT'])
                next_contract = windcode_to_shortened_windcode(df.iloc[idx + 1]['CONTRACT'])
                if curr_contract == next_contract:
                    # 把下一行的STARTDATE改成当前行的STARTDATE，然后删除当前行
                    df.at[idx + 1, 'STARTDATE'] = df.iloc[idx]['STARTDATE']
                    drop_indices.append(idx)
            if drop_indices:
                df = df.drop(df.index[drop_indices]).reset_index(drop=True)

            def calculate_decade_str(row):
                enddate_str = str(row['ENDDATE'])
                if len(enddate_str) < 4:
                    raise ValueError(f"{product_id}: ENDDATE格式不正确，无法提取decade_str")
                contract_digits = ''.join([c for c in row['CONTRACT'] if c.isdigit()])
                if len(contract_digits) < 3 or len(contract_digits) > 4:
                    raise ValueError(f"{product_id}: CONTRACT格式不正确，无法提取decade_str")
                contract_third_last = contract_digits[-3]
                if contract_third_last == '0' and enddate_str[3] != '0':
                    return str(int(enddate_str[2]) + 1)
                else:
                    return enddate_str[2]
            
            # 遍历所有行，检测展期点
            for idx in range(len(df)):
                # 
                if idx + 1 >= len(df):
                    continue

                is_valid = True

                this_row = df.iloc[idx]
                next_row = df.iloc[idx + 1]

                old_contract = this_row['CONTRACT']
                new_contract = next_row['CONTRACT']

                old_contract_UID = windcode_to_unique_instrument_id(old_contract, decade_str=calculate_decade_str(this_row))
                new_contract_UID = windcode_to_unique_instrument_id(new_contract, decade_str=calculate_decade_str(next_row))

                if idx == 0:
                    self.add_data_table("old_contract_tick", self.contract_data_loader(old_contract_UID, 'contract_dayk'))
                    self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, 'contract_dayk'))
                else:
                    self.data_tables['old_contract_tick'] = self.data_tables['new_contract_tick']
                    self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, 'contract_dayk'))
                    
                # 检查数据表是否为空，跳过不合法情况
                old_tick_empty = self.data_tables["old_contract_tick"].empty
                new_tick_empty = self.data_tables["new_contract_tick"].empty
                if old_tick_empty and new_tick_empty:
                    is_valid = False
                if not old_tick_empty and new_tick_empty:
                    raise ValueError(f"{product_id}: 第{idx}行: old_contract_tick非空但new_contract_tick为空，不合法")
                
                this_start_date = pd.to_datetime(this_row['STARTDATE'])
                this_end_date = pd.to_datetime(this_row['ENDDATE'])
                next_start_date = pd.to_datetime(next_row['STARTDATE'])
                next_end_date = pd.to_datetime(next_row['ENDDATE'])

                old_trading_days = pd.to_datetime(self.data_tables['old_contract_tick']['trading_day'])
                new_trading_days = pd.to_datetime(self.data_tables['new_contract_tick']['trading_day'])
                
                # 找到小于等于this_end_date的最大日期
                mask = pd.Series([False] * len(old_trading_days), index=old_trading_days.index)
                if not old_trading_days.empty:
                    mask = old_trading_days == old_trading_days[old_trading_days <= this_end_date].max()
                if mask.any():
                    old_contract_old_data = self.data_tables['old_contract_tick'][mask].iloc[[-1]]
                else:
                    old_contract_old_data = pd.DataFrame()
                    is_valid = False
                
                # 找到小于等于this_end_date的最大日期
                mask_new = pd.Series([False] * len(new_trading_days), index=new_trading_days.index)
                if not new_trading_days.empty:
                    mask_new = new_trading_days == new_trading_days[new_trading_days <= this_end_date].max()
                if mask_new.any():
                    new_contract_old_data = self.data_tables['new_contract_tick'][mask_new].iloc[[-1]]
                else:
                    new_contract_old_data = pd.DataFrame()
                    is_valid = False
                
                rollover = ContractRollover(
                    old_contract=old_contract_UID,
                    new_contract=new_contract_UID,
                    old_contract_old_data=old_contract_old_data,
                    old_contract_new_data=pd.DataFrame(),
                    new_contract_old_data=new_contract_old_data,
                    new_contract_new_data=pd.DataFrame(),
                    old_contract_end_date=this_end_date.date(),
                    old_contract_end_datetime=None,
                    new_contract_start_date=next_start_date.date(),
                    new_contract_start_datetime=None,
                    is_valid=is_valid
                )
                self.rollover_points[product_id].append(rollover)
        return self.rollover_points
    
    def detect_rollover_points(self,
                               mink_folder_path: Optional[str] = None,
                               main_contract_series_mink_bool: bool = False,
                               generate_main_contract_series: bool = False,
                               adjust_main_contract_series: bool = False,
                               strategy_selector: Optional[ProductPeriodStrategySelector] = None) -> Dict[str, List[ContractRollover]]:
        """
        检测期货合约切换点（Rollover Points）。
        本方法根据合约起止日期表（'product_contract_start_end'）自动检测合约切换点，并加载切换点前后两个合约的行情数据。
        支持对异常数据的多重校验与处理，确保切换点的准确性和数据完整性。
            path (str): 数据文件路径，用于加载合约行情数据。
            suppress_year_before (int, optional): 若合约的ENDDATE年份早于该值，则自动忽略该合约。默认为当前年份。
            List[ContractRollover]: 检测到的合约切换点对象列表，每个对象包含切换前后合约的关键信息及行情数据片段。
            ValueError: 
                - 'PRODUCT'列不唯一时抛出。
                - 存在'STARTDATE'或'ENDDATE'有一个为空的行。
                - 合约重复但ENDDATE顺序不正确。
                - old_contract_tick非空但new_contract_tick为空。
                - 日期格式不正确或无法提取decade_str。
            NotImplementedError: 若未实现相关子类方法。
        注意事项:
            - 本方法依赖于self.data_tables['product_contract_start_end']的数据结构和内容。
            - 需要实现self.contract_data_loader方法用于加载合约行情数据。
            - 合约切换点的判定基于合约起止日期的连续性与唯一性。
        
        """
        if not generate_main_contract_series:
            adjust_main_contract_series = False

        if adjust_main_contract_series:
            generate_main_contract_series = True
            if strategy_selector is None:
                raise ValueError("adjust_main_contract_series为True时，必须提供strategy_selector参数")
            
        if main_contract_series_mink_bool and mink_folder_path is None:
            raise ValueError("main_contract_series_mink_bool为True时，必须提供mink_folder_path参数")
            
        all_df = self.data_tables.get('product_contract_start_end')
        if all_df is None or all_df.empty:
            raise ValueError("'product_contract_start_end'数据表为空或不存在")
        self.product_id_list = sorted(all_df['PRODUCT'].unique().tolist())
        if not self.product_id_list:
            raise ValueError("'PRODUCT'列为空")
        
        all_issues = []
        main_contract_series = []
        for product_id in tqdm(self.product_id_list, desc="Detecting rollover points"):

            df = all_df[all_df['PRODUCT'] == product_id].reset_index(drop=True)
            
            if not hasattr(self, 'rollover_points'):
                self.rollover_points = {}
            if product_id not in self.rollover_points:
                self.rollover_points[product_id] = []
            
            # 检查'STARTDATE'和'ENDDATE'是否都有数据
            mask_start_null = df['STARTDATE'].isnull()
            mask_end_null = df['ENDDATE'].isnull()
            both_null = mask_start_null & mask_end_null
            one_null = mask_start_null ^ mask_end_null

            # 删除'STARTDATE'和'ENDDATE'都为空的行
            df = df.drop(df.index[both_null])
            if one_null.any():
                raise ValueError(f"{product_id}: 存在'STARTDATE'或'ENDDATE'有一个为空的行")
            
            # 按'STARTDATE'排序
            df = df.sort_values('STARTDATE').reset_index(drop=True)

            # 将STARTDATE和ENDDATE全部转为date格式（兼容float/int/str）
            def to_date(val):
                if pd.isnull(val):
                    return pd.NaT
                if isinstance(val, (int, float)):
                    val = int(val)
                    return pd.to_datetime(str(val)).date()
                else:
                    return pd.to_datetime(val).date()

            df['STARTDATE'] = df['STARTDATE'].map(to_date)
            df['ENDDATE'] = df['ENDDATE'].map(to_date)

            # 打印所有行
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     if product_id == 'OI.CZC':
            #         print(df)

            # 遍历所有行，检查CONTRACT重复且ENDDATE顺序
            drop_indices = []
            for idx in range(len(df) - 1):
                curr_end = df.iloc[idx]['ENDDATE']
                next_start = df.iloc[idx + 1]['STARTDATE']
                if curr_end >= next_start:
                    raise ValueError(f"{product_id}: 第{idx}行的ENDDATE不在下一行的STARTDATE之前: {curr_end} >= {next_start}")
                curr_contract = windcode_to_shortened_windcode(df.iloc[idx]['CONTRACT'])
                next_contract = windcode_to_shortened_windcode(df.iloc[idx + 1]['CONTRACT'])
                if curr_contract == next_contract:
                    # 把下一行的STARTDATE改成当前行的STARTDATE，然后删除当前行
                    df.at[idx + 1, 'STARTDATE'] = df.iloc[idx]['STARTDATE']
                    drop_indices.append(idx)
            if drop_indices:
                df = df.drop(df.index[drop_indices]).reset_index(drop=True)

            def calculate_decade_str(row):
                enddate_str = str(row['ENDDATE'])
                if len(enddate_str) < 4:
                    raise ValueError(f"{product_id}: ENDDATE格式不正确，无法提取decade_str")
                contract_digits = ''.join([c for c in row['CONTRACT'] if c.isdigit()])
                if len(contract_digits) < 3 or len(contract_digits) > 4:
                    raise ValueError(f"{product_id}: CONTRACT格式不正确，无法提取decade_str")
                contract_third_last = contract_digits[-3]
                if contract_third_last == '0' and enddate_str[3] != '0':
                    return str(int(enddate_str[2]) + 1)
                else:
                    return enddate_str[2]
            
            main_contract_series_checked = []
            for idx in range(len(df)):

                new_product_bool = False
                if idx == 0 and len(df) == 1 and generate_main_contract_series:
                    new_product_bool = True

                if idx + 1 >= len(df) and not new_product_bool:
                    continue

                if not new_product_bool:

                    is_valid = True

                    this_row = df.iloc[idx]
                    next_row = df.iloc[idx + 1]

                    old_contract = this_row['CONTRACT']
                    new_contract = next_row['CONTRACT']

                    old_contract_UID = windcode_to_unique_instrument_id(old_contract, decade_str=calculate_decade_str(this_row))
                    new_contract_UID = windcode_to_unique_instrument_id(new_contract, decade_str=calculate_decade_str(next_row))

                    if idx == 0:
                        self.add_data_table("old_contract_tick", self.contract_data_loader(old_contract_UID, 'contract_dayk'))
                        self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, 'contract_dayk'))
                    else:
                        self.data_tables['old_contract_tick'] = self.data_tables['new_contract_tick']
                        self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, 'contract_dayk'))
                        
                    # 检查数据表是否为空，跳过不合法情况
                    old_tick_empty = self.data_tables["old_contract_tick"].empty
                    new_tick_empty = self.data_tables["new_contract_tick"].empty
                    if old_tick_empty and new_tick_empty:
                        is_valid = False
                    if not old_tick_empty and new_tick_empty:
                        raise ValueError(f"{product_id}: 第{idx}行: old_contract_tick非空但new_contract_tick为空，不合法")
                    
                    this_start_date = pd.to_datetime(this_row['STARTDATE'])
                    this_end_date = pd.to_datetime(this_row['ENDDATE'])
                    next_start_date = pd.to_datetime(next_row['STARTDATE'])
                    next_end_date = pd.to_datetime(next_row['ENDDATE'])

                    old_trading_days = pd.to_datetime(self.data_tables['old_contract_tick']['trading_day'])
                    new_trading_days = pd.to_datetime(self.data_tables['new_contract_tick']['trading_day'])
                    
                    # 找到小于等于this_end_date的最大日期
                    mask = pd.Series([False] * len(old_trading_days), index=old_trading_days.index)
                    if not old_trading_days.empty:
                        mask = old_trading_days == old_trading_days[old_trading_days <= this_end_date].max()
                    if mask.any():
                        old_contract_old_data = self.data_tables['old_contract_tick'][mask].iloc[[-1]]
                    else:
                        old_contract_old_data = pd.DataFrame()
                        is_valid = False
                    
                    # 找到小于等于this_end_date的最大日期
                    mask_new = pd.Series([False] * len(new_trading_days), index=new_trading_days.index)
                    if not new_trading_days.empty:
                        mask_new = new_trading_days == new_trading_days[new_trading_days <= this_end_date].max()
                    if mask_new.any():
                        new_contract_old_data = self.data_tables['new_contract_tick'][mask_new].iloc[[-1]]
                    else:
                        new_contract_old_data = pd.DataFrame()
                        is_valid = False
                    
                    rollover = ContractRollover(
                        old_contract=old_contract_UID,
                        new_contract=new_contract_UID,
                        old_contract_old_data=old_contract_old_data,
                        old_contract_new_data=pd.DataFrame(),
                        new_contract_old_data=new_contract_old_data,
                        new_contract_new_data=pd.DataFrame(),
                        old_contract_end_date=this_end_date.date(),
                        old_contract_end_datetime=None,
                        new_contract_start_date=next_start_date.date(),
                        new_contract_start_datetime=None,
                        is_valid=is_valid
                    )
                    self.rollover_points[product_id].append(rollover)

                else:
                    # new_product_bool为True的情况，表示只有一个合约
                    this_row = df.iloc[idx]
                    this_contract = this_row['CONTRACT']
                    old_contract_UID = windcode_to_unique_instrument_id(this_contract, decade_str=calculate_decade_str(this_row))
                    self.add_data_table("new_contract_tick", self.contract_data_loader(old_contract_UID, 'contract_dayk'))
                    next_start_date = pd.to_datetime(this_row['STARTDATE'])
                    next_end_date = pd.to_datetime(f"{datetime.now().year + 1}-01-01")
                    new_trading_days = pd.to_datetime(self.data_tables['new_contract_tick']['trading_day'])
                    # Dummy values for suppressing code checker warnings
                    this_start_date = next_start_date
                    this_end_date = next_end_date
                    old_trading_days = new_trading_days
                    new_contract_UID = old_contract_UID

                if generate_main_contract_series:
                    if main_contract_series_mink_bool:
                        if new_product_bool:
                            self.add_data_table("new_contract_tick", self.contract_data_loader(old_contract_UID, 'contract_mink', mink_folder_path))
                            new_trading_days = pd.to_datetime(self.data_tables['new_contract_tick']['trading_day'])
                        else:
                            if idx == 0:
                                self.add_data_table("old_contract_tick", self.contract_data_loader(old_contract_UID, 'contract_mink', mink_folder_path))
                                self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, 'contract_mink', mink_folder_path))
                            else:
                                self.data_tables['old_contract_tick'] = self.data_tables['new_contract_tick']
                                self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, 'contract_mink', mink_folder_path))
                            old_trading_days = pd.to_datetime(self.data_tables['old_contract_tick']['trading_day'])
                            new_trading_days = pd.to_datetime(self.data_tables['new_contract_tick']['trading_day'])
                    # 将第一个1之前的0也改成1，并返回是否存在这样的0
                    def mask_alter_zero_before_first_one(mask: pd.Series) -> Tuple[pd.Series, bool]:
                        mask_arr = mask.values
                        exists_zero_before_first_one = False
                        # 遍历mask_arr，找到第一个1的索引，把它前一个idx的值改为1
                        first_one_idx = None
                        for i, val in enumerate(mask_arr):
                            if val:
                                first_one_idx = i
                                break
                        if first_one_idx is not None and first_one_idx > 0:
                            exists_zero_before_first_one = True
                            mask_arr[first_one_idx - 1] = 1
                            mask = pd.Series(mask_arr, index=mask.index)
                        return mask, exists_zero_before_first_one
                    table_mask_dict = {}
                    if idx == 0 and not new_product_bool:
                        table_mask_dict['old_contract_tick'] = mask_alter_zero_before_first_one(
                            (old_trading_days >= this_start_date) & (old_trading_days <= this_end_date))
                    table_mask_dict['new_contract_tick'] = mask_alter_zero_before_first_one(
                        (new_trading_days >= next_start_date) & (new_trading_days <= next_end_date))
                    for table_name, (mask, mask_bool) in table_mask_dict.items():
                        extended_main_tick = self.data_tables[table_name][mask]
                        if not mask_bool and extended_main_tick.empty:
                            main_contract_series_checked.append(pd.DataFrame())
                            continue
                        if mask_bool and len(extended_main_tick) <= 1:
                            main_contract_series_checked.append(pd.DataFrame())
                            continue
                        # Generate main contract series
                        if mask_bool:
                            main_contract_series.append(extended_main_tick[1:])
                        else:
                            main_contract_series.append(extended_main_tick)
                        # Data Quality Checker (Volume)
                        checker_2 = DataQualityChecker(extended_main_tick,
                                                    columns=['volume'],
                                                    column_mapping={'symbol': 'unique_instrument_id', 'time': 'trade_time'})
                        checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_LONG] = DataIssueSolution.NO_ACTION
                        checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_SHORT] = DataIssueSolution.NO_ACTION
                        checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_ALL] = DataIssueSolution.NO_ACTION
                        checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_AT_END] = DataIssueSolution.NO_ACTION
                        checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_AT_START] = DataIssueSolution.NO_ACTION
                        checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_AT_VERY_START] = DataIssueSolution.NO_ACTION
                        checker_2._assign_solution_by_issue_label()
                        issues_df_2 = checker_2.issues_df
                        if issues_df_2 is not None and not issues_df_2.empty:
                            all_issues.append(issues_df_2)
                        main_tick_checked = checker_2.process_dataframe(mapping={
                            DataIssueLabel.ZERO_SEQUENCE_LONG: DataIssueSolution.FORWARD_FILL,
                            DataIssueLabel.ZERO_SEQUENCE_SHORT: DataIssueSolution.FORWARD_FILL,
                            DataIssueLabel.ZERO_SEQUENCE_ALL: DataIssueSolution.FORWARD_FILL,
                            DataIssueLabel.ZERO_SEQUENCE_AT_END: DataIssueSolution.FORWARD_FILL,
                            DataIssueLabel.ZERO_SEQUENCE_AT_START: DataIssueSolution.NO_ACTION,
                            DataIssueLabel.ZERO_SEQUENCE_AT_VERY_START: DataIssueSolution.NO_ACTION
                        })
                        # Data Quality Checker (Price)
                        checker = DataQualityChecker(main_tick_checked, 
                                                    columns=['open_price', 'highest_price', 'lowest_price', 'close_price'],
                                                    column_mapping={'symbol': 'unique_instrument_id', 'time': 'trade_time'})
                        issues_df = checker.issues_df
                        if issues_df is not None and not issues_df.empty:
                            all_issues.append(issues_df)
                        main_tick_checked = checker.process_dataframe()
                        
                        if main_tick_checked is not None and not main_tick_checked.empty:
                            if mask_bool:
                                main_contract_series_checked.append(main_tick_checked[1:])
                            else:
                                main_contract_series_checked.append(main_tick_checked)
                        else:
                            main_contract_series_checked.append(pd.DataFrame())
            
            if not hasattr(self, 'main_tick_checked') or not isinstance(self.main_tick_checked, dict):
                self.main_tick_checked = {}
            self.main_tick_checked[product_id] = main_contract_series_checked

        if generate_main_contract_series:
            self.data_tables['main_tick_issues'] = pd.concat(all_issues, ignore_index=True) if all_issues else pd.DataFrame()
            self.data_tables['main_tick'] = pd.concat(main_contract_series, ignore_index=True) if main_contract_series else pd.DataFrame()
            if adjust_main_contract_series and strategy_selector is not None:
                self.get_adjustment_factor(strategy_selector=strategy_selector, adjust_main_contract_series=True)
        return self.rollover_points
    
    def generate_main_contract_series(self, mink_folder_path: Optional[str] = None, mink_bool: bool = True) -> pd.DataFrame:
        self.detect_rollover_points(mink_folder_path=mink_folder_path, main_contract_series_mink_bool=mink_bool, generate_main_contract_series=True)
        return self.data_tables.get('main_tick', pd.DataFrame())
    
    def generate_main_contract_series_adjusted(self, strategy_selector: ProductPeriodStrategySelector, mink_folder_path: Optional[str] = None, mink_bool: bool = True) -> pd.DataFrame:
        self.detect_rollover_points(mink_folder_path=mink_folder_path, main_contract_series_mink_bool=mink_bool, generate_main_contract_series=True, 
                                    adjust_main_contract_series=True, strategy_selector=strategy_selector)
        return self.data_tables.get('main_tick_adjusted', pd.DataFrame())
    
    def contract_data_loader(self, unique_instrument_id: str, table_name: str, mink_folder_path: Optional[str] = None) -> pd.DataFrame:
        """
        根据合约的unique_instrument_id加载对应的行情数据。

        参数:
            unique_instrument_id: 合约的唯一标识符
            table_name: 数据表名称，如'old_contract_tick'或'new_contract_tick'

        返回:
            返回该合约对应的DataFrame数据

        说明:
            本方法会从self.data_tables[table_name]中筛选出unique_instrument_id等于指定值的所有数据行。
            若数据表不存在或为空，则抛出异常。
        """
        if table_name == 'contract_dayk':
            df = self.data_tables.get(table_name)
            if df is None or df.empty:
                raise ValueError(f"'{table_name}'数据表为空或不存在")
            return df[df['unique_instrument_id'] == unique_instrument_id].sort_values('trading_day')
        elif table_name == 'contract_mink':
            file_name = unique_instrument_id + '.csv'
            if mink_folder_path is None:
                raise ValueError("mink_folder_path参数不能为空")
            file_path = os.path.join(mink_folder_path, file_name)
            if not os.path.exists(file_path):
                # 返回一个包含 old_contract_tick 所需列的空 DataFrame
                required_cols = self.REQUIRED_COLUMNS['old_contract_tick']
                # print(f"警告: 文件 {file_path} 不存在，返回空DataFrame。")
                return pd.DataFrame(columns=required_cols)
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"未知的table_name: {table_name}")
    
    def get_adjustment_factor(self, 
                              product_id: Optional[str] = None,
                              strategy_selector: Optional[ProductPeriodStrategySelector] = None,
                              adjust_main_contract_series: bool = False) -> pd.DataFrame:
        """
        计算并返回每个合约切换点的复权因子（adjustment_factor），并记录所用的AdjustmentStrategy信息。

        参数:
            strategy_selector: ProductPeriodStrategySelector对象，包含默认的AdjustmentStrategy。

        返回:
            pd.DataFrame: 每个切换点的复权因子及相关信息，包含product_id、old/new unique_instrument_id、切换日期和adjustment。
        """

        if strategy_selector is None:
            strategy_selector = ProductPeriodStrategySelector(default_strategy={
                "AdjustmentStrategy": PercentageAdjustmentStrategy(
                    old_price_field='close_price', new_price_field='close_price', 
                    new_price_old_data_bool=True, use_window=False
                )
            })

        if not hasattr(strategy_selector, "default_strategy") or "AdjustmentStrategy" not in strategy_selector.default_strategy:
            raise ValueError("ProductPeriodStrategySelector.default_strategy 中缺少 'AdjustmentStrategy' 键")

        strategy = strategy_selector.default_strategy["AdjustmentStrategy"]

        if product_id is None:
            df = self.data_tables.get('product_contract_start_end')
            if df is None or df.empty:
                raise ValueError("'product_contract_start_end'数据表为空或不存在")
            product_id_list = sorted(df['PRODUCT'].unique().tolist())
            if not product_id_list:
                raise ValueError("'PRODUCT'列为空")
        else:
            product_id_list = [product_id]

        for product_id in tqdm(product_id_list, desc="Calculating adjustment factors"):

            if product_id is None:
                continue

            if hasattr(self, 'main_tick_checked'):
                if isinstance(self.main_tick_checked, dict):
                    main_tick_checked = self.main_tick_checked.get(product_id, [])
                else:
                    raise ValueError("self.main_tick_checked应为字典类型，键为product_id")
            else:
                main_tick_checked = []

            # 获取rollover_points
            if not hasattr(self, 'rollover_points'):
                self.detect_rollover_points()
            elif product_id not in self.rollover_points:
                raise ValueError(f"{product_id}: 未检测到rollover_points")
            rollover_points = self.rollover_points[product_id]

            if adjust_main_contract_series and hasattr(self, 'main_tick_checked'):
                if len(rollover_points) + 1 == len(main_tick_checked):
                    pass
                else:
                    print(len(rollover_points), len(main_tick_checked))
                    raise ValueError(f"{product_id}: 调整主合约序列时，main_tick_checked的数量应比rollover_points多1")

            results = []
            for idx in range(len(rollover_points)):
                rollover = rollover_points[idx]
                strategy = strategy_selector.get_strategy(product_id, pd.to_datetime(str(rollover.new_contract_start_date)))['AdjustmentStrategy']

                is_valid, validity_status = strategy.is_valid(rollover)
                if is_valid:
                    adjustment_mul, adjustment_add = strategy.calculate_adjustment(rollover)
                    if strategy.adjustment_operation == AdjustmentOperation.ADDITIVE:
                        adjustment = adjustment_add
                    elif strategy.adjustment_operation == AdjustmentOperation.MULTIPLICATIVE:
                        adjustment = adjustment_mul
                    else:
                        raise ValueError(f"未知的调整操作类型: {strategy.adjustment_operation}")
                else:
                    adjustment = None
                
                adjustment_new = strategy.apply_adjustment_to_results(adjustment, results)

                results.append({
                    'product_id': product_id,
                    'old_unique_instrument_id': rollover.old_contract,
                    'new_unique_instrument_id': rollover.new_contract,
                    'rollover_date': rollover.new_contract_start_date,
                    'adjustment_strategy': strategy.__class__.__name__,
                    'adjustment_operation': strategy.adjustment_operation.name,
                    'is_valid': is_valid,
                    'validity_status': validity_status,
                    'val_adjust_old': adjustment,
                    'val_adjust_new': adjustment_new
                })
                
                if adjust_main_contract_series and main_tick_checked:
                    if adjustment is None:
                        continue
                    # Ensure all 'trade_time' are <= rollover.new_contract_start_datetime
                    if 'trade_time' in main_tick_checked[idx].columns and rollover.new_contract_start_datetime is not None:
                        trade_times = pd.to_datetime(main_tick_checked[idx]['trade_time'])
                        rollover_time = pd.to_datetime(rollover.new_contract_start_datetime)
                        if not trade_times.le(rollover_time).all():
                            raise ValueError(
                                f"{product_id}: main_tick_checked[{idx}]的'trade_time'列存在大于rollover.new_contract_start_datetime的值"
                            )
                    if idx == len(rollover_points) - 1:
                        pass
                    elif strategy.adjustment_direction == AdjustmentDirection.ADJUST_OLD:
                        if main_tick_checked[idx].empty:
                            pass
                        elif strategy.adjustment_operation == AdjustmentOperation.ADDITIVE:
                            main_tick_checked[idx][['open_price', 'highest_price', 'lowest_price', 'close_price']] += adjustment
                        elif strategy.adjustment_operation == AdjustmentOperation.MULTIPLICATIVE:
                            main_tick_checked[idx][['open_price', 'highest_price', 'lowest_price', 'close_price']] *= adjustment
                        else:
                            raise ValueError(f"未知的调整操作类型: {strategy.adjustment_operation}")
                    elif strategy.adjustment_direction == AdjustmentDirection.ADJUST_NEW:
                        if main_tick_checked[idx+1].empty:
                            pass
                        elif strategy.adjustment_operation == AdjustmentOperation.ADDITIVE:
                            main_tick_checked[idx+1][['open_price', 'highest_price', 'lowest_price', 'close_price']] -= adjustment_new
                        elif strategy.adjustment_operation == AdjustmentOperation.MULTIPLICATIVE:
                            main_tick_checked[idx+1][['open_price', 'highest_price', 'lowest_price', 'close_price']] /= adjustment_new
                        else:
                            raise ValueError(f"未知的调整操作类型: {strategy.adjustment_operation}")
                    else:
                        raise ValueError(f"未知的调整方向: {strategy.adjustment_direction}")

            if 'adjustment_factors' in self.data_tables and not self.data_tables['adjustment_factors'].empty:
                self.data_tables['adjustment_factors'] = pd.concat(
                    [self.data_tables['adjustment_factors'], pd.DataFrame(results).dropna(axis=1, how='all')],
                    ignore_index=True)
            else:
                self.data_tables['adjustment_factors'] = pd.DataFrame(results)

            if adjust_main_contract_series and main_tick_checked:
                main_tick_checked = [df for df in main_tick_checked if not df.empty]
                if main_tick_checked:
                    if 'main_tick_adjusted' in self.data_tables and not self.data_tables['main_tick_adjusted'].empty:
                        self.data_tables['main_tick_adjusted'] = pd.concat(
                            [self.data_tables['main_tick_adjusted'], pd.concat(main_tick_checked, ignore_index=True)],
                            ignore_index=True)
                    else:
                        self.data_tables['main_tick_adjusted'] = pd.concat(main_tick_checked, ignore_index=True)
                else:
                    self.data_tables['main_tick_adjusted'] = pd.DataFrame()

        return self.data_tables['adjustment_factors']
    
class MultipleFuturesProcessor(FuturesProcessorBase):
    @property
    def EXPECTED_TABLE_NAMES(self) -> List[str]:
        return ['main_tick_path', 'main_tick', 'return_tick', 'adjustment_factors', 'main_tick_issues']
    

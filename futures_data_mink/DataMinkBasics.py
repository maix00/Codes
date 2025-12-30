"""
DataMinkBasics.py

本模块为期货数据处理基础工具，提供期货合约唯一标识符（unique_instrument_id）、产品代码（product_id）与 Wind 代码（windcode）之间的相互转换函数。

主要内容包括：
- 交易所代码映射表（exchange_map）
- 当前年份十位数字的获取方法
- unique_instrument_id 与 product_id、windcode 之间的转换函数

函数说明：
- unique_instrument_id_to_product_id(unique_instrument_id: str) -> str
    将 unique_instrument_id 转换为产品代码（product_id），若以 'F' 结尾则加 '_F' 后缀。
- unique_instrument_id_to_windcode(unique_instrument_id: str) -> str
    将 unique_instrument_id 转换为 Wind 代码（windcode），格式为“产品代码+月份.交易所代码”。
- unique_instrument_id_to_windcode_simp(unique_instrument_id: str) -> str
    将 unique_instrument_id 转换为简化版 Wind 代码，月份部分去掉首位字符。
- windcode_to_unique_instrument_id(windcode: str, decade_str: str = year_tens) -> str
    将 Wind 代码转换为 unique_instrument_id，自动补全年份十位数字。

注意事项：
- 所有函数假定 unique_instrument_id 格式为“交易所|F|产品代码|月份”。
- 输入格式不正确时，函数会抛出 ValueError 异常。
"""

from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from ContractRollover import ContractRollover
from FuturesProcessor import FuturesProcessorBase, ContractRollover
from StrategySelector import ProductPeriodStrategySelector
from DataQualityChecker import DataQualityChecker, DataIssueLabel, DataIssueSolution
from AdjustmentStrategy import ValidityStatus, AdjustmentStrategy, PercentageAdjustmentStrategy, \
    AdjustmentDirection, AdjustmentOperation
import pandas as pd
from tqdm import tqdm
import pickle

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

    # 取出第一个竖线前的字母（交易所代码），并用exchange_map转换
    exchange = exchange_map.get(parts[0], parts[0])
    product_id = f"{product_id}.{exchange}"

    return product_id

def unique_instrument_id_to_windcode(unique_instrument_id: str) -> str:
    parts = unique_instrument_id.split('|')
    if len(parts) < 4:
        raise ValueError(f"unique_instrument_id格式不正确: {unique_instrument_id}")
    exchange = exchange_map.get(parts[0], parts[0])
    return f"{parts[2]}{parts[3]}.{exchange}"

def unique_instrument_id_to_windcode_simp(unique_instrument_id: str) -> str:
    parts = unique_instrument_id.split('|')
    if len(parts) < 4:
        raise ValueError(f"unique_instrument_id格式不正确: {unique_instrument_id}")
    exchange = exchange_map.get(parts[0], parts[0])
    return f"{parts[2]}{parts[3][1:]}.{exchange}"

def windcode_to_shortened_windcode(windcode: str) -> str:
    # 去掉第一个数字，如果数字有四位的话，否则三位则不动
    # 例如: RB2210F.DCE -> RB210F.DCE, RB210.DCE -> RB210.DCE
    first_digit_idx = -1
    last_digit_idx = -1
    for i in range(len(windcode)):
        if i > 0 and not windcode[i - 1].isdigit() and windcode[i].isdigit():
            first_digit_idx = i
        if i < len(windcode) - 1 and windcode[i].isdigit() and not windcode[i + 1].isdigit():
            last_digit_idx = i
            break
    if first_digit_idx == -1 or last_digit_idx == -1:
        raise ValueError(f"windcode格式不正确: {windcode}")
    if last_digit_idx - first_digit_idx == 3:
        return windcode[:first_digit_idx] + windcode[first_digit_idx + 1:]
    elif last_digit_idx - first_digit_idx == 2:
        return windcode
    else:
        raise ValueError(f"windcode格式不正确: {windcode}")

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
        return ['product_contract_start_end', 'contract_dayk', 'contract_mink', 
                'main_contract_series', 'rollover_adjustments', 'main_contract_series_adjusted']
    
    @property
    def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
        return {
            'product_contract_start_end': ['PRODUCT', 'CONTRACT', 'STARTDATE', 'ENDDATE'],
            'contract_dayk': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'contract_mink': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id']
        }

    @property
    def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
        return {
            'product_contract_start_end': ['PRODUCT', 'CONTRACT', 'STARTDATE', 'ENDDATE'],
            'contract_dayk': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
            'contract_mink': ['trading_day', 'trade_time', 'open_price', 'highest_price', 'lowest_price', 'close_price', 'volume', 'unique_instrument_id'],
        }
    
    def detect_rollover_points(self, rollover_points_cache_path: Optional[str] = None) -> Dict[str, Dict[str, ContractRollover]]:
        """
        检测期货合约的展期点（Rollover Points）。
        
        本方法根据合约起止日期表（'product_contract_start_end'），自动检测每个期货品种的合约展期点，并为每个展期点加载切换前后两个合约的行情数据片段。方法包含对数据完整性和合理性的多重校验，确保展期点的准确性和数据的有效性。
        
        返回值:
            Dict[str, Dict[str, ContractRollover]]: 返回一个字典，键为产品ID，值为该产品下所有展期点（以切换日期为键）的ContractRollover对象。每个对象包含切换前后合约的唯一标识、行情数据片段、切换日期等关键信息。
        
        异常:
            ValueError:
            - 'product_contract_start_end'数据表为空或不存在时抛出。
            - 'PRODUCT'列为空时抛出。
            - 存在'STARTDATE'或'ENDDATE'有一个为空的行时抛出。
            - 合约重复但ENDDATE顺序不正确时抛出。
            - 行情数据缺失或格式不正确时抛出。
        
        注意事项:
            - 本方法依赖于self.data_tables['product_contract_start_end']的数据结构和内容。
            - 需要实现self.contract_data_loader方法用于加载合约行情数据。
            - 展期点的判定基于合约起止日期的连续性、唯一性及行情数据的有效性。
            - 若数据表中存在异常或不一致的数据，将抛出相应异常以提示用户修正数据。
        """

        # 获取所有产品的合约起止日期表
        all_df = self.data_tables.get('product_contract_start_end')
        if all_df is None or all_df.empty:
            raise ValueError("'product_contract_start_end'数据表为空或不存在")
        
        # 检查是否已经有product_id_list属性
        if not self.product_id_list:
            self.product_id_list = sorted(all_df['PRODUCT'].unique().tolist())
        if not self.product_id_list:
            raise ValueError("'PRODUCT'列为空")
        
        # 支持缓存rollover_points到本地文件
        if rollover_points_cache_path is not None:
            self.rollover_points_cache_path = rollover_points_cache_path
        if self.rollover_points_cache_path and os.path.exists(self.rollover_points_cache_path):
            with open(self.rollover_points_cache_path, 'rb') as f:
                self.rollover_points = pickle.load(f)
        
        # 遍历每个产品，检测展期点
        for product_id in tqdm(self.product_id_list, desc="Detecting rollover points"):

            # 初始化该产品的rollover_points字典
            if product_id not in self.rollover_points:
                self.rollover_points[product_id] = {}

            # 获取该产品的合约起止日期数据
            df = all_df[all_df['PRODUCT'] == product_id].reset_index(drop=True)
            
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

            # # 打印所有行
            # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            #     print(df)

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
            
            # 如果已经有缓存的展期点，且数量接近，则进行对比，避免重复计算
            if product_id in self.rollover_points and self.rollover_points[product_id]:
                if abs(len(df) - len(self.rollover_points[product_id])) < 3:
                    existing_dates = set(self.rollover_points[product_id].keys())
                    current_dates = set(str(df.iloc[i + 1]['STARTDATE']) for i in range(len(df) - 1))
                    # print(product_id, existing_dates == current_dates)
                    if existing_dates == current_dates:
                        continue

            # 遍历所有行，检测展期点
            # 缓存合约行情数据，避免重复加载
            contract_data_cache = {}
            for idx in range(len(df)):

                if idx + 1 >= len(df):
                    continue

                is_valid = True

                this_row = df.iloc[idx]
                next_row = df.iloc[idx + 1]
                
                this_start_date = pd.to_datetime(this_row['STARTDATE'])
                this_end_date = pd.to_datetime(this_row['ENDDATE'])
                next_start_date = pd.to_datetime(next_row['STARTDATE'])
                next_end_date = pd.to_datetime(next_row['ENDDATE'])

                if product_id in self.rollover_points and next_start_date.date() in self.rollover_points[product_id]:
                    continue

                old_contract = this_row['CONTRACT']
                new_contract = next_row['CONTRACT']

                old_contract_UID = windcode_to_unique_instrument_id(old_contract, decade_str=self.calculate_decade_str(this_row))
                new_contract_UID = windcode_to_unique_instrument_id(new_contract, decade_str=self.calculate_decade_str(next_row))

                # 使用缓存避免重复加载行情数据
                if old_contract_UID not in contract_data_cache:
                    contract_data_cache[old_contract_UID] = self.contract_data_loader(old_contract_UID, 'contract_dayk')
                if new_contract_UID not in contract_data_cache:
                    contract_data_cache[new_contract_UID] = self.contract_data_loader(new_contract_UID, 'contract_dayk')

                # 检查数据表是否为空
                new_contract_empty = contract_data_cache[new_contract_UID].empty
                if new_contract_empty:
                    is_valid = False

                old_trading_days = pd.to_datetime(contract_data_cache[old_contract_UID]['trading_day'])
                new_trading_days = pd.to_datetime(contract_data_cache[new_contract_UID]['trading_day'])
                
                # 找到小于等于this_end_date的最大日期
                mask = pd.Series([False] * len(old_trading_days), index=old_trading_days.index)
                if not old_trading_days.empty:
                    mask = old_trading_days == old_trading_days[old_trading_days <= this_end_date].max()
                if mask.any():
                    old_contract_old_data = contract_data_cache[old_contract_UID][mask].iloc[[-1]]
                    if old_contract_old_data.empty:
                        is_valid = False
                else:
                    old_contract_old_data = pd.DataFrame()
                    is_valid = False

                # 找到小于等于this_end_date的最大日期
                mask_new = pd.Series([False] * len(new_trading_days), index=new_trading_days.index)
                if not new_trading_days.empty:
                    mask_new = new_trading_days == new_trading_days[new_trading_days <= this_end_date].max()
                if mask_new.any():
                    new_contract_old_data = contract_data_cache[new_contract_UID][mask_new].iloc[[-1]]
                    if new_contract_old_data.empty:
                        is_valid = False
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
                    old_contract_start_date=this_start_date.date(),
                    old_contract_start_datetime=None,
                    old_contract_end_date=this_end_date.date(),
                    old_contract_end_datetime=None,
                    new_contract_start_date=next_start_date.date(),
                    new_contract_start_datetime=None,
                    new_contract_end_date=next_end_date.date(),
                    new_contract_end_datetime=None,
                    is_valid=is_valid
                )
                self.rollover_points[product_id][str(next_start_date.date())] = rollover
        
        # 如果有缓存路径，则保存rollover_points到本地文件
        if self.rollover_points_cache_path is not None:
            with open(self.rollover_points_cache_path, 'wb') as f:
                pickle.dump(self.rollover_points, f)

        # 删除self._contract_dayk_grouped以释放内存
        if hasattr(self, '_contract_dayk_grouped'):
            del self._contract_dayk_grouped

        return self.rollover_points
    
    @staticmethod
    def calculate_decade_str(row):
        """
        计算合约十年期字符串（decade_str）。
        
        本方法根据传入数据行中的 'ENDDATE' 和 'CONTRACT' 字段，自动提取并计算期货合约的十年期字符串（decade_str），用于唯一标识合约所属的年代。
        
        参数:
            row (pd.Series): 包含至少 'ENDDATE'、'PRODUCT' 和 'CONTRACT' 字段的数据行。
        
        返回值:
            str: 计算得到的十年期字符串。
        
        异常:
            ValueError:
            - 当 'ENDDATE' 字符串长度小于4时抛出。
            - 当 'CONTRACT' 中的数字长度不在3到4之间时抛出。
        
        注意事项:
            - 本方法假设 'ENDDATE' 字段为8位日期字符串（如20231231），'CONTRACT' 字段包含合约代码及年月。
            - decade_str 的判断逻辑：若 'CONTRACT' 数字部分倒数第三位为 '0' 且 'ENDDATE' 第四位不为 '0'，则返回 'ENDDATE' 的第三位加1，否则返回 'ENDDATE' 的第三位。
        """
        enddate_str = str(row['ENDDATE'])
        product_id = row['PRODUCT']
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
    
    def contract_data_loader(self, unique_instrument_id: str, table_name: str, mink_folder_path: Optional[str] = None) -> pd.DataFrame:
        """
        加载指定合约的行情数据。

        本方法根据合约的唯一标识符（unique_instrument_id）和数据表名称（table_name），返回对应的行情数据DataFrame。支持日线（contract_dayk）和分钟线（contract_mink）两类行情数据。

        参数:
            unique_instrument_id (str): 合约唯一标识符。
            table_name (str): 数据表名称，支持'contract_dayk'或'contract_mink'。
            mink_folder_path (Optional[str]): 分钟线数据文件夹路径，仅在table_name为'contract_mink'时需要指定。

        返回值:
            pd.DataFrame: 返回对应合约的行情数据，若数据不存在则返回包含所需字段的空DataFrame。

        异常:
            ValueError:
            - 数据表为空或不存在时抛出。
            - mink_folder_path未指定或文件不存在时抛出。
            - table_name不支持时抛出。

        注意事项:
            - 日线数据直接从self.data_tables['contract_dayk']按unique_instrument_id筛选。
            - 分钟线数据从指定文件夹按unique_instrument_id加载csv文件。
            - 若数据不存在，返回结构一致的空DataFrame，便于后续处理。
        """
        if table_name == 'contract_dayk':
            df = self.data_tables.get(table_name)
            if not hasattr(self, '_contract_dayk_grouped'):
                df = self.data_tables.get(table_name)
                if df is None or df.empty:
                    raise ValueError(f"'{table_name}'数据表为空或不存在")
                self._contract_dayk_grouped = df.groupby('unique_instrument_id')
            if unique_instrument_id not in self._contract_dayk_grouped.groups:
                # 返回一个包含 contract_dayk 所需列的空 DataFrame
                required_cols = self.REQUIRED_COLUMNS['contract_dayk']
                return pd.DataFrame(columns=required_cols)
            return self._contract_dayk_grouped.get_group(unique_instrument_id).sort_values('trading_day')
        elif table_name == 'contract_mink':
            file_name = unique_instrument_id + '.parquet'
            if mink_folder_path is None:
                raise ValueError("mink_folder_path参数不能为空")
            file_path = os.path.join(mink_folder_path, file_name)
            if not os.path.exists(file_path):
                # 返回一个包含 contract_mink 所需列的空 DataFrame
                required_cols = self.REQUIRED_COLUMNS['contract_mink']
                # print(f"警告: 文件 {file_path} 不存在，返回空DataFrame。")
                return pd.DataFrame(columns=required_cols)
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"未知的table_name: {table_name}")
    
    @staticmethod
    def dataframe_loader(data_path: str, column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
        """
        加载指定路径的数据文件，支持CSV和Parquet格式。

        参数:
            data_path (str): 数据文件的完整路径。

        返回值:
            pd.DataFrame: 返回加载的数据DataFrame。

        异常:
            ValueError:
            - 文件格式不支持时抛出。
            - 文件不存在时抛出。
        """

        if not os.path.exists(data_path):
            raise ValueError(f"文件不存在: {data_path}")
        _, ext = os.path.splitext(data_path)
        if ext.lower() == '.csv':
            df = pd.read_csv(data_path)
        elif ext.lower() == '.parquet':
            df = pd.read_parquet(data_path)
        else:
            raise ValueError(f"不支持的文件格式: {ext}")
        
        if column_mapping:
            df = df.rename(columns=column_mapping)
        
        return df

    def calculate_adjustment(self, 
                             product_id_list: Optional[List[str]] = None,
                             strategy_selector: Optional[ProductPeriodStrategySelector] = None,
                             adjustment_direction: AdjustmentDirection = AdjustmentDirection.ADJUST_OLD,
                             adjustment_backward_datetime: Optional[datetime] = None,
                             rollover_adjustments_cache_path: Optional[str] = None,) -> pd.DataFrame:
        """
        计算期货合约展期时的价格调整因子（Adjustment Factors）。

        本方法针对每个产品ID，结合指定的调整策略（AdjustmentStrategy）和合约展期点（rollover_points），计算合约切换时的价格调整因子。支持加法和乘法两种调整方式，并对每个展期点的有效性进行校验。所有结果以DataFrame形式返回，并存储于self.data_tables['rollover_adjustments']。

        参数:
            product_id_list (Optional[List[str]]): 指定需要计算的产品ID列表，若为None则对所有产品进行计算。
            strategy_selector (Optional[ProductPeriodStrategySelector]): 产品-周期调整策略选择器，若未指定则使用默认策略。
            rollover_adjustments_cache_path (Optional[str]): 调整因子结果缓存文件路径，若指定则支持结果缓存。

        返回值:
            pd.DataFrame: 包含所有产品合约展期点的调整因子结果表。每行包括产品ID、切换前后合约ID、展期日期、调整策略、调整操作类型、有效性状态、调整因子等信息。

        异常:
            ValueError:
            - 'product_contract_start_end'数据表为空或不存在时抛出。
            - 'PRODUCT'列为空时抛出。
            - ProductPeriodStrategySelector.default_strategy中缺少'AdjustmentStrategy'键时抛出。
            - 未检测到指定产品的rollover_points时抛出。
            - 调整操作类型未知时抛出。

        注意事项:
            - 本方法依赖于self.data_tables['product_contract_start_end']和self.rollover_points的数据结构。
            - 若未检测到rollover_points，将自动调用self.detect_rollover_points()进行检测。
            - 支持自定义调整策略，需实现相关接口（如is_valid、calculate_adjustment、apply_adjustment_to_results等）。
            - 结果表会自动合并至已有的rollover_adjustments表，避免重复计算。
        """

        # 校验adjustment_direction和adjustment_backward_datetime的组合
        if adjustment_direction == AdjustmentDirection.ADJUST_OLD:
            assert adjustment_backward_datetime is None, "当adjustment_direction为ADJUST_OLD时，adjustment_backward_datetime必须为None"

        # 设置默认的strategy_selector
        if strategy_selector is not None:
            self.strategy_selector = strategy_selector
        if self.strategy_selector is None:
            self.strategy_selector = ProductPeriodStrategySelector(default_strategy={
                "AdjustmentStrategy": PercentageAdjustmentStrategy(
                old_price_field='close_price', new_price_field='close_price', 
                new_price_old_data_bool=True, use_window=False,
                description="prev_day_new_close_over_prev_day_old_close",
                adjustment_direction = adjustment_direction,
                adjustment_backward_datetime = adjustment_backward_datetime
                )
            })
        
        # 获取默认的调整策略
        # 检查AdjustmentStrategy是否存在
        if "AdjustmentStrategy" not in self.strategy_selector.default_strategy:
            raise ValueError("ProductPeriodStrategySelector.default_strategy 中缺少 'AdjustmentStrategy' 键")
        strategy = self.strategy_selector.default_strategy["AdjustmentStrategy"]
        # 设置调整策略的方向和回溯时间
        assert hasattr(strategy, 'adjustment_direction')
        strategy.adjustment_direction = adjustment_direction
        assert hasattr(strategy, 'adjustment_backward_datetime')
        strategy.adjustment_backward_datetime = adjustment_backward_datetime
        
        # 获取产品ID列表
        if product_id_list is None:
            # 检查是否已经有product_id_list属性
            if not self.product_id_list:
                # 获取所有产品的合约起止日期表
                all_df = self.data_tables.get('product_contract_start_end')
                if all_df is None or all_df.empty:
                    raise ValueError("'product_contract_start_end'数据表为空或不存在")
                self.product_id_list = sorted(all_df['PRODUCT'].unique().tolist())
            product_id_list = self.product_id_list
        if not product_id_list:
            raise ValueError("'PRODUCT'列为空")
        
        # 初始化rollover_adjustments_grouped_new字典
        if not hasattr(self, '_rollover_adjustments_grouped_new'):
            self._rollover_adjustments_grouped_new = {}
        
        # 如果有缓存路径且缓存文件存在，则尝试加载rollover_adjustments
        if rollover_adjustments_cache_path is not None:
            self.rollover_adjustments_cache_path = rollover_adjustments_cache_path
        cache_path = getattr(self, 'rollover_adjustments_cache_path', None)
        if cache_path and os.path.exists(cache_path):
            self.add_data_table('rollover_adjustments', pd.read_csv(cache_path))
            self._rollover_adjustments_grouped = self.data_tables['rollover_adjustments'].groupby('product_id')
            self._rollover_adjustments_grouped_new = {k: v for k, v in self._rollover_adjustments_grouped}
                
        for product_id in tqdm(product_id_list, desc="Calculating adjustment factors"):

            if product_id is None:
                continue

            # 获取rollover_points
            if not self.rollover_points:
                self.detect_rollover_points()
            elif product_id not in self.rollover_points:
                raise ValueError(f"{product_id}: 未检测到rollover_points")
            rollover_keys = sorted(self.rollover_points[product_id].keys())

            if hasattr(self, '_rollover_adjustments_grouped') and product_id in self._rollover_adjustments_grouped.groups:
                self._rollover_adjustments_grouped_2 = self._rollover_adjustments_grouped.get_group(product_id).groupby('new_contract_start_date')

            results = []
            is_valid = False
            renew_flag = False
            reference_time = None
            # 按照rollover_points的key（日期）从小到大排序遍历
            for idx, rollover_date in enumerate(rollover_keys):
                rollover = self.rollover_points[product_id][rollover_date]
                strategy = self.strategy_selector.get_strategy(product_id, pd.to_datetime(str(rollover_date)))['AdjustmentStrategy']

                prev_is_valid = is_valid
                is_valid, validity_status = strategy.is_valid(rollover)

                if prev_is_valid and not is_valid:
                    for res in results:
                        if res['is_valid']:
                            res['is_valid'] = False
                            res['validity_status'] = ValidityStatus.LATER_INSUFFICIENT_DATA
                    prev_is_valid = False

                # 跳过已存在的相同product_id、adjustment_strategy、adjustment_operation、description、rollover_date的行
                # 判断符合条件的行的个数，超过1个则比较val_adjust的值，如果相同则只保留一个
                if hasattr(self, '_rollover_adjustments_grouped_2') and str(rollover_date) in self._rollover_adjustments_grouped_2.groups:
                    existing_group = self._rollover_adjustments_grouped_2.get_group(str(rollover_date))
                    # 查找is_valid为True的行
                    mask = ((existing_group['adjustment_strategy'] == strategy.__class__.__name__)
                        & (existing_group['adjustment_operation'] == strategy.adjustment_operation.name)
                        & (existing_group['description'] == strategy.description)
                        & (existing_group['is_valid'] == True))
                    filtered_group = existing_group[mask]
                    # 如果is_valid为False的行存在，且当前is_valid为True，则设置renew_flag为True
                    mask_invalid = ((existing_group['adjustment_strategy'] == strategy.__class__.__name__)
                        & (existing_group['adjustment_operation'] == strategy.adjustment_operation.name)
                        & (existing_group['description'] == strategy.description)
                        & (existing_group['is_valid'] == False))
                    filtered_group_invalid = existing_group[mask_invalid]
                    if len(filtered_group_invalid) >= 1 and len(filtered_group) == 0 and is_valid:
                        renew_flag = True
                    # 如果找到了符合条件的行，则直接使用该行的数据
                    if len(filtered_group) >= 1:
                        if len(filtered_group) > 1 and len(filtered_group['val_adjust'].dropna().unique()) > 1:
                            raise ValueError(f"{product_id} {rollover_date}: 存在多个不同val_adjust的重复行，无法决定保留哪一个")
                        if renew_flag:
                            adjustment = filtered_group.iloc[0]['val_adjust']
                            strategy.apply_adjustment_to_results(adjustment, results)
                            row = filtered_group.iloc[0]
                            row_dict = row.to_dict()
                            row_dict['val_adjust_old'] = adjustment
                            results.append(row_dict)
                            continue
                        else:
                            results.append(filtered_group.iloc[0].to_dict())
                            continue

                if is_valid:
                    adjustment_mul, adjustment_add = strategy.calculate_adjustment(rollover)
                    if strategy.adjustment_operation == AdjustmentOperation.ADDITIVE:
                        adjustment = adjustment_add
                    elif strategy.adjustment_operation == AdjustmentOperation.MULTIPLICATIVE:
                        adjustment = adjustment_mul
                    else:
                        raise ValueError(f"未知的调整操作类型: {strategy.adjustment_operation}")
                    if strategy.adjustment_direction == AdjustmentDirection.ADJUST_NEW:
                        if prev_is_valid and not is_valid:
                            reference_time = None
                        if not prev_is_valid and is_valid and rollover.old_contract_start_date \
                            and (strategy.adjustment_backward_datetime is None 
                                 or (strategy.adjustment_backward_datetime is not None 
                                     and rollover.old_contract_start_date 
                                     and rollover.old_contract_start_date >= strategy.adjustment_backward_datetime.date())) \
                            and reference_time is None:
                            reference_time = rollover.old_contract_start_date
                        elif is_valid and rollover.old_contract_end_date and strategy.adjustment_backward_datetime is not None \
                            and rollover.old_contract_end_date >= strategy.adjustment_backward_datetime.date() \
                            and reference_time is None:
                            reference_time = strategy.adjustment_backward_datetime
                        adjustment_new = strategy.apply_adjustment_to_results(adjustment, results, adjustment_backward_bool=True)
                    else:
                        adjustment_new = None
                else:
                    adjustment = None
                    adjustment_new = None
                    reference_time = None

                results.append({
                    'product_id': product_id,
                    'old_unique_instrument_id': rollover.old_contract,
                    'new_unique_instrument_id': rollover.new_contract,
                    'old_contract_start_date': str(rollover.old_contract_start_date),
                    'old_contract_end_date': str(rollover.old_contract_end_date),
                    'new_contract_start_date': str(rollover.new_contract_start_date),
                    'new_contract_end_date': str(rollover.new_contract_end_date),
                    'is_valid': is_valid,
                    'validity_status': validity_status,
                    'val_adjust': adjustment,
                    'val_adjust_old': adjustment,
                    'val_adjust_new': adjustment_new,
                    'reference_time': reference_time,
                    'adjustment_strategy': strategy.__class__.__name__,
                    'adjustment_operation': strategy.adjustment_operation.name,
                    'description': strategy.description,
                })

            self._rollover_adjustments_grouped_new[product_id] = pd.DataFrame(results).dropna(axis=1, how='all')

        # 合并所有product_id的结果到self.data_tables['rollover_adjustments']
        if hasattr(self, '_rollover_adjustments_grouped_new'):
            all_result_df = pd.concat(self._rollover_adjustments_grouped_new.values(), ignore_index=True)
            self.data_tables['rollover_adjustments'] = all_result_df
        else:
            self.data_tables['rollover_adjustments'] = pd.DataFrame()

        # 如果有缓存路径，则保存rollover_adjustments到本地文件
        if cache_path is not None:
            self.data_tables['rollover_adjustments'].to_csv(cache_path, index=False)

        # 删除self._rollover_adjustments_grouped以释放内存
        if hasattr(self, '_rollover_adjustments_grouped'):
            del self._rollover_adjustments_grouped
        if hasattr(self, '_rollover_adjustments_grouped_new'):
            del self._rollover_adjustments_grouped_new
        if hasattr(self, '_rollover_adjustments_grouped_2'):
            del self._rollover_adjustments_grouped_2

        return self.data_tables['rollover_adjustments']
    
    def generate_main_contract_series(self, source_data_label: str = 'dayk', 
                                      source_data_folder_UID_path: Optional[str] = None,
                                      add_adjust_col_bool: bool = False,
                                      save_path: Optional[str] = None, issues_save_path: Optional[str] = None,
                                      save_ext: str = 'parquet', update_mode: bool = False, save_per_product: bool = False) -> pd.DataFrame:
        '''
        生成主力合约序列（Main Contract Series）。
       
        本方法针对每个产品ID，依据合约展期点（rollover_points）和指定的数据源，拼接生成主力合约的连续行情序列。可选地支持添加价格调整因子（如前复权/后复权），并对数据质量进行检测和修正。最终结果以DataFrame形式返回，包含所有产品的主力合约行情数据。
        
        参数:
            source_data_label (str): 指定行情数据类型标签（如'dayk'），用于加载合约行情数据。
            source_data_folder_UID_path (Optional[str]): 指定行情数据文件夹路径，若为None则使用默认路径。
            add_adjust_col_bool (bool): 是否添加价格调整因子列（adjustment_mul, adjustment_add），用于主力合约切换时的价格连续性调整。
        
        返回值:
            pd.DataFrame: 返回所有产品主力合约的拼接行情数据表。每行包括产品ID、主力合约ID、交易日、行情数据、（可选）调整因子等信息。
        
        异常:
            ValueError:
                - 'product_contract_start_end'数据表为空或不存在时抛出。
                - 'PRODUCT'列为空时抛出。
                - 未检测到指定产品的rollover_points时抛出。
                - 调整因子记录缺失或存在多条匹配时抛出。
                - 数据表结构异常或产品信息不唯一时抛出。
        
        注意事项:
            - 本方法依赖于self.data_tables['product_contract_start_end']和self.rollover_points的数据结构和内容。
            - 若未检测到rollover_points，将自动调用self.detect_rollover_points()进行检测。
            - 若需添加调整因子，需先计算并加载self.data_tables['rollover_adjustments']，并指定调整策略。
            - 内部自动进行数据质量检测（如成交量为零的连续区段），并可根据预设方案进行修正。
            - 结果表按产品ID拼接，适用于主力合约连续行情的后续分析与建模。
        '''
        # 检查是否已经有product_id_list属性
        if not self.product_id_list:
            # 获取所有产品的合约起止日期表
            PCSE = self.data_tables.get('product_contract_start_end')
            if PCSE is None or PCSE.empty:
                raise ValueError("'product_contract_start_end'数据表为空或不存在")
            self.product_id_list = sorted(PCSE['PRODUCT'].unique().tolist())
        if not self.product_id_list:
            raise ValueError("'PRODUCT'列为空")

        # 准备调整因子数据
        if add_adjust_col_bool:
            # 确保有rollover_adjustments属性，没有则先跑calculate_adjustment
            if 'rollover_adjustments' not in self.data_tables:
                self.calculate_adjustment()
            assert self.strategy_selector is not None
            strategy = self.strategy_selector.default_strategy["AdjustmentStrategy"]
            self._rollover_adjustments_grouped = self.data_tables['rollover_adjustments'].groupby('product_id')
        else:
            strategy = None
        
        # 确保有rollover_points属性，没有则先跑detect_rollover_points
        if not hasattr(self, 'rollover_points'):
            self.detect_rollover_points()
        assert hasattr(self, 'rollover_points')

        # 初始化all_issues属性
        if not hasattr(self, 'all_issues'):
            self.all_issues = []

        # 如果在update_mode下，加载已有的main_contract_series
        if update_mode:
            assert save_path is not None, "在update_mode下，save_path不能为空"
            if not os.path.exists(save_path) and save_per_product:
                os.mkdir(save_path)
            assert os.path.exists(save_path), f"在update_mode下，save_path不存在: {save_path}"
            if save_per_product:
                assert os.path.isdir(save_path), "当save_per_product为True时，save_path必须是文件夹路径"
            else:
                assert os.path.isfile(save_path), "当save_per_product为False时，save_path必须是文件路径"
                self.main_contract_series = self.dataframe_loader(save_path, self.column_mapping.get('main_contract_series', None))

        # 定义添加调整列的内部函数
        def add_adjustment_columns(filtered: pd.DataFrame, strategy: AdjustmentStrategy, 
                                   product_id: str, rollover_date_str: str, 
                                   first_line: bool = False, last_line: bool = False) -> pd.DataFrame:
            if strategy.adjustment_direction == AdjustmentDirection.ADJUST_OLD:
                default_flag = True if last_line else False
                grouped_flag = 'curr' if first_line or last_line else 'prev'
                val_adjustment_col_name = 'val_adjust_old'
            elif strategy.adjustment_direction == AdjustmentDirection.ADJUST_NEW:
                default_flag = True if first_line else False
                grouped_flag = 'curr'
                val_adjustment_col_name = 'val_adjust_new'
            else:
                raise ValueError(f"{product_id} {rollover_date_str}: 未知的调整方向: {strategy.adjustment_direction}")
            if hasattr(self, '_rollover_adjustments_grouped_date_' + grouped_flag) \
                    and rollover_date_str in getattr(self, '_rollover_adjustments_grouped_date_' + grouped_flag).groups:
                existing_group = getattr(self, '_rollover_adjustments_grouped_date_' + grouped_flag).get_group(rollover_date_str)
                filtered_group = existing_group[
                    (existing_group['adjustment_strategy'] == strategy.__class__.__name__)
                    & (existing_group['adjustment_operation'] == strategy.adjustment_operation.name)
                    & (existing_group['description'] == strategy.description)
                    & (existing_group['is_valid'] == True)]
                if len(filtered_group) == 1:
                    if default_flag:
                        adjustment_mul = 1.0
                        adjustment_add = 0.0
                    else:
                        if strategy.adjustment_operation == AdjustmentOperation.ADDITIVE:
                            adjustment_add = filtered_group.iloc[0][val_adjustment_col_name]
                            adjustment_mul = 1.0
                        elif strategy.adjustment_operation == AdjustmentOperation.MULTIPLICATIVE:
                            adjustment_add = 0.0
                            adjustment_mul = filtered_group.iloc[0][val_adjustment_col_name]
                        else:
                            raise ValueError(f"{product_id} {rollover_date_str} {grouped_flag}: 未知的调整操作类型: {strategy.adjustment_operation}")
                    filtered.insert(6, 'adjustment_mul', adjustment_mul)
                    filtered.insert(7, 'adjustment_add', adjustment_add)
                    filtered.insert(8, 'adjustment_direction', strategy.adjustment_direction.value)
                elif len(filtered_group) == 0:
                    filtered_group_invalid = existing_group[
                        (existing_group['adjustment_strategy'] == strategy.__class__.__name__)
                        & (existing_group['adjustment_operation'] == strategy.adjustment_operation.name)
                        & (existing_group['description'] == strategy.description)
                        & (existing_group['is_valid'] == False)]
                    if len(filtered_group_invalid) > 0:
                        if all(status in [ValidityStatus.INSUFFICIENT_DATA, ValidityStatus.LATER_INSUFFICIENT_DATA] \
                               for status in filtered_group_invalid['validity_status']):
                            return filtered
                    raise ValueError(f"{product_id} {rollover_date_str} {grouped_flag}: 在rollover_adjustments中未找到对应的调整因子记录")
                else:
                    raise ValueError(f"{product_id} {rollover_date_str} {grouped_flag}: 在rollover_adjustments中找到多个匹配的调整因子记录")
            else:
                raise ValueError(f"{product_id} {rollover_date_str} {grouped_flag}: 在rollover_adjustments中未找到对应的调整因子记录")
            return filtered
        
        def add_data_quality_columns(filtered: pd.DataFrame) -> pd.DataFrame:
            if filtered is None or filtered.empty:
                return filtered
            # Data Quality Checker (Volume)
            checker_2 = DataQualityChecker(filtered,
                                            columns=['volume'],
                                            column_mapping={'symbol': 'main_uid', 'time': 'trade_time'})
            checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_LONG] = DataIssueSolution.NO_ACTION
            checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_SHORT] = DataIssueSolution.NO_ACTION
            checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_ALL] = DataIssueSolution.NO_ACTION
            checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_AT_END] = DataIssueSolution.NO_ACTION
            checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_AT_START] = DataIssueSolution.NO_ACTION
            checker_2.solution_mapping[DataIssueLabel.ZERO_SEQUENCE_AT_VERY_START] = DataIssueSolution.NO_ACTION
            checker_2._assign_solution_by_issue_label()
            issues_df_2 = checker_2.issues_df
            if issues_df_2 is not None and not issues_df_2.empty:
                self.all_issues.append(issues_df_2)
            checked = checker_2.process_dataframe(
                mapping={
                DataIssueLabel.ZERO_SEQUENCE_LONG: DataIssueSolution.FORWARD_FILL,
                DataIssueLabel.ZERO_SEQUENCE_SHORT: DataIssueSolution.FORWARD_FILL,
                DataIssueLabel.ZERO_SEQUENCE_ALL: DataIssueSolution.FORWARD_FILL,
                DataIssueLabel.ZERO_SEQUENCE_AT_END: DataIssueSolution.FORWARD_FILL,
                DataIssueLabel.ZERO_SEQUENCE_AT_START: DataIssueSolution.NO_ACTION,
                DataIssueLabel.ZERO_SEQUENCE_AT_VERY_START: DataIssueSolution.NO_ACTION
            })
            # Data Quality Checker (Price)
            checker = DataQualityChecker(checked, 
                                            columns=['open_price', 'highest_price', 'lowest_price', 'close_price'],
                                            column_mapping={'symbol': 'main_uid', 'time': 'trade_time'})
            issues_df = checker.issues_df
            if issues_df is not None and not issues_df.empty:
                self.all_issues.append(issues_df)
            checked = checker.process_dataframe()
            return checked if checked is not None else filtered

        main_series = {}
        product_save_path = None
        for product_id in tqdm(self.product_id_list, desc="Generating main series " + source_data_label):

            if product_id not in main_series:
                main_series[product_id] = []

            # 获取product_save_path
            if save_per_product:
                assert save_path is not None
                if not os.path.exists(os.path.dirname(save_path)):
                    os.makedirs(os.path.dirname(save_path))
                assert os.path.isdir(save_path), "当save_per_product为True时，save_path必须是文件夹路径"
                product_save_path = os.path.join(save_path, product_id + '.' + save_ext)

            # 初始化 last_trading_day 以避免未绑定错误
            last_trading_day = None

            # 在update_mode下，加载已有的main_contract_series以获取last_trading_day
            if update_mode:
                if save_per_product:
                    assert product_save_path is not None, "在save_per_product模式下，product_save_path不能为空"
                    if os.path.exists(product_save_path):
                        this_main_contract_series = self.dataframe_loader(product_save_path, self.column_mapping.get('main_contract_series', None))
                        main_series[product_id].append(this_main_contract_series)
                    else:
                        this_main_contract_series = None
                else:
                    this_main_contract_series = self.main_contract_series[
                        self.main_contract_series['unique_instrument_id'] == product_id]
                    main_series[product_id].append(this_main_contract_series)
                if this_main_contract_series is not None and not this_main_contract_series.empty:
                    last_trading_day = this_main_contract_series['trading_day'].max()
                else:
                    last_trading_day = None

            if add_adjust_col_bool and hasattr(self, '_rollover_adjustments_grouped') and product_id in self._rollover_adjustments_grouped.groups:
                self._rollover_adjustments_grouped_date_curr = self._rollover_adjustments_grouped.get_group(product_id).groupby('new_contract_start_date')
                self._rollover_adjustments_grouped_date_prev = self._rollover_adjustments_grouped.get_group(product_id).groupby('old_contract_start_date')
            
            if product_id in self.rollover_points:
                rollover_keys = sorted(self.rollover_points[product_id].keys())

                lt_keys = []
                if update_mode and last_trading_day is not None:
                    # 只保留第一个小于last_trading_day的rollover_key和所有大于等于last_trading_day的rollover_keys
                    rollover_dates = [pd.to_datetime(k) for k in rollover_keys]
                    last_trading_day_dt = pd.to_datetime(last_trading_day)
                    # 如果有小于last_trading_day的，保留最后一个
                    lt_keys = [k for k, d in zip(rollover_keys, rollover_dates) if d < last_trading_day_dt]
                    rollover_keys = [k for k, d in zip(rollover_keys, rollover_dates) if d >= last_trading_day_dt]
                    if lt_keys:
                        rollover_keys = [lt_keys[-1]] + rollover_keys

                # 如果在rollover_points中，合并所有rollover的主合约数据
                for idx, rollover_date in enumerate(rollover_keys):
                    rollover = self.rollover_points[product_id][rollover_date]
                    # 只处理合法的rollover
                    if not rollover.is_valid:
                        continue
                    # 加载该合约的主合约数据，并按rollover_date筛选
                    # old_contract: 取在rollover_date之前的行（如果有idx-1行，则取[idx-1].rollover_date当天及之后且本行rollover_date之前的行）
                    # new_contract: 取本行rollover_date当天及之后的行（如果有下一行，则只取到[next].rollover_date之前）
                    assert rollover.new_contract_start_date is not None
                    rollover_date = pd.to_datetime(rollover.new_contract_start_date)
                    # old_contract部分
                    if not lt_keys and (idx == 0 or (idx > 0 and not self.rollover_points[product_id][rollover_keys[idx - 1]].is_valid)):
                        contract_data = self.contract_data_loader(rollover.old_contract, 'contract_' + source_data_label, 
                                                                  mink_folder_path=source_data_folder_UID_path)
                        trading_days = pd.to_datetime(contract_data['trading_day'])
                        if idx == 0:
                            filtered = contract_data[trading_days < rollover_date]
                        else:
                            prev_rollover = self.rollover_points[product_id][rollover_keys[idx - 1]]
                            assert prev_rollover.new_contract_start_date is not None
                            prev_rollover_date = pd.to_datetime(prev_rollover.new_contract_start_date)
                            filtered = contract_data[(trading_days >= prev_rollover_date) & (trading_days < rollover_date)]
                        if add_adjust_col_bool and strategy is not None:
                            assert self.strategy_selector is not None
                            if product_id in self.strategy_selector.strategy_map and \
                                    len(self.strategy_selector.strategy_map[product_id]) > 0:
                                max_trading_day = filtered['trading_day'].max()
                                for start, end, strategy_name, _strategy in self.strategy_selector.strategy_map[product_id]:
                                    if strategy_name == 'AdjustmentStrategy' and start <= max_trading_day <= end:
                                        strategy = _strategy
                                        break
                            filtered = add_adjustment_columns(filtered, strategy, product_id, 
                                                              str(rollover.new_contract_start_date), first_line=True)
                        filtered = filtered.rename(columns={'unique_instrument_id': 'main_uid'})
                        filtered.insert(0, 'unique_instrument_id', product_id)
                        filtered = add_data_quality_columns(filtered)
                        if update_mode and last_trading_day is not None:
                            trading_days = pd.to_datetime(filtered['trading_day'])
                            filtered = filtered[trading_days > pd.to_datetime(last_trading_day)]
                        main_series[product_id].append(filtered.dropna(axis=1, how='all'))
                    # new_contract部分
                    contract_data = self.contract_data_loader(rollover.new_contract, 'contract_' + source_data_label, 
                                                              mink_folder_path=source_data_folder_UID_path)
                    trading_days = pd.to_datetime(contract_data['trading_day'])
                    if idx + 1 < len(rollover_keys):
                        last_line = False
                        next_rollover = self.rollover_points[product_id][rollover_keys[idx + 1]]
                        assert next_rollover.new_contract_start_date is not None
                        next_rollover_date = pd.to_datetime(next_rollover.new_contract_start_date)
                        filtered = contract_data[(trading_days >= rollover_date) & (trading_days < next_rollover_date)]
                    else:
                        last_line = True
                        filtered = contract_data[trading_days >= rollover_date]
                    if add_adjust_col_bool and strategy is not None:
                        assert self.strategy_selector is not None
                        if product_id in self.strategy_selector.strategy_map and \
                                len(self.strategy_selector.strategy_map[product_id]) > 0:
                            min_trading_day = filtered['trading_day'].min()
                            for start, end, strategy_name, _strategy in self.strategy_selector.strategy_map[product_id]:
                                if strategy_name == 'AdjustmentStrategy' and start <= min_trading_day <= end:
                                    strategy = _strategy
                                    break
                        filtered = add_adjustment_columns(filtered, strategy, product_id, 
                                                            str(rollover.new_contract_start_date), last_line=last_line)
                    filtered = filtered.rename(columns={'unique_instrument_id': 'main_uid'})
                    filtered.insert(0, 'unique_instrument_id', product_id)
                    filtered = add_data_quality_columns(filtered)
                    if update_mode and last_trading_day is not None:
                        trading_days = pd.to_datetime(filtered['trading_day'])
                        filtered = filtered[trading_days > pd.to_datetime(last_trading_day)]
                    main_series[product_id].append(filtered.dropna(axis=1, how='all'))
            else:
                # 如果不在rollover_points中，直接加载该产品的主合约数据
                PCSE = self.data_tables.get('product_contract_start_end')
                if PCSE is None or PCSE.empty:
                    raise ValueError("'product_contract_start_end'数据表为空或不存在")
                product_df = PCSE[PCSE['PRODUCT'] == product_id].reset_index(drop=True)
                if product_df.empty or len(product_df) > 1:
                    raise ValueError(f"{product_id}: 'product_contract_start_end'中应当存在唯一的产品信息")
                # 取出这一行对应的CONTRACT，转化为unique_instrument_id
                contract = product_df.iloc[0]['CONTRACT']
                uid = windcode_to_unique_instrument_id(contract, decade_str=self.calculate_decade_str(product_df.iloc[0]))
                # 加载该合约的主合约数据
                contract_data = self.contract_data_loader(uid, 'contract_' + source_data_label, 
                                                            mink_folder_path=source_data_folder_UID_path)
                if add_adjust_col_bool:
                    contract_data.insert(6, 'adjustment_mul', 1.0)
                    contract_data.insert(7, 'adjustment_add', 0.0)
                contract_data = contract_data.rename(columns={'unique_instrument_id': 'main_uid'})
                contract_data.insert(0, 'unique_instrument_id', product_id)
                contract_data = add_data_quality_columns(contract_data)
                if update_mode and last_trading_day is not None:
                    trading_days = pd.to_datetime(contract_data['trading_day'])
                    contract_data = contract_data[trading_days > pd.to_datetime(last_trading_day)]
                main_series[product_id].append(contract_data.dropna(axis=1, how='all'))
            
            if save_per_product:
                if main_series[product_id]:
                    product_df = pd.concat(main_series[product_id], ignore_index=True)
                    # Ensure consistent format for 'trading_day' and 'trade_time'
                    if 'trading_day' in product_df.columns:
                        product_df['trading_day'] = pd.to_datetime(product_df['trading_day']).dt.strftime('%Y-%m-%d')
                    if 'trade_time' in product_df.columns:
                        format = '%Y-%m-%d' if source_data_label == 'dayk' else '%Y-%m-%d %H:%M:%S'
                        product_df['trade_time'] = pd.to_datetime(product_df['trade_time']).dt.strftime(format)
                    self.save_data_frame_to_path(product_df, product_save_path)
                # main_series[product_id] = []  # 清空以节省内存
        
        # 删除self._rollover_adjustments_grouped以释放内存
        if hasattr(self, '_rollover_adjustments_grouped'):
            del self._rollover_adjustments_grouped
        if hasattr(self, '_rollover_adjustments_grouped_date_curr'):
            del self._rollover_adjustments_grouped_date_curr
        if hasattr(self, '_rollover_adjustments_grouped_date_prev'):
            del self._rollover_adjustments_grouped_date_prev

        if main_series:
            # Ensure consistent format for 'trading_day' and 'trade_time' before concatenation
            all_main_series = [pd.concat(dfs, ignore_index=True) for dfs in main_series.values() if dfs]
            if all_main_series:
                concat_df = pd.concat(all_main_series, ignore_index=True)
                if 'trading_day' in concat_df.columns:
                    concat_df['trading_day'] = pd.to_datetime(concat_df['trading_day']).dt.strftime('%Y-%m-%d')
                if 'trade_time' in concat_df.columns:
                    format = '%Y-%m-%d' if source_data_label == 'dayk' else '%Y-%m-%d %H:%M:%S'
                    concat_df['trade_time'] = pd.to_datetime(concat_df['trade_time']).dt.strftime(format)
                self.add_data_table('main_contract_series', concat_df)
            assert 'main_contract_series' in self.data_tables
            assert not self.data_tables['main_contract_series'].empty
            if not save_per_product:
                assert (save_path is None) or os.path.isfile(save_path)
                self.save_data_frame_to_path(self.data_tables['main_contract_series'], save_path)
            if hasattr(self, 'all_issues') and self.all_issues:
                self.save_data_frame_to_path(pd.concat(self.all_issues, ignore_index=True), issues_save_path)
            return self.data_tables['main_contract_series']
        else:
            return pd.DataFrame()
        
    @staticmethod
    def save_data_frame_to_path(data: pd.DataFrame, save_path: Optional[str] = None):
        '''
        将DataFrame保存到指定路径的文件中，支持csv和parquet格式。
        
        参数:
            data (pd.DataFrame): 需要保存的DataFrame数据。
            save_path (str): 保存文件的完整路径，支持以.csv或.parquet结尾的文件名。
        '''
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_path.endswith('.csv'):
                data.to_csv(save_path, index=False)
            elif save_path.endswith('.parquet'):
                # Convert Enum columns to string before saving to parquet
                for col in data.columns:
                    if data[col].dtype == 'object' and data[col].apply(lambda x: hasattr(x, 'name')).any():
                        data = data.copy()
                        data[col] = data[col].apply(lambda x: x.name if hasattr(x, 'name') else x)
                data.to_parquet(save_path, index=False)
            else:
                raise ValueError("仅支持保存为csv或parquet格式文件")
        
    def generate_main_contract_series_adjusted(self, 
                                               data: Optional[pd.DataFrame] = None, save_path: Optional[str] = None,
                                               uid_col: str = 'unique_instrument_id', time_col: str = 'trade_time', rollover_idx_col: str = 'trading_day',
                                               price_cols: List[str] = ['open_price', 'highest_price', 'lowest_price', 'close_price', 'twap', 'vwap', 
                                                                        'settlement_price', 'upper_limit_price', 'lower_limit_price', 'pre_settlement_price'],
                                               report_bool: bool = True, report_save_path: Optional[str] = None,
                                               plot_bool: bool = False, plot_col_name: str = 'close_price', plot_save_path: Optional[str] = None,
                                               plot_start_date: Optional[datetime|str] = None, plot_end_date: Optional[datetime|str] = None,
                                               plot_sample_step: Optional[int] = None, plot_max_points: int = 5000) -> pd.DataFrame:
        '''
        生成主力合约序列的复权行情数据（主力合约连续行情调整后数据）。

        本方法基于已有的主力合约行情数据表（'main_contract_series'），结合价格调整因子（adjustment_mul, adjustment_add），对指定的行情价格字段（如开盘价、最高价、最低价、收盘价）进行复权处理，生成调整后的主力合约连续行情序列。支持输出每个产品的复权数据统计报告，并可选地绘制复权前后的价格对比图。最终结果以DataFrame形式返回，并存入self.data_tables['main_contract_series_adjusted']。

        参数:
            data (Optional[pd.DataFrame]): 输入的主力合约行情数据表，若为None则自动读取self.data_tables['main_contract_series']。
            save_path (Optional[str]): 结果保存路径，支持csv或parquet格式。
            price_cols (List[str]): 需要进行复权调整的行情价格字段列表，默认为['open_price', 'highest_price', 'lowest_price', 'close_price']。
            report_bool (bool): 是否输出每个产品的复权后行情统计报告，默认为True。
            report_save_path (Optional[str]): 复权统计报告保存路径。
            plot_bool (bool): 是否绘制复权前后价格对比图，默认为False。
            plot_col_name (str): 绘图时使用的价格字段，默认为'close_price'。
            plot_save_path (Optional[str]): 绘图文件保存文件夹路径。
            plot_start_date (Optional[datetime|str]): 绘图起始日期，可选。
            plot_end_date (Optional[datetime|str]): 绘图结束日期，可选。
            plot_sample_step (Optional[int]): 绘图采样步长，减少点数以加快绘图速度。
            plot_max_points (int): 单张图最大采样点数，默认为5000。

        返回值:
            pd.DataFrame: 返回复权调整后的主力合约连续行情数据表。每行包括原始行情数据及对应的复权行情字段（如open_price_adjusted等）。

        异常:
            ValueError:
            - 当未找到'main_contract_series'数据表且未传入data参数时抛出。
            AssertionError:
            - 当输入数据缺少adjustment_mul或adjustment_add字段时抛出。
            - 结果表未成功写入self.data_tables时抛出。

        注意事项:
            - 输入数据需包含adjustment_mul和adjustment_add两列，分别为价格的乘法和加法调整因子。
            - 复权处理方式为：adjusted_price = 原价 * adjustment_mul + adjustment_add。
            - 若report_bool为True，将按unique_instrument_id分组输出每个产品的复权行情统计信息（最大值、最小值、有效行数等），并可保存到指定路径。
            - 若plot_bool为True，将为每个产品绘制复权前后价格对比图，自动标注合约切换点。
            - 结果表会自动写入self.data_tables['main_contract_series_adjusted']，便于后续分析与调用。
        '''
        if data is None:
            if 'main_contract_series' not in self.data_tables:
                raise ValueError("'main_contract_series'数据表不存在，请先运行generate_main_contract_series方法")
            data = self.data_tables['main_contract_series']
        
        assert 'adjustment_mul' in data.columns and 'adjustment_add' in data.columns

        adjusted_data = data.copy()

        # 删除所有adjusted列全为NaN的行
        adjusted_col_names = ['adjustment_mul', 'adjustment_add']
        if adjusted_col_names:
            adjusted_data = adjusted_data.dropna(subset=adjusted_col_names, how='all')

        assert uid_col in data.columns
        assert uid_col in adjusted_data.columns
        
        # 计算删除行前后的product_id列表差异
        removed_product_ids = sorted(list(set(data[uid_col].unique()) - set(adjusted_data[uid_col].unique())))
        if removed_product_ids:
            print("删除没有adjustment值的行后减少的product_id列表:", removed_product_ids)

        for col in price_cols:
            if col in adjusted_data.columns:
                adjusted_col_name = col + '_adjusted'
                adjusted_data[adjusted_col_name] = adjusted_data[col] * adjusted_data['adjustment_mul'] + adjusted_data['adjustment_add']

        if report_bool:
            adjusted_data_grouped = adjusted_data.groupby(uid_col)
            all_reports = []
            for product_id, group in tqdm(adjusted_data_grouped, desc="Generating adjustment reports"):
                adjusted_rows = group[price_cols].notnull().all(axis=1)
                report = pd.DataFrame({'product_id': [product_id],
                                        'num_rows': [len(group)]})
                if adjusted_rows.any():
                    first_idx = adjusted_rows.idxmax()
                    last_idx = adjusted_rows[::-1].idxmax()
                    report['adjusted_start_date'] = group.loc[first_idx, rollover_idx_col]
                    report['adjusted_end_date'] = group.loc[last_idx, rollover_idx_col]
                else:
                    report['adjusted_start_date'] = None
                    report['adjusted_end_date'] = None
                
                for col in price_cols:
                    adjusted_col_name = col + '_adjusted'
                    if adjusted_col_name in group.columns:
                        max_adjusted = group[adjusted_col_name].max()
                        min_adjusted = group[adjusted_col_name].min()
                        report[adjusted_col_name + '_max'] = max_adjusted
                        report[adjusted_col_name + '_min'] = min_adjusted
                all_reports.append(report)
            if all_reports:
                if report_save_path is not None:
                    self.save_data_frame_to_path(pd.concat(all_reports, ignore_index=True), report_save_path)

        if plot_bool:
            self.plot_adjusted(data=data, adjusted_data=adjusted_data, uid_col=uid_col, time_col=time_col,
                               rollover_idx_col=rollover_idx_col, plot_col_name=plot_col_name,
                               plot_save_path=plot_save_path, plot_start_date=plot_start_date,
                               plot_end_date=plot_end_date, plot_sample_step=plot_sample_step,
                               plot_max_points=plot_max_points)

        self.add_data_table('main_contract_series_adjusted', adjusted_data)
        assert 'main_contract_series_adjusted' in self.data_tables
        assert not self.data_tables['main_contract_series_adjusted'].empty
        self.save_data_frame_to_path(self.data_tables['main_contract_series_adjusted'], save_path)
        return self.data_tables['main_contract_series_adjusted']
    
    def plot_adjusted(self, data: pd.DataFrame, adjusted_data: pd.DataFrame, uid_col: str = 'unique_instrument_id',
                      time_col: str = 'trade_time', rollover_idx_col: str = 'trading_day',
                      plot_col_name: str = 'close_price', plot_save_path: Optional[str] = None,
                      plot_start_date: Optional[datetime|str] = None, plot_end_date: Optional[datetime|str] = None,
                      plot_sample_step: Optional[int] = None, plot_max_points: int = 5000):
        '''绘制复权前后的价格对比图，标注合约切换点。'''
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'Heiti TC', 'STHeiti', 'PingFang SC']
        plt.rcParams['axes.unicode_minus'] = False

        adjusted_data_grouped = adjusted_data.groupby(uid_col)

        if plot_start_date:
            plot_start_date = pd.to_datetime(plot_start_date)
        if plot_end_date:
            plot_end_date = pd.to_datetime(plot_end_date)

        for uid, plot_data in tqdm(adjusted_data_grouped, desc="Plotting adjusted prices"):

            assert type(uid) == str
            
            plot_data[time_col] = pd.to_datetime(plot_data[time_col])
            if plot_start_date:
                plot_data = plot_data[plot_data[time_col] >= plot_start_date]
            if plot_end_date:
                plot_data = plot_data[plot_data[time_col] <= plot_end_date]

            # 确保plot_sampled中包含所有rollover_points的old_contract_end_date和new_contract_start_date
            if not self.rollover_points or uid not in self.rollover_points:
                self.detect_rollover_points()
            rollover_dates = []
            for rollover in self.rollover_points[uid].values():
                assert rollover.old_contract_end_date is not None and rollover.new_contract_start_date is not None
                rollover_dates.append(pd.to_datetime(rollover.old_contract_end_date))
                rollover_dates.append(pd.to_datetime(rollover.new_contract_start_date))
            rollover_dates = [d for d in rollover_dates if not pd.isnull(d)]
            # 取所有需要保留的索引
            must_keep_idx = plot_data[plot_data[rollover_idx_col].isin(rollover_dates)].index.tolist()

            # 抽样以减少数据点，但保留所有rollover点
            step = max(len(plot_data) // plot_max_points, 1)
            sampled_idx = set(plot_data.iloc[::(plot_sample_step if plot_sample_step else step)].index.tolist())
            all_idx = sorted(set(sampled_idx).union(must_keep_idx))
            plot_sampled = plot_data.loc[all_idx]

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))

            # 先画切换点的线和文本（在价格线下方）
            # 先获取y轴范围（先画一条线获取ylim）
            ax.plot(plot_sampled[time_col], plot_sampled[plot_col_name], alpha=0)
            ylim = ax.get_ylim()
            for rollover in self.rollover_points[uid].values():
                assert rollover.new_contract_start_date is not None
                rv = pd.to_datetime(rollover.new_contract_start_date)

                # 检查切换点是否在绘图范围内
                if plot_start_date and rv < plot_start_date:
                    continue
                if plot_end_date and rv > plot_end_date:
                    continue
                
                rv_num = float(mdates.date2num(rv))
                ax.axvline(rv_num, color="#FF9999AB", linestyle='--', alpha=0.5, zorder=1)
                # ax.text(
                #     rv, ylim[1],
                #     f'{rollover.old_contract}→{rollover.new_contract}',
                #     rotation=90, va='top', ha='right', fontsize=8, color="#FF9999AB", alpha=0.6, zorder=1
                # )

            # 再画两条价格线（在切换点标记之上）
            ax.plot(plot_sampled[time_col], plot_sampled[plot_col_name], label='Original', color='blue', alpha=0.7, linewidth=1, zorder=2)
            ax.plot(plot_sampled[time_col], plot_sampled[plot_col_name + '_adjusted'], label='Adjusted', color='green', alpha=0.7, linewidth=1, zorder=2)

            # 智能设置x轴刻度
            self._set_smart_date_ticks(ax, plot_sampled[time_col])

            ax.set_title(f'{uid}: 原始与前复权连续合约价格 ({plot_col_name})')
            ax.set_xlabel('时间')
            ax.set_ylabel('价格')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            # plt.tight_layout()
            if plot_save_path is not None:
                if not os.path.exists(plot_save_path):
                    os.makedirs(plot_save_path)
                plt.savefig(os.path.join(plot_save_path, f'{uid}_adjusted_prices.png'), dpi=200, bbox_inches='tight')
            plt.close()
    
    @staticmethod
    def _set_smart_date_ticks(ax, datetimes):
        """智能设置日期刻度 - 避免AutoDateLocator警告"""
        import matplotlib.dates as mdates
        
        if len(datetimes) == 0:
            return
        
        # 计算时间范围
        time_range = datetimes.max() - datetimes.min()
        days = time_range.days
        hours = time_range.total_seconds() / 3600
        
        # 根据时间范围选择合适的刻度间隔
        if days > 365 * 2:  # 超过2年
            locator = mdates.YearLocator(1)
            formatter = mdates.DateFormatter('%Y')
        elif days > 180:  # 超过6个月
            locator = mdates.MonthLocator(interval=3)
            formatter = mdates.DateFormatter('%Y-%m')
        elif days > 60:  # 超过2个月
            locator = mdates.MonthLocator(interval=1)
            formatter = mdates.DateFormatter('%Y-%m')
        elif days > 14:  # 超过2周
            locator = mdates.WeekdayLocator(byweekday=mdates.MO.weekday, interval=1)
            formatter = mdates.DateFormatter('%m-%d')
        elif days > 2:  # 超过2天
            locator = mdates.DayLocator(interval=1)
            formatter = mdates.DateFormatter('%m-%d')
        elif hours > 12:  # 超过12小时
            locator = mdates.HourLocator(interval=6)
            formatter = mdates.DateFormatter('%H:%M')
        elif hours > 6:  # 超过6小时
            locator = mdates.HourLocator(interval=2)
            formatter = mdates.DateFormatter('%H:%M')
        elif hours > 2:  # 超过2小时
            locator = mdates.HourLocator(interval=1)
            formatter = mdates.DateFormatter('%H:%M')
        else:  # 短时间范围
            locator = mdates.MinuteLocator(interval=30)
            formatter = mdates.DateFormatter('%H:%M')
        
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        
        # 自动调整日期标签格式
        fig = ax.get_figure()
        if fig:
            fig.autofmt_xdate()
    
# End of DataMinkBasics.py
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

from datetime import datetime, date
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ContractRollover import ContractRollover
from FuturesProcessor import FuturesProcessorBase 
from StrategySelector import ProductPeriodStrategySelector
from DataQualityChecker import DataQualityChecker, DataIssueLabel, DataIssueSolution
from AdjustmentStrategy import ValidityStatus, AdjustmentStrategy, PercentageAdjustmentStrategy, AdjustmentDirection, AdjustmentOperation
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
        return ['product_contract_start_end', 'contract_dayk', 'contract_mink', 'main_tick', 'rollover_adjustments', 'main_tick_adjusted']
    
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
    
    def detect_rollover_points(self, rollover_points_cache_path: Optional[str] = None) -> Dict[str, List[ContractRollover]]:
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
        if not hasattr(self, 'product_id_list'):
            self.product_id_list = sorted(all_df['PRODUCT'].unique().tolist())
        if not self.product_id_list:
            raise ValueError("'PRODUCT'列为空")
        
        # 支持缓存rollover_points到本地文件
        if rollover_points_cache_path is not None:
            self.rollover_points_cache_path = rollover_points_cache_path
        cache_path = getattr(self, 'rollover_points_cache_path', None)
        if cache_path and os.path.exists(cache_path):
            with open(cache_path, 'rb') as f:
                self.rollover_points = pickle.load(f)

        # 初始化rollover_points字典
        if not hasattr(self, 'rollover_points'):
            self.rollover_points = {}
        
        # 遍历每个产品，检测展期点
        for product_id in tqdm(self.product_id_list, desc="Detecting rollover points"):
            
            # 初始化该产品的展期点列表
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
                else:
                    old_contract_old_data = pd.DataFrame()
                    is_valid = False

                # 找到小于等于this_end_date的最大日期
                mask_new = pd.Series([False] * len(new_trading_days), index=new_trading_days.index)
                if not new_trading_days.empty:
                    mask_new = new_trading_days == new_trading_days[new_trading_days <= this_end_date].max()
                if mask_new.any():
                    new_contract_old_data = contract_data_cache[new_contract_UID][mask_new].iloc[[-1]]
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
                if rollover.is_valid:
                    rollover.validate_data_tables(['old_contract_old_data', 'new_contract_old_data'])
                self.rollover_points[product_id][str(next_start_date.date())] = rollover
        
        # 如果有缓存路径，则保存rollover_points到本地文件
        if cache_path is not None:
            with open(cache_path, 'wb') as f:
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
            file_name = unique_instrument_id + '.csv'
            if mink_folder_path is None:
                raise ValueError("mink_folder_path参数不能为空")
            file_path = os.path.join(mink_folder_path, file_name)
            if not os.path.exists(file_path):
                # 返回一个包含 contract_mink 所需列的空 DataFrame
                required_cols = self.REQUIRED_COLUMNS['contract_mink']
                # print(f"警告: 文件 {file_path} 不存在，返回空DataFrame。")
                return pd.DataFrame(columns=required_cols)
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"未知的table_name: {table_name}")
    
    def calculate_adjustment(self, 
                             product_id_list: Optional[List[str]] = None,
                             strategy_selector: Optional[ProductPeriodStrategySelector] = None,
                             rollover_adjustments_cache_path: Optional[str] = None) -> pd.DataFrame:
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

        # 判断self是否存在strategy_selector属性
        if not hasattr(self, 'strategy_selector'):
            if strategy_selector is not None:
                self.strategy_selector = strategy_selector
            else:
                self.strategy_selector = ProductPeriodStrategySelector(default_strategy={
                    "AdjustmentStrategy": PercentageAdjustmentStrategy(
                    old_price_field='close_price', new_price_field='close_price', 
                    new_price_old_data_bool=True, use_window=False,
                    description="prev_day_old_close_over_prev_day_new_close",
                    )
                })
            strategy_selector = self.strategy_selector
        else:
            if strategy_selector is None:
                strategy_selector = self.strategy_selector
        
        # 检查AdjustmentStrategy是否存在
        if not hasattr(strategy_selector, "default_strategy") or "AdjustmentStrategy" not in strategy_selector.default_strategy:
            raise ValueError("ProductPeriodStrategySelector.default_strategy 中缺少 'AdjustmentStrategy' 键")
        
        # 获取调整策略
        strategy = strategy_selector.default_strategy["AdjustmentStrategy"]
        
        # 获取产品ID列表
        if product_id_list is None:
            # 检查是否已经有product_id_list属性
            if not hasattr(self, 'product_id_list'):
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
            if not hasattr(self, 'rollover_points'):
                self.detect_rollover_points()
            elif product_id not in self.rollover_points:
                raise ValueError(f"{product_id}: 未检测到rollover_points")
            rollover_keys = sorted(self.rollover_points[product_id].keys())

            if hasattr(self, '_rollover_adjustments_grouped') and product_id in self._rollover_adjustments_grouped.groups:
                self._rollover_adjustments_grouped_2 = self._rollover_adjustments_grouped.get_group(product_id).groupby('new_contract_start_date')

            results = []
            is_valid = False
            renew_flag = False
            # 按照rollover_points的key（日期）从小到大排序遍历
            for idx, rollover_date in enumerate(rollover_keys):
                rollover = self.rollover_points[product_id][rollover_date]
                strategy = strategy_selector.get_strategy(product_id, pd.to_datetime(str(rollover_date)))['AdjustmentStrategy']

                prev_is_valid = is_valid
                is_valid, validity_status = strategy.is_valid(rollover)

                if prev_is_valid and not is_valid:
                    for res in results:
                        if res['is_valid']:
                            res['is_valid'] = False
                            res['validity_status'] = ValidityStatus.LATER_INSUFFICIENT_DATA

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
                    strategy.apply_adjustment_to_results(adjustment, results)
                else:
                    adjustment = None

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
                             add_adjust_col_bool: bool = False) -> pd.DataFrame:
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
        if not hasattr(self, 'product_id_list'):
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
            assert hasattr(self, 'strategy_selector')
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

        main_series = []
        for product_id in tqdm(self.product_id_list, desc="Generating main series"):

            if add_adjust_col_bool and hasattr(self, '_rollover_adjustments_grouped') and product_id in self._rollover_adjustments_grouped.groups:
                self._rollover_adjustments_grouped_date_curr = self._rollover_adjustments_grouped.get_group(product_id).groupby('new_contract_start_date')
                self._rollover_adjustments_grouped_date_prev = self._rollover_adjustments_grouped.get_group(product_id).groupby('old_contract_start_date')
            
            if product_id in self.rollover_points:
                rollover_keys = sorted(self.rollover_points[product_id].keys())
                # 如果在rollover_points中，合并所有rollover的主合约数据
                for idx, rollover_date in enumerate(rollover_keys):
                    rollover = self.rollover_points[product_id][rollover_date]
                    # 只处理合法的rollover
                    if not rollover.is_valid:
                        continue
                    # 加载该合约的主合约数据，并按rollover_date筛选
                    # old_contract: 取在rollover_date之前的行（如果有idx-1行，则取[idx-1].rollover_date当天及之后且本行rollover_date之前的行）
                    # new_contract: 取本行rollover_date当天及之后的行（如果有下一行，则只取到[next].rollover_date之前）
                    rollover_date = pd.to_datetime(rollover.new_contract_start_date)
                    # old_contract部分
                    if idx == 0 or (idx > 0 and not self.rollover_points[product_id][rollover_keys[idx - 1]].is_valid):
                        contract_data = self.contract_data_loader(rollover.old_contract, 'contract_' + source_data_label, 
                                                                  mink_folder_path=source_data_folder_UID_path)
                        trading_days = pd.to_datetime(contract_data['trading_day'])
                        if idx == 0:
                            filtered = contract_data[trading_days < rollover_date]
                        else:
                            prev_rollover_date = pd.to_datetime(self.rollover_points[product_id][rollover_keys[idx - 1]].new_contract_start_date)
                            filtered = contract_data[(trading_days >= prev_rollover_date) & (trading_days < rollover_date)]
                        if add_adjust_col_bool and strategy is not None:
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
                        main_series.append(filtered.dropna(axis=1, how='all'))
                    # new_contract部分
                    contract_data = self.contract_data_loader(rollover.new_contract, 'contract_' + source_data_label, 
                                                              mink_folder_path=source_data_folder_UID_path)
                    trading_days = pd.to_datetime(contract_data['trading_day'])
                    if idx + 1 < len(rollover_keys):
                        last_line = False
                        next_rollover_date = pd.to_datetime(self.rollover_points[product_id][rollover_keys[idx + 1]].new_contract_start_date)
                        filtered = contract_data[(trading_days >= rollover_date) & (trading_days < next_rollover_date)]
                    else:
                        last_line = True
                        filtered = contract_data[trading_days >= rollover_date]
                    if add_adjust_col_bool and strategy is not None:
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
                    main_series.append(filtered.dropna(axis=1, how='all'))
            else:
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
                main_series.append(contract_data.dropna(axis=1, how='all'))
        
        # 删除self._rollover_adjustments_grouped以释放内存
        if hasattr(self, '_rollover_adjustments_grouped'):
            del self._rollover_adjustments_grouped
        if hasattr(self, '_rollover_adjustments_grouped_date_curr'):
            del self._rollover_adjustments_grouped_date_curr
        if hasattr(self, '_rollover_adjustments_grouped_date_prev'):
            del self._rollover_adjustments_grouped_date_prev

        if main_series:
            return pd.concat(main_series, ignore_index=True)
        else:
            return pd.DataFrame()
        
# End of DataMinkBasics.py
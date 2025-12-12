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
from typing import List, Dict
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ContractRollover import ContractRollover
from RolloverDetector import FuturesRolloverDetectorBase 
from StrategySelector import ProductPeriodStrategySelector
from DataQualityChecker import DataQualityChecker
from AdjustmentStrategy import AdjustmentStrategy
import pandas as pd


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

class RolloverDetector(FuturesRolloverDetectorBase):
    @property
    def EXPECTED_TABLE_NAMES(self) -> List[str]:
        return ['product_contract_start_end', 'old_contract_tick', 'new_contract_tick', 'main_tick']
    
    @property
    def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
        return {
            'product_contract_start_end': ['PRODUCT', 'CONTRACT', 'STARTDATE', 'ENDDATE'],
            'old_contract_tick': ['trading_day', 'trade_time'],
            'new_contract_tick': ['trading_day', 'trade_time']
        }

    @property
    def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
        return {
            'product_contract_start_end': ['PRODUCT', 'CONTRACT', 'STARTDATE', 'ENDDATE'],
            'old_contract_tick': ['trading_day', 'trade_time'],
            'new_contract_tick': ['trading_day', 'trade_time']
        }
    
    def detect_rollover_points(self, 
                               path: str, 
                               suppress_year_before: int = 0,
                               generate_main_contract_series: bool = False) -> List[ContractRollover]:
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
        # 检查'PRODUCT'列是否唯一
        df = self.data_tables['product_contract_start_end']
        if df['PRODUCT'].nunique() != 1:
            raise ValueError("'PRODUCT'列必须唯一")
        product_id = df['PRODUCT'].iloc[0]
        
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
            curr_start = df.iloc[idx]['STARTDATE']
            curr_end = df.iloc[idx]['ENDDATE']
            next_start = df.iloc[idx + 1]['STARTDATE']
            next_end = df.iloc[idx + 1]['ENDDATE']
            # 如果curr_end前四位（年份）小于suppress_year_before，则删除
            if curr_end.year < suppress_year_before:
                drop_indices.append(idx)
                continue

            curr_contract = windcode_to_shortened_windcode(df.iloc[idx]['CONTRACT'])
            next_contract = windcode_to_shortened_windcode(df.iloc[idx + 1]['CONTRACT'])
            # if curr_contract == 'SF609.CZC':
            #     print(curr_contract, next_contract, curr_start, curr_end, next_start, next_end)
            if curr_contract == next_contract:
                if pd.isnull(curr_end) or pd.isnull(next_end) or pd.isnull(next_start):
                    raise ValueError(f"{product_id}: 第{idx}或{idx+1}行STARTDATE/ENDDATE有空值，无法比较")
                if curr_end < next_start:
                    # 把当前行的ENDDATE改成下一行的ENDDATE，然后删除下一行
                    df.at[idx, 'ENDDATE'] = df.iloc[idx + 1]['ENDDATE']
                    drop_indices.append(idx + 1)
                    continue
                if curr_end >= next_start and idx > 0 and curr_start <= df.iloc[idx - 1]['ENDDATE']:
                    drop_indices.append(idx)
                    # print(f"{product_id}: {df.iloc[idx - 1]}, {df.iloc[idx]}")
                    continue
                if (curr_end >= next_start) and (curr_end >= next_end):
                    drop_indices.append(idx + 1)
                    continue
                if curr_end >= next_start:
                    drop_indices.append(idx + 1)
                    # print(curr_contract, next_contract)
                    continue
                raise ValueError(f"{product_id}: 第{idx}和{idx+1}行CONTRACT重复但ENDDATE顺序不正确")
        if drop_indices:
            df = df.drop(df.index[drop_indices]).reset_index(drop=True)

        # # 打印所有行
        # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        #     print(df)

        # 检查每一行的ENDDATE是否在下一行的STARTDATE之前
        for idx in range(len(df) - 1):
            end_date = df.iloc[idx]['ENDDATE']
            next_start_date = df.iloc[idx + 1]['STARTDATE']
            if pd.isnull(end_date) or pd.isnull(next_start_date):
                continue
            if end_date >= next_start_date:
                df.at[idx, 'ENDDATE'] = next_start_date - pd.Timedelta(days=1)
                # print(f"警告: {product_id}: 第{idx}行的ENDDATE不在下一行的STARTDATE之前: {end_date} >= {next_start_date}，已自动调整ENDDATE为{df.at[idx, 'ENDDATE']}")
                # raise ValueError(f"第{idx}行的ENDDATE不在下一行的STARTDATE之前: {end_date} >= {next_start_date}")
        
        def calculate_decade_str(row):
            enddate_str = str(row['ENDDATE'])
            if len(enddate_str) < 4:
                raise ValueError(f"{product_id}: ENDDATE格式不正确，无法提取decade_str")
            # old_contract的小数点前倒数第三个数字字符
            contract_digits = ''.join([c for c in row['CONTRACT'] if c.isdigit()])
            if len(contract_digits) < 3:
                raise ValueError(f"{product_id}: contract中数字字符不足3位")
            contract_third_last = contract_digits[-3]
            if enddate_str[3] == contract_third_last:
                return enddate_str[2]
            else:
                startdate_str = str(row['STARTDATE'])
                if len(startdate_str) < 3:
                    raise ValueError(f"{product_id}: STARTDATE格式不正确，无法提取decade_str")
                return startdate_str[2]
                
        rollovers = []
        main_contract_series = []
        for idx in range(len(df)):
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
                self.add_data_table("old_contract_tick", self.contract_data_loader(old_contract_UID, path))
                self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, path))
            else:
                self.data_tables['old_contract_tick'] = self.data_tables['new_contract_tick']
                self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract_UID, path))
                
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
            
            new_contract_start_datetime = None  # Ensure variable is always defined
            old_contract_end_datetime = None  # Ensure variable is always defined
            
            # 找到小于等于this_end_date的最大日期
            mask = pd.Series([False] * len(old_trading_days), index=old_trading_days.index)
            if not old_trading_days.empty:
                mask = old_trading_days == old_trading_days[old_trading_days <= this_end_date].max()
            if mask.any():
                old_contract_old_data = self.data_tables['old_contract_tick'][mask].iloc[[-1]]
                old_contract_end_datetime = old_contract_old_data['trade_time'].iloc[0]
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
            
            # 找到等于next_start_date的日期
            mask_new_next = new_trading_days == next_start_date
            if mask_new_next.any():
                new_contract_new_data = self.data_tables['new_contract_tick'][mask_new_next].iloc[[0]]
                new_contract_start_datetime = new_contract_new_data['trade_time'].iloc[0]
            else:
                new_contract_new_data = pd.DataFrame()
                is_valid = False
            
            rollover = ContractRollover(
                old_contract=old_contract_UID,
                new_contract=new_contract_UID,
                datetime_col_name='trade_time',
                old_contract_old_data=old_contract_old_data,
                old_contract_new_data=pd.DataFrame(),
                new_contract_old_data=new_contract_old_data,
                new_contract_new_data=new_contract_new_data,
                old_contract_end_date=this_end_date.date(),
                old_contract_end_datetime=old_contract_end_datetime,
                new_contract_start_date=next_start_date.date(),
                new_contract_start_datetime=new_contract_start_datetime,
                is_valid=is_valid
            )
            rollovers.append(rollover)

            if generate_main_contract_series:
                if idx == 0:
                    mask = (old_trading_days >= this_start_date) & (old_trading_days <= this_end_date)
                    main_contract_series.append(self.data_tables['old_contract_tick'][mask])
                mask = (new_trading_days >= next_start_date) & (new_trading_days <= next_end_date)
                main_contract_series.append(self.data_tables['new_contract_tick'][mask])

        self.rollover_points = rollovers
        if generate_main_contract_series:
            all_issues = []
            for df in main_contract_series:
                if df.empty:
                    continue
                checker = DataQualityChecker(df, columns=['open_price', 'highest_price', 'lowest_price', 'close_price'],
                                             column_mapping={'symbol': 'unique_instrument_id', 'time': 'trade_time'})
                all_issues.append(checker.issues_df)
            if all_issues:
                self.data_tables['main_tick_issues'] = pd.concat(all_issues, ignore_index=True)
            else:
                self.data_tables['main_tick_issues'] = pd.DataFrame()
            non_empty_series = [df for df in main_contract_series if not df.empty]
            self.data_tables['main_tick'] = pd.concat(non_empty_series, ignore_index=True) if non_empty_series else pd.DataFrame()
        return rollovers
    
    def generate_main_contract_series(self, path: str) -> pd.DataFrame:
        self.detect_rollover_points(path, generate_main_contract_series=True)
        return self.data_tables.get('main_tick', pd.DataFrame())
    
    def contract_data_loader(self, contract: str, path : str) -> pd.DataFrame:
        """
        根据合约字符串和路径加载对应的DataFrame数据。

        参数:
            contract: 合约字符串（如 Wind 代码）
            path: 数据文件所在目录

        返回:
            对应合约的DataFrame数据

        说明:
            本方法根据合约字符串生成unique_instrument_id，并拼接为csv文件名，从指定路径读取数据。
        """
        file_name = contract + '.csv'
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            # 返回一个包含 old_contract_tick 所需列的空 DataFrame
            required_cols = self.REQUIRED_COLUMNS['old_contract_tick']
            # print(f"警告: 文件 {file_path} 不存在，返回空DataFrame。")
            return pd.DataFrame(columns=required_cols)
        return pd.read_csv(file_path)
    
    def get_adjustment_factor(self, strategy_selector: ProductPeriodStrategySelector) -> pd.DataFrame:
        """
        计算并返回每个合约切换点的复权因子（adjustment_factor），并记录所用的AdjustmentStrategy信息。

        参数:
            strategy_selector: ProductPeriodStrategySelector对象，包含默认的AdjustmentStrategy。

        返回:
            pd.DataFrame: 每个切换点的复权因子及相关信息，包含product_id、old/new unique_instrument_id、切换日期和adjustment。
        """
        if not hasattr(strategy_selector, "default_strategy") or "AdjustmentStrategy" not in strategy_selector.default_strategy:
            raise ValueError("ProductPeriodStrategySelector.default_strategy 中缺少 'AdjustmentStrategy' 键")

        strategy = strategy_selector.default_strategy["AdjustmentStrategy"]

        results = []
        for rollover in self.rollover_points:
            product_id = self.data_tables['product_contract_start_end']['PRODUCT'].iloc[0]
            if rollover.new_contract_start_datetime:
                reference_time = rollover.new_contract_start_datetime
            elif rollover.new_contract_start_date:
                reference_time = pd.to_datetime(str(rollover.new_contract_start_date))
            else:
                reference_time = pd.to_datetime('today')
            strategy = strategy_selector.get_strategy(product_id, reference_time)['AdjustmentStrategy']

            is_valid, validity_status = strategy.is_valid(rollover)
            if is_valid:
                adjustment, _ = strategy.calculate_adjustment(rollover)
            else:
                adjustment = None
            
            # 在每次循环时，将之前results中的adjustment都乘以当前adjustment（仅当adjustment不为None时）
            strategy.apply_adjustment_to_results(adjustment, results)

            results.append({
                'product_id': product_id,
                'old_unique_instrument_id': rollover.old_contract,
                'new_unique_instrument_id': rollover.new_contract,
                'rollover_date': rollover.new_contract_start_date,
                'rollover_datetime': rollover.new_contract_start_datetime,
                'adjustment_strategy': strategy.__class__.__name__,
                'is_valid': is_valid,
                'validity_status': validity_status,
                'adjustment': adjustment
            })
        return pd.DataFrame(results)
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

def windcode_to_unique_instrument_id(windcode: str) -> str:
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
        month = year_tens + month

    # 构造unique_instrument_id
    unique_instrument_id = f"{exchange}|F|{product_id}|{month}"
    return unique_instrument_id

class RolloverDetector(FuturesRolloverDetectorBase):
    @property
    def EXPECTED_TABLE_NAMES(self) -> List[str]:
        return ['product_contract_start_end', 'old_contract_tick', 'new_contract_tick']
    
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
    
    def detect_rollover_points(self, path: str) -> List[ContractRollover]:
        """
        检测合约切换点 - 子类实现

        Args:
            contract_data_loader: 一个函数，输入合约字符串，返回对应的DataFrame

        Returns:
            检测到的切换点列表

        Raises:
            NotImplementedError: 如果未实现
        """
        # 检查'PRODUCT'列是否唯一
        df = self.data_tables['product_contract_start_end']
        if df['PRODUCT'].nunique() != 1:
            raise ValueError("'PRODUCT'列必须唯一")
        
        # 检查'STARTDATE'和'ENDDATE'是否都有数据
        mask_start_null = df['STARTDATE'].isnull()
        mask_end_null = df['ENDDATE'].isnull()
        both_null = mask_start_null & mask_end_null
        one_null = mask_start_null ^ mask_end_null

        # 删除'STARTDATE'和'ENDDATE'都为空的行
        df.drop(df.index[both_null], inplace=True)
        if one_null.any():
            raise ValueError("存在'STARTDATE'或'ENDDATE'有一个为空的行")
        
        # 
        df = df.sort_values('STARTDATE').reset_index(drop=True)

        # 遍历所有行，检查CONTRACT重复且ENDDATE顺序
        drop_indices = []
        for idx in range(len(df) - 1):
            curr_contract = df.iloc[idx]['CONTRACT']
            next_contract = df.iloc[idx + 1]['CONTRACT']
            curr_end = df.iloc[idx]['ENDDATE']
            next_start = df.iloc[idx + 1]['STARTDATE']
            next_end = df.iloc[idx + 1]['ENDDATE']
            if curr_contract == next_contract:
                if pd.isnull(curr_end) or pd.isnull(next_end) or pd.isnull(next_start):
                    raise ValueError(f"第{idx}或{idx+1}行STARTDATE/ENDDATE有空值，无法比较")
                if curr_end < next_start:
                    # 把当前行的ENDDATE改成下一行的ENDDATE，然后删除下一行
                    df.at[idx, 'ENDDATE'] = df.iloc[idx + 1]['ENDDATE']
                    drop_indices.append(idx + 1)
                    continue
                if (curr_end >= next_start) and (curr_end >= next_end):
                    drop_indices.append(idx + 1)
                else:
                    raise ValueError(f"第{idx}和{idx+1}行CONTRACT重复但ENDDATE顺序不正确")
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
                raise ValueError(f"第{idx}行的ENDDATE不在下一行的STARTDATE之前: {end_date} >= {next_start_date}")
        
        rollovers = []
        for idx in range(len(df)):
            if idx + 1 >= len(df):
                continue

            this_row = df.iloc[idx]
            next_row = df.iloc[idx + 1]

            old_contract = this_row['CONTRACT']
            new_contract = next_row['CONTRACT']

            if idx == 0:
                self.add_data_table("old_contract_tick", self.contract_data_loader(old_contract, path))
                self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract, path))
            else:
                self.data_tables['old_contract_tick'] = self.data_tables['new_contract_tick']
                self.add_data_table("new_contract_tick", self.contract_data_loader(new_contract, path))

            # 检查数据表是否为空，跳过不合法情况
            old_tick_empty = self.data_tables["old_contract_tick"].empty
            new_tick_empty = self.data_tables["new_contract_tick"].empty
            if old_tick_empty and new_tick_empty:
                continue
            if old_tick_empty and not new_tick_empty:
                continue
            if not old_tick_empty and new_tick_empty:
                raise ValueError("old_contract_tick非空但new_contract_tick为空，不合法")
            
            # 兼容ENDDATE为数值（如20250312.0）或字符串格式
            this_end_val = this_row['ENDDATE']
            if pd.isnull(this_end_val):
                continue
            if isinstance(this_end_val, (int, float)):
                this_end_val = int(this_end_val)
                this_end_date = pd.to_datetime(str(this_end_val))
            else:
                this_end_date = pd.to_datetime(this_end_val)
            # 兼容STARTDATE为数值（如20250312.0）或字符串格式
            next_start_val = next_row['STARTDATE']
            if pd.isnull(next_start_val):
                continue
            elif isinstance(next_start_val, (int, float)):
                next_start_val = int(next_start_val)
                next_start_date = pd.to_datetime(str(next_start_val))
            else:
                next_start_date = pd.to_datetime(next_start_val)

            old_trading_days = pd.to_datetime(self.data_tables['old_contract_tick']['trading_day'])
            new_trading_days = pd.to_datetime(self.data_tables['new_contract_tick']['trading_day'])
            
            is_valid = True
            new_contract_start_datetime = None  # Ensure variable is always defined
            old_contract_end_datetime = None  # Ensure variable is always defined
            
            mask = old_trading_days == this_end_date
            if mask.any():
                old_contract_old_data = self.data_tables['old_contract_tick'][mask].iloc[[-1]]
                old_contract_end_datetime = old_contract_old_data['trade_time'].iloc[0]
            else:
                old_contract_old_data = pd.DataFrame()
                is_valid = False

            mask_new = new_trading_days == this_end_date
            if mask_new.any():
                new_contract_old_data = self.data_tables['new_contract_tick'][mask_new].iloc[[-1]]
            else:
                new_contract_old_data = pd.DataFrame()
                is_valid = False
                
            mask_new_next = new_trading_days == next_start_date
            if mask_new_next.any():
                new_contract_new_data = self.data_tables['new_contract_tick'][mask_new_next].iloc[[0]]
                new_contract_start_datetime = new_contract_new_data['trade_time'].iloc[0]
            else:
                new_contract_new_data = pd.DataFrame()
                is_valid = False
            
            rollover = ContractRollover(
                old_contract=old_contract,
                new_contract=new_contract,
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
        self.rollover_points = rollovers
        return rollovers
    
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
        file_name = windcode_to_unique_instrument_id(contract) + '.csv'
        file_path = os.path.join(path, file_name)
        if not os.path.exists(file_path):
            # 返回一个包含 old_contract_tick 所需列的空 DataFrame
            required_cols = self.REQUIRED_COLUMNS['old_contract_tick']
            # print(f"警告: 文件 {file_path} 不存在，返回空DataFrame。")
            return pd.DataFrame(columns=required_cols)
        return pd.read_csv(file_path)
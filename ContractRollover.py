'''
ContractRollover 数据结构用于管理和描述期货合约切换（Rollover）事件及相关数据。
主要功能：
- 存储旧合约与新合约的代码、切换时间点、有效性等基础信息。
- 管理四种数据表：旧合约历史数据、旧合约当前数据、新合约历史数据、新合约当前数据，便于分析合约切换前后的行情数据。
- 自动计算关键时间点，包括新合约开始时间/日期、旧合约结束时间/日期，辅助后续数据处理与分析。
- 提供数据摘要（get_data_summary），便于快速了解各数据表的基本情况（数据量、时间范围、字段）。
- 提供数据表有效性校验（validate_data_tables），确保指定的数据表存在有效数据。
输入：
- old_contract, new_contract: 字符串，分别为旧合约和新合约代码。
- rollover_datetime: datetime，可选，合约切换时间点。
- old_contract_old_data, old_contract_new_data, new_contract_old_data, new_contract_new_data: pd.DataFrame，四种数据表，存储不同阶段的行情数据。
- datetime_col_name: 字符串，时间列名称，默认为"datetime"。
输出：
- 关键时间点属性（如new_contract_start_datetime等），便于后续分析。
- get_data_summary() 方法返回各数据表的摘要信息（数据量、时间范围、字段）。
- validate_data_tables(table_names) 方法返回布尔值，指示指定数据表是否均有数据。
适用场景：
- 期货合约切换时的数据管理、分析与回测。
- 需要对合约切换前后行情数据进行对比、拼接或校验的场景。
'''

import pandas as pd
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

@dataclass
class ContractRollover:
    """合约切换事件的数据结构"""

    # 基础信息
    old_contract: str
    new_contract: str
    rollover_datetime: Optional[datetime] = None  # 切换时间点（新合约开始的时间）
    datetime_col_name: str = "datetime"  # 时间列名称
    is_valid: bool = True
    
    # 四种数据组合
    old_contract_old_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())  # 旧合约的历史数据
    old_contract_new_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())  # 旧合约的当前数据
    new_contract_old_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())  # 新合约的历史数据
    new_contract_new_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())  # 新合约的当前数据
    
    # 关键时间点
    new_contract_start_datetime: Optional[datetime] = None  # 新合约开始时间点
    new_contract_start_date: Optional[date] = None          # 新合约开始日期
    old_contract_end_datetime: Optional[datetime] = None    # 旧合约结束时间点
    old_contract_end_date: Optional[date] = None            # 旧合约结束日期
    
    def __post_init__(self):
        """确保有配置对象并计算关键时间点"""

        # 计算关键时间点
        self._calculate_key_timepoints()
    
    def _calculate_key_timepoints(self):
        """计算关键时间点"""
        if self.new_contract_start_datetime is None:
            # 新合约开始时间点就是rollover_datetime
            if self.rollover_datetime is not None:
                self.new_contract_start_datetime = self.rollover_datetime
            elif not self.new_contract_new_data.empty and self.datetime_col_name in self.new_contract_new_data.columns:
                self.new_contract_start_datetime = pd.to_datetime(self.new_contract_new_data[self.datetime_col_name].iloc[0])
            else:
                self.new_contract_start_datetime = None
        
        if self.old_contract_end_datetime is None or self.old_contract_end_date is None:
            # 旧合约结束时间点通过old_contract_old_data的最后一条数据获取
            if not self.old_contract_old_data.empty and self.datetime_col_name in self.old_contract_old_data.columns:
                self.old_contract_end_datetime = self.old_contract_old_data[self.datetime_col_name].max()
                # 旧合约结束日期
                if self.old_contract_end_datetime is not None:
                    self.old_contract_end_date = self.old_contract_end_datetime.date()
                else:
                    self.old_contract_end_date = None
            else:
                self.old_contract_end_datetime = None
                self.old_contract_end_date = None
        
        if self.new_contract_start_date is None:
            # 新合约开始日期需要通过遍历new_contract_new_data的datetime找到第一个与old_contract_end_date不一致的date
            if (not self.new_contract_new_data.empty and self.datetime_col_name in self.new_contract_new_data.columns and 
                self.old_contract_end_date):
                new_dates = pd.to_datetime(self.new_contract_new_data[self.datetime_col_name]).dt.date
                # 找到第一个与旧合约结束日期不同的日期
                different_dates = new_dates[new_dates != self.old_contract_end_date]
                if not different_dates.empty:
                    self.new_contract_start_date = different_dates.iloc[0]
                else:
                    # 如果所有日期都相同，则使用新合约的第一条数据的日期
                    self.new_contract_start_date = new_dates.iloc[0]
            elif not self.new_contract_new_data.empty and self.datetime_col_name in self.new_contract_new_data.columns:
                # 如果没有旧合约结束日期，则直接使用新合约第一条数据的日期
                self.new_contract_start_date = pd.to_datetime(self.new_contract_new_data[self.datetime_col_name]).iloc[0].date()
            else:
                self.new_contract_start_date = None

    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        return {
            'old_contract_old_data': {
                'data_points': len(self.old_contract_old_data),
                'time_range': (self.old_contract_old_data[self.datetime_col_name].min(), self.old_contract_old_data[self.datetime_col_name].max()) if not self.old_contract_old_data.empty else (None, None),
                'columns': list(self.old_contract_old_data.columns) if not self.old_contract_old_data.empty else []
            },
            'old_contract_new_data': {
                'data_points': len(self.old_contract_new_data),
                'time_range': (self.old_contract_new_data[self.datetime_col_name].min(), self.old_contract_new_data[self.datetime_col_name].max()) if not self.old_contract_new_data.empty else (None, None),
                'columns': list(self.old_contract_new_data.columns) if not self.old_contract_new_data.empty else []
            },
            'new_contract_old_data': {
                'data_points': len(self.new_contract_old_data),
                'time_range': (self.new_contract_old_data[self.datetime_col_name].min(), self.new_contract_old_data[self.datetime_col_name].max()) if not self.new_contract_old_data.empty else (None, None),
                'columns': list(self.new_contract_old_data.columns) if not self.new_contract_old_data.empty else []
            },
            'new_contract_new_data': {
                'data_points': len(self.new_contract_new_data),
                'time_range': (self.new_contract_new_data[self.datetime_col_name].min(), self.new_contract_new_data[self.datetime_col_name].max()) if not self.new_contract_new_data.empty else (None, None),
                'columns': list(self.new_contract_new_data.columns) if not self.new_contract_new_data.empty else []
            }
        }
    
    def validate_data_tables(self, table_names: List[str]) -> bool:
        """
        验证指定的数据表是否有数据
        
        Args:
            table_names: 数据表名称列表，应为以下值的子集：
                        ['old_contract_old_data', 'old_contract_new_data', 
                         'new_contract_old_data', 'new_contract_new_data']
            
        Returns:
            bool: 所有指定的数据表都有数据时返回True，否则返回False
        """
        for table_name in table_names:
            # 检查表名是否有效
            if table_name not in ['old_contract_old_data', 'old_contract_new_data', 
                                 'new_contract_old_data', 'new_contract_new_data']:
                raise ValueError(f"无效的数据表名称: {table_name}")
            
            # 获取数据表
            table_data = getattr(self, table_name)
            
            # 检查数据表是否有数据
            if table_data is None or table_data.empty:
                print(f"  数据表 {table_name} 无数据")
                return False
                
        return True
    
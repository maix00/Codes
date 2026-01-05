from enum import Enum
import os
import sys
from typing import List, Optional, Dict
import pandas as pd
from datetime import datetime
from abc import ABC  # Add this import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weakref import WeakValueDictionary  # Using weak references to avoid memory issues
# 定义一个金融产品的基类，包括它的产品名字、类别、币种、产业类别等信息
class ProductBase(ABC):
    _instances = WeakValueDictionary()  # Class-level dictionary to store instances by name
    
    def __new__(cls, name: str, *args, **kwargs):
        # Check if an instance with this name already exists
        if name in cls._instances:
            return cls._instances[name]
        
        # Create a new instance if it doesn't exist
        instance = super().__new__(cls)
        cls._instances[name] = instance
        return instance

    def __init__(self, name: str,
                 point_value: Optional[int] = None, currency: Optional[str] = None):
        # Only initialize if this is a new instance (not already initialized)
        if not hasattr(self, 'initialized'):
            self.name = name
            self.point_value = point_value
            self.currency = currency
            self.initialized = True
        
        # Always set security_type to the actual class
        self.security_type = self.__class__

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return self.name

class FuturesContract(ProductBase):
    def __init__(self, name: str, point_value: Optional[int] = None, currency: Optional[str] = None):
        super().__init__(name, point_value, currency)

class Futures(ProductBase):
    def __init__(self, name: str, point_value: Optional[int] = None, currency: Optional[str] = None,
                 mappings_path: Optional[str] = None):
        super().__init__(name, point_value, currency)
        self.mappings_path = mappings_path
        if self.mappings_path is not None:
            self.set_mappings(self.mappings_path)

    def set_mappings(self, path: str):
        self.mappings_path = path
        # 根据mappings_path的文件类型，导入映射表
        if path.endswith('.csv'):
            self.mappings = pd.read_csv(path)
        elif path.endswith('.xlsx'):
            self.mappings = pd.read_excel(path)
        elif path.endswith('.parquet'):
            self.mappings = pd.read_parquet(path)
        else:
            raise ValueError("Unsupported file type")
        # 筛选self.mappings中第一列中与self.name相同的行
        self.mappings = self.mappings[self.mappings.iloc[:, 0] == self.name]
        # 第三列是起始日期，第四列是终止日期，要进行数据类型的转化
        self.mappings.iloc[:, 2] = pd.to_datetime(self.mappings.iloc[:, 2])
        self.mappings.iloc[:, 3] = pd.to_datetime(self.mappings.iloc[:, 3])
    
    def get_contract_from_trading_day(self, trading_day: datetime|str) -> FuturesContract:
        if self.mappings is None:
            raise ValueError("Mappings not set, use set_mappings() first")
        # 先将trading_day转化为datetime
        trading_day = pd.to_datetime(trading_day)
        # 根据第三列和第四列，返回落于该区间内的第二列的值
        return FuturesContract(self.mappings[(self.mappings.iloc[:, 2] <= trading_day) & (self.mappings.iloc[:, 3] >= trading_day)].iloc[:, 1].values[0])

class PortfolioBackTester:
    def __init__(self, start_date: Optional[datetime|str] = None, end_date: Optional[datetime|str] = None,
                 initial_capital: Optional[float] = None, risk_free_rate: Optional[float] = None,
                 transaction_cost: Optional[float] = None, margin_rate: Optional[float] = None,
                 weight_type: Optional[str] = None, holdings_history: Optional[Dict[str, Dict[str, List[ProductBase]]]] = None):
        self.start_date: Optional[datetime] = None
        self.end_date: Optional[datetime] = None
        self.dates = []
        if start_date is not None:
            self.start_date = pd.to_datetime(start_date)
        if end_date is not None:
            self.end_date = pd.to_datetime(end_date)
        self.initial_capital = initial_capital
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
        self.margin_rate = margin_rate
        self.weight_type = weight_type
        self.holdings_history = holdings_history
        self.portfolio_history: Dict[str, Dict[str, Dict[ProductBase, float]]] = {}
        if self.holdings_history is not None:
            self._calc_portfolio_history_from_holdings_history()
        self.trade_history = []
    
    def _calc_portfolio_history_from_holdings_history(self, holdings_history: Optional[Dict[str, Dict[str, List[ProductBase]]]] = None,
                                                      weight_type: Optional[str] = None):
        # 如果已经计算过，则直接返回
        if self.portfolio_history and len(self.portfolio_history) > 0 and holdings_history is None:
            return
        # 如果传入了holdings_history，则更新
        if holdings_history is not None:
            self.holdings_history = holdings_history
        assert self.holdings_history is not None
        # 更新weight_type
        if weight_type is not None:
            self.weight_type = weight_type
        # 计算
        for holding_type, holding_type_history in self.holdings_history.items():
            self.portfolio_history[holding_type] = {}
            for date, holdings in holding_type_history.items():
                self.portfolio_history[holding_type][date] = {}
                num_holdings = len(holdings)
                if self.weight_type is None or self.weight_type == 'equal':
                    for holding in holdings:
                        if isinstance(holding, Futures):
                            holding = holding.get_contract_from_trading_day(date)
                        self.portfolio_history[holding_type][date][holding] = 1 / num_holdings
                else:
                    raise ValueError(f"Invalid weight type: {self.weight_type}")
            new_dates = set(holding_type_history.keys()) - set(self.dates)
            self.dates.extend(list(new_dates))
        self.dates.sort()

    def run_backtest(self, start_date: Optional[datetime|str] = None,
                     end_date: Optional[datetime|str] = None,
                     holdings_history: Optional[Dict[str, Dict[str, List[ProductBase]]]] = None,
                     weight_type: Optional[str] = None):
        # 更新start_date和end_date
        if start_date is not None:
            self.start_date = pd.to_datetime(start_date)
        if end_date is not None:
            self.end_date = pd.to_datetime(end_date)
        # 更新holdings_history
        if self.portfolio_history is not None and holdings_history is None:
            pass
        elif holdings_history is not None:
            self.holdings_history = holdings_history
            self._calc_portfolio_history_from_holdings_history(holdings_history, weight_type)
        else:
            raise ValueError("Invalid holdings_history")
        assert self.portfolio_history is not None

        # 初始化变量
        capital = self.initial_capital
        available_capital = capital
        margin_used = 0

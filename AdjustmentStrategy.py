'''
本模块定义了期货合约复权（调整）策略的基类 `AdjustmentStrategy` 及其若干实现类，用于在合约换月时对价格序列进行调整。主要内容包括：
1. `ValidityStatus` 枚举：定义了复权策略有效性的多种状态，如有效、零除无效、数据不足、负价格、手动覆盖等。
2. `AdjustmentStrategy` 抽象基类：规定了所有复权策略需实现的核心接口，包括：
    - `calculate_adjustment(rollover)`：计算调整因子和价差，输入为合约换月事件对象 `ContractRollover`，输出为 `(调整因子, 价差)` 的元组。
    - `is_valid(rollover)`：判断当前策略对给定换月事件是否有效，返回 `(是否有效, 状态)` 的元组。
    - `get_name()`：返回策略名称。
    - `clone()`：返回策略对象的深拷贝。
3. `PercentageAdjustmentStrategy`：基于百分比的复权策略，可选用窗口均值或单点价格，支持自定义价格字段。
4. `ManualOverrideStrategy`：手动指定调整因子和价差的策略，适用于特殊或人工干预场景。
输入：
    - `ContractRollover` 对象，包含旧合约与新合约的价格数据（如收盘价、结算价等）。
输出：
    - `calculate_adjustment` 返回 `(调整因子: float, 价差: float)`，用于后续价格序列的复权处理。
    - `is_valid` 返回 `(bool, ValidityStatus)`，指示策略是否适用及原因。
    - `get_name` 返回策略名称字符串。
功能说明：
    - 该模块为期货合约换月时的价格调整提供了统一的策略接口和多种实现方式，便于扩展和维护。
    - 支持自动和手动复权，能够根据实际业务需求灵活选择合适的调整方法。
'''

from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Optional, Tuple, Dict
from enum import Enum
from ContractRollover import ContractRollover

class ValidityStatus(Enum):
    VALID = "valid"
    INVALID_ZERO_DIVISION = "invalid_zero_division"
    INSUFFICIENT_DATA = "insufficient_data"
    NEGATIVE_PRICES = "negative_prices"
    MANUALLY_OVERRIDDEN_VALID = "manually_overridden_valid"
    MANUALLY_OVERRIDDEN_INVALID = "manually_overridden_invalid"

class AdjustmentStrategy(ABC):
    """复权策略基类"""
    
    @abstractmethod
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        """
        计算调整因子和价差
        
        Returns:
            Tuple[调整因子, 价差]
        """
        pass
    
    @abstractmethod
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        """
        检查策略是否适用于给定的切换事件
        
        Returns:
            Tuple[是否有效, 状态]
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass
    
    def clone(self) -> 'AdjustmentStrategy':
        """创建策略的副本"""
        import copy
        return copy.deepcopy(self)

class PercentageAdjustmentStrategy(AdjustmentStrategy):
    """百分比复权策略"""
    
    def __init__(self, use_window: bool = False, window_size: int = 120,
                 old_price_field: str = 'close', new_price_field: str = 'close',
                 new_price_old_data_bool: bool = True):
        self.use_window = use_window
        self.window_size = window_size
        self.old_price_field = old_price_field
        self.new_price_field = new_price_field
        self.new_price_old_data_bool = new_price_old_data_bool
        self.name = "percentage_window" if use_window else "percentage"

    @staticmethod
    def calculate_settlement(rollover: 'ContractRollover',
                             old_price_field: str,
                             new_price_field: str,
                             window_size: int,
                             new_price_old_data_bool: bool) -> Tuple[float, float]:
        """
        计算窗口内的旧合约和新合约的结算价格均值

        Args:
            rollover: ContractRollover对象
            old_price_field: 旧合约价格字段名
            new_price_field: 新合约价格字段名
            window_size: 窗口大小（行数）

        Returns:
            (旧合约结算均值, 新合约结算均值)
        """
        old_prices = rollover.old_contract_old_data[old_price_field].iloc[-window_size:]
        if new_price_old_data_bool:
            new_prices = rollover.new_contract_old_data[new_price_field].iloc[-window_size:]
        else:
            new_prices = rollover.new_contract_new_data[new_price_field].iloc[:window_size]
        old_price = float(old_prices.mean()) if not old_prices.empty else 0.0
        new_price = float(new_prices.mean()) if not new_prices.empty else 0.0
        return old_price, new_price
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        if self.use_window:
            old_price, new_price = self.calculate_settlement(
                rollover,
                self.old_price_field,
                self.new_price_field,
                self.window_size,
                self.new_price_old_data_bool
            )
        else:
            old_price = rollover.old_contract_old_data[self.old_price_field].iloc[-1]
            if self.new_price_old_data_bool:
                new_price = rollover.new_contract_old_data[self.new_price_field].iloc[-1]
            else:
                new_price = rollover.new_contract_new_data[self.new_price_field].iloc[0]
            
        if old_price == 0:
            print(f"  警告: 旧合约价格为0，无法计算百分比调整，使用1.0")
            return 1.0, 0.0
        
        adjustment = new_price / old_price
        # print(f"  调整因子计算: {new_price:.4f} / {old_price:.4f} = {adjustment:.6f}")
        return adjustment, 0.0
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        if not rollover.is_valid:
            return False, ValidityStatus.INSUFFICIENT_DATA
        if self.use_window:
            old_price, new_price = self.calculate_settlement(
                rollover,
                self.old_price_field,
                self.new_price_field,
                self.window_size,
                self.new_price_old_data_bool
            )
        else:
            old_price = rollover.old_contract_old_data[self.old_price_field].iloc[-1]
            if self.new_price_old_data_bool:
                new_price = rollover.new_contract_old_data[self.new_price_field].iloc[-1]
            else:
                new_price = rollover.new_contract_new_data[self.new_price_field].iloc[0]
        
        if old_price == 0:
            print(f"  无效原因: 旧合约价格为0")
            return False, ValidityStatus.INVALID_ZERO_DIVISION
        
        # 检查负价格
        if old_price < 0 or new_price < 0:
            print(f"  无效原因: 存在负价格")
            return False, ValidityStatus.NEGATIVE_PRICES
        
        # print(f"  策略有效")
        return True, ValidityStatus.VALID
    
    def get_name(self) -> str:
        return self.name

# class SpreadAdjustmentStrategy(AdjustmentStrategy):
#     """价差复权策略"""
    
#     def __init__(self, use_window: bool = True, window_size_hours: float = 2.0):
#         self.use_window = use_window
#         self.window_size_hours = window_size_hours
    
#     def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
#         if self.use_window:
#             old_price, new_price = rollover._calculate_settlement_prices()
#             gap = new_price - old_price
#             return 1.0, gap
#         else:
#             old_price, new_price = rollover._calculate_single_point_prices()
#             gap = new_price - old_price
#             return 1.0, gap
    
#     def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
#         # 价差复权几乎总是有效的，除非数据不足
#         if len(rollover.old_data) == 0 or len(rollover.new_data) == 0:
#             return False, ValidityStatus.INSUFFICIENT_DATA
#         return True, ValidityStatus.VALID
    
#     def get_name(self) -> str:
#         return "spread_window" if self.use_window else "spread"

# class WeightedAverageStrategy(AdjustmentStrategy):
#     """加权平均复权策略 - 用于处理异常情况"""
    
#     def __init__(self, volume_weights: Optional[Dict[str, float]] = None):
#         self.volume_weights = volume_weights or {}
    
#     def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
#         # 基于成交量加权的平均价格计算
#         old_avg = self._calculate_weighted_average(rollover.old_data)
#         new_avg = self._calculate_weighted_average(rollover.new_data)
        
#         if old_avg == 0:
#             return 1.0, 0.0
        
#         adjustment = new_avg / old_avg
#         gap = new_avg - old_avg
        
#         return adjustment, gap
    
#     def _calculate_weighted_average(self, prices: List[Tuple[datetime, float]]) -> float:
#         if not prices:
#             return 0.0
        
#         # 如果有成交量权重，使用加权平均
#         if self.volume_weights:
#             total_weight = 0
#             weighted_sum = 0
#             for ts, price in prices:
#                 contract_key = ts.strftime("%Y%m")
#                 weight = self.volume_weights.get(contract_key, 1.0)
#                 weighted_sum += price * weight
#                 total_weight += weight
            
#             if total_weight > 0:
#                 return weighted_sum / total_weight
        
#         # 否则使用简单平均
#         all_prices = [price for _, price in prices]
#         return sum(all_prices) / len(all_prices)
    
#     def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
#         if len(rollover.old_data) == 0 or len(rollover.new_data) == 0:
#             return False, ValidityStatus.INSUFFICIENT_DATA
#         return True, ValidityStatus.VALID
    
#     def get_name(self) -> str:
#         return "weighted_average"

class ManualOverrideStrategy(AdjustmentStrategy):
    """手动覆盖策略 - 允许用户手动指定调整参数"""
    
    def __init__(self, adjustment_factor: float = 1.0, price_gap: float = 0.0):
        self.adjustment_factor = adjustment_factor
        self.price_gap = price_gap
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        return self.adjustment_factor, self.price_gap
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        return True, ValidityStatus.MANUALLY_OVERRIDDEN_VALID
    
    def get_name(self) -> str:
        return "manual_override"

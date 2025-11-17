from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Optional, Tuple, Dict, Any, Callable, Union
import warnings
from enum import Enum
import numpy as np

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
        """计算调整因子和价差"""
        pass
    
    @abstractmethod
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        """检查策略是否适用于给定的切换事件"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass
    
    def clone(self) -> 'AdjustmentStrategy':
        """创建策略的副本"""
        import copy
        return copy.deepcopy(self)

# 具体的策略实现（与之前相同，这里省略以节省空间）
# PercentageAdjustmentStrategy, SpreadAdjustmentStrategy, WeightedAverageStrategy, ManualOverrideStrategy

class RolloverAdjustmentConfig:
    """合约切换调整配置 - 支持全局配置和个性化配置"""
    
    def __init__(self):
        # 全局默认策略
        self.global_default_strategy = "percentage_window"
        self.global_fallback_strategy = "spread_window"
        
        # 注册的策略
        self.registered_strategies: Dict[str, AdjustmentStrategy] = {}
        
        # 个性化配置：合约 -> 策略映射
        self.contract_specific_config: Dict[str, str] = {}  # 合约 -> 策略名
        
        # 时间范围特定配置
        self.time_based_config: List[Tuple[datetime, datetime, str]] = []  # (开始时间, 结束时间, 策略名)
        
        # 条件配置
        self.conditional_config: List[Tuple[Callable, str]] = []  # (条件函数, 策略名)
        
        # 注册默认策略
        self._register_default_strategies()
    
    def _register_default_strategies(self):
        """注册默认策略"""
        self.register_strategy(PercentageAdjustmentStrategy(use_window=False))
        self.register_strategy(PercentageAdjustmentStrategy(use_window=True))
        self.register_strategy(SpreadAdjustmentStrategy(use_window=False))
        self.register_strategy(SpreadAdjustmentStrategy(use_window=True))
        self.register_strategy(WeightedAverageStrategy())
    
    def register_strategy(self, strategy: AdjustmentStrategy):
        """注册策略"""
        self.registered_strategies[strategy.get_name()] = strategy
    
    def set_global_default(self, strategy_name: str):
        """设置全局默认策略"""
        if strategy_name not in self.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.global_default_strategy = strategy_name
    
    def set_global_fallback(self, strategy_name: str):
        """设置全局回退策略"""
        if strategy_name not in self.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.global_fallback_strategy = strategy_name
    
    def set_contract_strategy(self, contract_pattern: str, strategy_name: str):
        """为特定合约设置策略"""
        if strategy_name not in self.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.contract_specific_config[contract_pattern] = strategy_name
    
    def set_time_based_strategy(self, start_time: datetime, end_time: datetime, strategy_name: str):
        """为特定时间范围设置策略"""
        if strategy_name not in self.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.time_based_config.append((start_time, end_time, strategy_name))
    
    def set_conditional_strategy(self, condition: Callable[['ContractRollover'], bool], strategy_name: str):
        """设置条件策略"""
        if strategy_name not in self.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.conditional_config.append((condition, strategy_name))
    
    def get_strategy_for_rollover(self, rollover: 'ContractRollover') -> str:
        """为给定的切换事件获取适用的策略"""
        # 1. 检查条件配置（最高优先级）
        for condition, strategy_name in self.conditional_config:
            if condition(rollover):
                return strategy_name
        
        # 2. 检查时间范围配置
        for start_time, end_time, strategy_name in self.time_based_config:
            if start_time <= rollover.rollover_datetime <= end_time:
                return strategy_name
        
        # 3. 检查合约特定配置
        for contract_pattern, strategy_name in self.contract_specific_config.items():
            if contract_pattern in rollover.old_contract or contract_pattern in rollover.new_contract:
                return strategy_name
        
        # 4. 使用全局默认策略
        return self.global_default_strategy
    
    def get_fallback_strategy(self) -> str:
        """获取回退策略"""
        return self.global_fallback_strategy
    
    def get_available_strategies(self) -> List[str]:
        """获取所有已注册的策略名称"""
        return list(self.registered_strategies.keys())
    
    def get_strategy(self, name: str) -> Optional[AdjustmentStrategy]:
        """获取策略实例"""
        return self.registered_strategies.get(name)
    
    def validate_strategy_for_rollover(self, strategy_name: str, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        """验证策略对切换事件是否有效"""
        strategy = self.get_strategy(strategy_name)
        if not strategy:
            return False, ValidityStatus.INSUFFICIENT_DATA
        return strategy.is_valid(rollover)
    
    def get_recommended_strategy(self, rollover: 'ContractRollover') -> str:
        """为切换事件推荐策略（考虑有效性）"""
        # 获取配置的策略
        configured_strategy = self.get_strategy_for_rollover(rollover)
        
        # 检查配置的策略是否有效
        is_valid, status = self.validate_strategy_for_rollover(configured_strategy, rollover)
        if is_valid:
            return configured_strategy
        
        # 如果配置的策略无效，尝试回退策略
        is_fallback_valid, fallback_status = self.validate_strategy_for_rollover(
            self.global_fallback_strategy, rollover
        )
        if is_fallback_valid:
            warnings.warn(
                f"配置的策略 '{configured_strategy}' 无效 ({status.value})，"
                f"使用回退策略 '{self.global_fallback_strategy}'",
                UserWarning
            )
            return self.global_fallback_strategy
        
        # 如果回退策略也无效，尝试所有已注册的策略
        for strategy_name in self.registered_strategies:
            if strategy_name != configured_strategy and strategy_name != self.global_fallback_strategy:
                is_valid, status = self.validate_strategy_for_rollover(strategy_name, rollover)
                if is_valid:
                    warnings.warn(
                        f"配置的策略和回退策略均无效，使用策略 '{strategy_name}'",
                        UserWarning
                    )
                    return strategy_name
        
        # 如果所有策略都无效，返回配置的策略（尽管无效）
        warnings.warn(
            f"所有策略对切换事件均无效，使用配置的策略 '{configured_strategy}'（可能产生错误结果）",
            UserWarning
        )
        return configured_strategy

@dataclass
class ContractRollover:
    """合约切换事件的数据结构 - 使用统一配置和个性化策略"""
    
    # 基础信息
    rollover_datetime: datetime
    old_contract: str
    new_contract: str
    
    # 日内价格数据
    old_prices: List[Tuple[datetime, float]] = field(default_factory=list)
    new_prices: List[Tuple[datetime, float]] = field(default_factory=list)
    
    # 配置参数
    window_type: str = "time_based"
    window_size_hours: float = 2.0
    window_size_ticks: int = 100
    settlement_period_minutes: int = 15
    
    # 复权配置（可以是全局配置或个性化配置）
    adjustment_config: Optional[RolloverAdjustmentConfig] = None
    
    # 个性化策略覆盖
    custom_strategy: Optional[str] = None
    
    # 缓存字段
    _cached_settlement_prices: Optional[Tuple[float, float]] = None
    _cached_single_point_prices: Optional[Tuple[float, float]] = None
    _cached_negative_check: Optional[bool] = None
    
    def __post_init__(self):
        """确保有配置对象"""
        if self.adjustment_config is None:
            # 创建默认配置
            self.adjustment_config = RolloverAdjustmentConfig()
    
    def set_custom_strategy(self, strategy_name: str):
        """设置个性化策略（覆盖配置）"""
        if strategy_name not in self.adjustment_config.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.custom_strategy = strategy_name
    
    def clear_custom_strategy(self):
        """清除个性化策略（使用配置）"""
        self.custom_strategy = None
    
    def get_applied_strategy_name(self) -> str:
        """获取实际应用的策略名称"""
        if self.custom_strategy:
            return self.custom_strategy
        return self.adjustment_config.get_recommended_strategy(self)
    
    def get_applied_strategy(self) -> AdjustmentStrategy:
        """获取实际应用的策略实例"""
        strategy_name = self.get_applied_strategy_name()
        return self.adjustment_config.get_strategy(strategy_name)
    
    def get_adjustment_factor(self) -> float:
        """获取调整因子"""
        strategy = self.get_applied_strategy()
        adjustment, _ = strategy.calculate_adjustment(self)
        return adjustment
    
    def get_price_gap(self) -> float:
        """获取价差"""
        strategy = self.get_applied_strategy()
        _, gap = strategy.calculate_adjustment(self)
        return gap
    
    def is_strategy_valid(self, strategy_name: Optional[str] = None) -> Tuple[bool, ValidityStatus]:
        """检查策略是否有效"""
        if strategy_name is None:
            strategy_name = self.get_applied_strategy_name()
        return self.adjustment_config.validate_strategy_for_rollover(strategy_name, self)
    
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """获取所有可用策略及其状态"""
        available = []
        for name in self.adjustment_config.get_available_strategies():
            is_valid, status = self.is_strategy_valid(name)
            strategy = self.adjustment_config.get_strategy(name)
            available.append({
                "name": name,
                "is_valid": is_valid,
                "status": status,
                "is_applied": name == self.get_applied_strategy_name(),
                "is_custom": name == self.custom_strategy,
                "description": self._get_strategy_description(strategy)
            })
        return available
    
    def _get_strategy_description(self, strategy: AdjustmentStrategy) -> str:
        """获取策略描述"""
        if isinstance(strategy, PercentageAdjustmentStrategy):
            return "窗口百分比复权" if strategy.use_window else "单点百分比复权"
        elif isinstance(strategy, SpreadAdjustmentStrategy):
            return "窗口价差复权" if strategy.use_window else "单点价差复权"
        elif isinstance(strategy, WeightedAverageStrategy):
            return "加权平均复权"
        elif isinstance(strategy, ManualOverrideStrategy):
            return "手动覆盖复权"
        else:
            return "未知策略"
    
    # 原有的内部计算方法保持不变
    def _calculate_settlement_prices(self) -> Tuple[float, float]:
        if self._cached_settlement_prices is not None:
            return self._cached_settlement_prices
            
        window_start = self.rollover_datetime - self._get_window_duration()
        
        old_window_prices = self._get_prices_in_window(self.old_prices, window_start, self.rollover_datetime)
        new_window_prices = self._get_prices_in_window(self.new_prices, self.rollover_datetime, 
                                                      self.rollover_datetime + self._get_window_duration())
        
        old_settlement = self._calculate_period_settlement(old_window_prices, self.rollover_datetime)
        new_settlement = self._calculate_period_settlement(new_window_prices, self.rollover_datetime)
        
        self._cached_settlement_prices = (old_settlement, new_settlement)
        return self._cached_settlement_prices
    
    def _calculate_single_point_prices(self) -> Tuple[float, float]:
        if self._cached_single_point_prices is not None:
            return self._cached_single_point_prices
            
        old_near_price = self._get_price_near_time(self.old_prices, self.rollover_datetime)
        new_near_price = self._get_price_near_time(self.new_prices, self.rollover_datetime)
        
        self._cached_single_point_prices = (old_near_price, new_near_price)
        return self._cached_single_point_prices
    
    def _check_negative_prices_in_method(self, method: str) -> bool:
        if method == "percentage":
            old_price, new_price = self._calculate_single_point_prices()
            return old_price < 0 or new_price < 0
        elif method == "percentage_window":
            old_settlement, new_settlement = self._calculate_settlement_prices()
            return old_settlement < 0 or new_settlement < 0
        elif method == "spread":
            return False
        else:
            return False
    
    def has_negative_prices(self) -> bool:
        if self._cached_negative_check is not None:
            return self._cached_negative_check
            
        all_prices = [price for _, price in self.old_prices + self.new_prices]
        self._cached_negative_check = any(price < 0 for price in all_prices)
        return self._cached_negative_check
    
    # 原有的辅助方法保持不变
    def _get_window_duration(self) -> timedelta:
        if self.window_type == "time_based":
            return timedelta(hours=self.window_size_hours)
        else:
            if len(self.old_prices) > 1:
                avg_interval = (self.old_prices[-1][0] - self.old_prices[0][0]).total_seconds() / len(self.old_prices)
                window_seconds = avg_interval * self.window_size_ticks
                return timedelta(seconds=window_seconds)
            else:
                return timedelta(hours=1)
    
    def _get_prices_in_window(self, prices: List[Tuple[datetime, float]], 
                             start: datetime, end: datetime) -> List[Tuple[datetime, float]]:
        return [(ts, price) for ts, price in prices if start <= ts <= end]
    
    def _calculate_period_settlement(self, prices: List[Tuple[datetime, float]], 
                                   reference_time: datetime) -> float:
        if not prices:
            return 0.0
            
        period_end = reference_time
        period_start = period_end - timedelta(minutes=self.settlement_period_minutes)
        
        settlement_prices = [price for ts, price in prices if period_start <= ts <= period_end]
        
        if settlement_prices:
            return sum(settlement_prices) / len(settlement_prices)
        else:
            all_prices = [price for _, price in prices]
            return sum(all_prices) / len(all_prices) if all_prices else 0.0
    
    def _get_price_near_time(self, prices: List[Tuple[datetime, float]], 
                           target_time: datetime) -> float:
        if not prices:
            return 0.0
            
        closest_time = min(prices, key=lambda x: abs((x[0] - target_time).total_seconds()))
        return closest_time[1]
    
    def get_window_info(self) -> dict:
        old_settlement, new_settlement = self._calculate_settlement_prices()
        window_duration = self._get_window_duration()
        
        return {
            "window_type": self.window_type,
            "window_duration_hours": window_duration.total_seconds() / 3600,
            "old_settlement": old_settlement,
            "new_settlement": new_settlement,
            "old_prices_count": len(self.old_prices),
            "new_prices_count": len(self.new_prices),
            "settlement_period_minutes": self.settlement_period_minutes,
            "applied_strategy": self.get_applied_strategy_name(),
            "is_custom_strategy": self.custom_strategy is not None
        }
    

# 创建全局配置
global_config = RolloverAdjustmentConfig()

# 设置全局默认策略
global_config.set_global_default("percentage_window")
global_config.set_global_fallback("spread_window")

# 为特定合约设置策略（如原油合约使用价差复权）
global_config.set_contract_strategy("CL", "spread_window")

# 为特定时间范围设置策略（如2020年负油价期间使用加权平均）
covid_start = datetime(2020, 3, 1)
covid_end = datetime(2020, 6, 1)
global_config.set_time_based_strategy(covid_start, covid_end, "weighted_average")

# 设置条件策略（当检测到负价格时使用价差复权）
def has_negative_prices_condition(rollover):
    return rollover.has_negative_prices()

global_config.set_conditional_strategy(has_negative_prices_condition, "spread_window")

# 创建多个合约切换事件，使用相同的全局配置
rollover1 = ContractRollover(
    rollover_datetime=datetime(2020, 4, 20),
    old_contract="CLK0",
    new_contract="CLM0",
    old_prices=[(datetime(2020, 4, 20), -37.0)],  # 负价格
    new_prices=[(datetime(2020, 4, 20), 20.0)],
    adjustment_config=global_config
)

rollover2 = ContractRollover(
    rollover_datetime=datetime(2021, 1, 15),
    old_contract="CLZ0",
    new_contract="CLF1", 
    old_prices=[(datetime(2021, 1, 15), 50.0)],
    new_prices=[(datetime(2021, 1, 15), 52.0)],
    adjustment_config=global_config
)

# 检查应用的策略
print(f"Rollover1 策略: {rollover1.get_applied_strategy_name()}")  # spread_window（因为有负价格）
print(f"Rollover2 策略: {rollover2.get_applied_strategy_name()}")  # spread_window（因为是原油合约）

# 为特定切换事件设置个性化策略
rollover2.set_custom_strategy("percentage_window")
print(f"Rollover2 个性化策略: {rollover2.get_applied_strategy_name()}")  # percentage_window

# 查看所有可用策略
strategies = rollover1.get_available_strategies()
for strategy in strategies:
    status = "✓" if strategy["is_valid"] else "✗"
    applied = " [应用]" if strategy["is_applied"] else ""
    custom = " [自定义]" if strategy["is_custom"] else ""
    print(f"{status} {strategy['name']}: {strategy['description']}{applied}{custom}")

# 获取调整参数
adjustment1 = rollover1.get_adjustment_factor()
gap1 = rollover1.get_price_gap()
adjustment2 = rollover2.get_adjustment_factor() 
gap2 = rollover2.get_price_gap()

print(f"Rollover1 - 调整因子: {adjustment1}, 价差: {gap1}")
print(f"Rollover2 - 调整因子: {adjustment2}, 价差: {gap2}")
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings
from enum import Enum

@dataclass
class DataQualityIssue:
    """数据质量问题记录"""
    issue_type: str
    description: str
    timestamp: datetime
    contract: str
    severity: str  # low, medium, high
    action_taken: str

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
    
    def __init__(self, use_window: bool = False, window_size_hours: float = 2.0):
        self.use_window = use_window
        self.window_size_hours = window_size_hours
        self.name = "percentage_window" if use_window else "percentage"
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        if self.use_window:
            old_settlement, new_settlement = rollover._calculate_settlement_prices()
            print(f"  窗口百分比调整: 旧={old_settlement:.4f}, 新={new_settlement:.4f}")
            
            if old_settlement == 0:
                print(f"  警告: 旧合约结算价为0，无法计算百分比调整，使用1.0")
                return 1.0, 0.0
            
            adjustment = new_settlement / old_settlement
            print(f"  调整因子计算: {new_settlement:.4f} / {old_settlement:.4f} = {adjustment:.6f}")
            return adjustment, 0.0
        else:
            old_price, new_price = rollover._calculate_single_point_prices()
            print(f"  单点百分比调整: 旧={old_price:.4f}, 新={new_price:.4f}")
            
            if old_price == 0:
                print(f"  警告: 旧合约价格为0，无法计算百分比调整，使用1.0")
                return 1.0, 0.0
            
            adjustment = new_price / old_price
            print(f"  调整因子计算: {new_price:.4f} / {old_price:.4f} = {adjustment:.6f}")
            return adjustment, 0.0
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        if self.use_window:
            old_settlement, new_settlement = rollover._calculate_settlement_prices()
            print(f"  有效性检查 - 窗口百分比: 旧={old_settlement:.4f}, 新={new_settlement:.4f}")
            
            if old_settlement == 0:
                print(f"  无效原因: 旧合约结算价为0")
                return False, ValidityStatus.INVALID_ZERO_DIVISION
            
            # 检查负价格
            if old_settlement < 0 or new_settlement < 0:
                print(f"  无效原因: 存在负价格")
                return False, ValidityStatus.NEGATIVE_PRICES
        else:
            old_price, new_price = rollover._calculate_single_point_prices()
            print(f"  有效性检查 - 单点百分比: 旧={old_price:.4f}, 新={new_price:.4f}")
            
            if old_price == 0:
                print(f"  无效原因: 旧合约价格为0")
                return False, ValidityStatus.INVALID_ZERO_DIVISION
            
            # 检查负价格
            if old_price < 0 or new_price < 0:
                print(f"  无效原因: 存在负价格")
                return False, ValidityStatus.NEGATIVE_PRICES
        
        print(f"  策略有效")
        return True, ValidityStatus.VALID
    
    def get_name(self) -> str:
        return self.name

class SpreadAdjustmentStrategy(AdjustmentStrategy):
    """价差复权策略"""
    
    def __init__(self, use_window: bool = True, window_size_hours: float = 2.0):
        self.use_window = use_window
        self.window_size_hours = window_size_hours
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        if self.use_window:
            old_settlement, new_settlement = rollover._calculate_settlement_prices()
            gap = new_settlement - old_settlement
            return 1.0, gap
        else:
            old_price, new_price = rollover._calculate_single_point_prices()
            gap = new_price - old_price
            return 1.0, gap
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        # 价差复权几乎总是有效的，除非数据不足
        if len(rollover.old_prices) == 0 or len(rollover.new_prices) == 0:
            return False, ValidityStatus.INSUFFICIENT_DATA
        return True, ValidityStatus.VALID
    
    def get_name(self) -> str:
        return "spread_window" if self.use_window else "spread"

class WeightedAverageStrategy(AdjustmentStrategy):
    """加权平均复权策略 - 用于处理异常情况"""
    
    def __init__(self, volume_weights: Optional[Dict[str, float]] = None):
        self.volume_weights = volume_weights or {}
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        # 基于成交量加权的平均价格计算
        old_avg = self._calculate_weighted_average(rollover.old_prices)
        new_avg = self._calculate_weighted_average(rollover.new_prices)
        
        if old_avg == 0:
            return 1.0, 0.0
        
        adjustment = new_avg / old_avg
        gap = new_avg - old_avg
        
        return adjustment, gap
    
    def _calculate_weighted_average(self, prices: List[Tuple[datetime, float]]) -> float:
        if not prices:
            return 0.0
        
        # 如果有成交量权重，使用加权平均
        if self.volume_weights:
            total_weight = 0
            weighted_sum = 0
            for ts, price in prices:
                contract_key = ts.strftime("%Y%m")
                weight = self.volume_weights.get(contract_key, 1.0)
                weighted_sum += price * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
        
        # 否则使用简单平均
        all_prices = [price for _, price in prices]
        return sum(all_prices) / len(all_prices)
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        if len(rollover.old_prices) == 0 or len(rollover.new_prices) == 0:
            return False, ValidityStatus.INSUFFICIENT_DATA
        return True, ValidityStatus.VALID
    
    def get_name(self) -> str:
        return "weighted_average"

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

class ReturnCalculationStrategy(ABC):
    """收益率计算策略基类"""
    
    @abstractmethod
    def calculate_returns(self, prices: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
        """计算收益率序列"""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """获取策略名称"""
        pass

class SimpleReturnStrategy(ReturnCalculationStrategy):
    """简单收益率计算策略"""
    
    def calculate_returns(self, prices: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
        """计算简单收益率：(P_t - P_{t-1}) / P_{t-1}"""
        if len(prices) < 2:
            return []
        
        sorted_prices = sorted(prices, key=lambda x: x[0])
        returns = []
        
        for i in range(1, len(sorted_prices)):
            prev_time, prev_price = sorted_prices[i-1]
            curr_time, curr_price = sorted_prices[i]
            
            if prev_price != 0:
                ret = (curr_price - prev_price) / prev_price
            else:
                ret = 0.0
            
            returns.append((curr_time, ret))
        
        return returns
    
    def get_name(self) -> str:
        return "simple_returns"

class LogReturnStrategy(ReturnCalculationStrategy):
    """对数收益率计算策略"""
    
    def calculate_returns(self, prices: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
        """计算对数收益率：log(P_t / P_{t-1})"""
        if len(prices) < 2:
            return []
        
        sorted_prices = sorted(prices, key=lambda x: x[0])
        returns = []
        
        for i in range(1, len(sorted_prices)):
            prev_time, prev_price = sorted_prices[i-1]
            curr_time, curr_price = sorted_prices[i]
            
            if prev_price > 0 and curr_price > 0:
                ret = np.log(curr_price / prev_price)
            else:
                # 如果价格非正，回退到简单收益率
                if prev_price != 0:
                    ret = (curr_price - prev_price) / prev_price
                else:
                    ret = 0.0
            
            returns.append((curr_time, ret))
        
        return returns
    
    def get_name(self) -> str:
        return "log_returns"

# 现在让我们重新定义 RolloverAdjustmentConfig 和 ContractRollover 类

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
        self.register_strategy(ManualOverrideStrategy())

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
        
        print(f"  配置的策略: {configured_strategy}")
        
        # 检查配置的策略是否有效
        is_valid, status = self.validate_strategy_for_rollover(configured_strategy, rollover)
        
        if is_valid:
            print(f"  配置的策略有效，使用: {configured_strategy}")
            return configured_strategy
        
        print(f"  配置的策略无效: {status.value}")
        
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
        print(f"  回退策略也无效，尝试其他策略...")
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
    """合约切换事件的数据结构 - 使用DataFrame存储数据"""
    
    # 基础信息
    rollover_datetime: datetime  # 切换时间点（新合约开始的时间）
    old_contract: str
    new_contract: str
    
    # 关键时间点
    previous_day_end: Optional[datetime] = None  # 旧合约最后一个数据点时间
    current_day_start: Optional[datetime] = None  # 新合约第一个数据点时间
    
    # 完整的交易数据 - 使用DataFrame
    old_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    new_data: pd.DataFrame = field(default_factory=lambda: pd.DataFrame())
    
    # 配置参数
    window_type: str = "time_based"
    window_size_hours: float = 2.0
    window_size_ticks: int = 100
    settlement_period_minutes: int = 15
    
    # 复权配置
    adjustment_config: Optional[RolloverAdjustmentConfig] = None
    custom_strategy: Optional[str] = None
    
    # 缓存字段
    _cached_settlement_prices: Optional[Tuple[float, float]] = None
    _cached_single_point_prices: Optional[Tuple[float, float]] = None
    _cached_negative_check: Optional[bool] = None
    
    def __post_init__(self):
        """确保有配置对象并设置关键时间点"""
        if self.adjustment_config is None:
            self.adjustment_config = RolloverAdjustmentConfig()
        
        # 如果没有设置关键时间点，自动计算
        if self.previous_day_end is None and not self.old_data.empty:
            self.previous_day_end = self.old_data['datetime'].max()
        
        if self.current_day_start is None and not self.new_data.empty:
            self.current_day_start = self.new_data['datetime'].min()
        
        # 确保DataFrame有正确的列
        self._ensure_dataframe_columns()

    def _ensure_dataframe_columns(self):
        """确保DataFrame包含必要的列"""
        required_columns = ['datetime', 'open', 'high', 'low', 'close']
        
        for df_name in ['old_data', 'new_data']:
            df = getattr(self, df_name)
            if not df.empty:
                for col in required_columns:
                    if col not in df.columns:
                        warnings.warn(f"DataFrame {df_name} 缺少列: {col}")

    def _calculate_settlement_price(self, data: pd.DataFrame, 
                                reference_time: datetime, is_old: bool) -> float:
        """智能结算价计算 - 使用DataFrame"""
        if data.empty:
            print(f"      警告: 无数据可用")
            return 0.0
        
        # 获取结算期间的数据
        settlement_data = self._get_settlement_period_data(data, reference_time, is_old)
        
        if settlement_data.empty:
            print(f"      结算期间无数据，尝试替代方案...")
            # 尝试其他方法获取数据
            settlement_data = self._get_alternative_settlement_data(data, reference_time, is_old)
        
        if settlement_data.empty:
            print(f"      所有方法都无法获取数据，返回0")
            return 0.0
        
        # 检查成交量数据的质量
        total_volume = settlement_data['volume'].sum() if 'volume' in settlement_data.columns else 0
        unique_prices = settlement_data['close'].nunique()
        
        print(f"      结算数据: {len(settlement_data)}个点, 成交量={total_volume:.0f}, 价格数={unique_prices}")
        
        # 智能选择计算方法
        if total_volume > 10 and unique_prices >= 2:
            vwap = self._calculate_vwap(settlement_data)
            print(f"      使用VWAP: {vwap:.4f}")
            return vwap
        elif len(settlement_data) >= 3:
            twap = self._calculate_twap(settlement_data)
            print(f"      使用TWAP: {twap:.4f}")
            return twap
        else:
            avg_price = self._calculate_tick_weighted_price(settlement_data)
            print(f"      使用简单平均: {avg_price:.4f}")
            return avg_price

    def _get_settlement_period_data(self, data: pd.DataFrame, 
                                reference_time: datetime, is_old: bool) -> pd.DataFrame:
        """获取结算期间的数据 - 使用DataFrame"""
        period_end = reference_time
        period_start = period_end - timedelta(minutes=self.settlement_period_minutes)
        
        print(f"      结算期间: {period_start} 到 {period_end}")
        
        # 使用DataFrame的布尔索引
        mask = (data['datetime'] >= period_start) & (data['datetime'] <= period_end)
        settlement_data = data[mask]
        
        if not settlement_data.empty:
            print(f"      找到结算期间数据: {len(settlement_data)}个点")
            return settlement_data
        
        # 如果结算期间没有数据，尝试放宽条件
        print(f"      结算期间无数据，尝试放宽条件...")
        
        # 方法1: 扩大时间窗口
        expanded_start = period_start - timedelta(minutes=30)
        expanded_mask = (data['datetime'] >= expanded_start) & (data['datetime'] <= period_end)
        expanded_data = data[expanded_mask]
        
        if not expanded_data.empty:
            print(f"      扩大窗口后找到数据: {len(expanded_data)}个点")
            return expanded_data
        
        return pd.DataFrame()

    def _calculate_vwap(self, data: pd.DataFrame) -> float:
        """成交量加权平均价 - 使用DataFrame"""
        if data.empty:
            return 0.0
        
        if 'volume' not in data.columns or data['volume'].sum() == 0:
            print(f"      无成交量数据，回退到TWAP")
            return self._calculate_twap(data)
        
        # 使用典型价格 (high + low + close) / 3 作为VWAP计算的基础
        typical_price = (data['high'] + data['low'] + data['close']) / 3
        total_value = (typical_price * data['volume']).sum()
        total_volume = data['volume'].sum()
        
        vwap = total_value / total_volume
        print(f"      VWAP计算: {total_value:.2f} / {total_volume:.2f} = {vwap:.4f}")
        return vwap

    def _calculate_twap(self, data: pd.DataFrame) -> float:
        """时间加权平均价 - 使用DataFrame"""
        if len(data) < 2:
            return data['close'].iloc[0] if not data.empty else 0.0
        
        # 确保数据按时间排序
        data_sorted = data.sort_values('datetime')
        
        total_weighted_price = 0.0
        total_time = 0.0
        
        for i in range(1, len(data_sorted)):
            prev_row = data_sorted.iloc[i-1]
            curr_row = data_sorted.iloc[i]
            
            time_interval = (curr_row['datetime'] - prev_row['datetime']).total_seconds()
            # 使用前后两个时间点的平均价格
            avg_price = (prev_row['close'] + curr_row['close']) / 2
            
            total_weighted_price += avg_price * time_interval
            total_time += time_interval
        
        if total_time > 0:
            twap = total_weighted_price / total_time
            print(f"      TWAP计算: {total_weighted_price:.2f} / {total_time:.2f} = {twap:.4f}")
            return twap
        else:
            # 使用所有收盘价的简单平均
            avg_price = data['close'].mean()
            print(f"      时间间隔为0，使用简单平均: {avg_price:.4f}")
            return avg_price

    def _calculate_tick_weighted_price(self, data: pd.DataFrame) -> float:
        """交易次数加权（简单平均）"""
        if data.empty:
            return 0.0
        
        avg_price = data['close'].mean()
        print(f"      简单平均: {avg_price:.4f} (基于{len(data)}个价格)")
        return avg_price

    def _get_alternative_settlement_data(self, data: pd.DataFrame, 
                                    reference_time: datetime, is_old: bool) -> pd.DataFrame:
        """获取替代的结算数据 - 使用DataFrame"""
        if data.empty:
            return pd.DataFrame()
        
        if is_old:
            # 对于旧合约：使用最后几个数据点
            print(f"      旧合约使用最后数据点")
            # 取最后1小时的数据
            cutoff_time = reference_time - timedelta(hours=1)
            recent_data = data[data['datetime'] >= cutoff_time]
            
            if not recent_data.empty:
                # 取最后10个数据点或全部（取较少者）
                result = recent_data.tail(min(10, len(recent_data)))
                print(f"      使用最后{len(result)}个数据点")
                return result
            else:
                # 如果还是没有数据，使用所有可用数据
                print(f"      使用所有可用数据: {len(data)}个点")
                return data.tail(min(10, len(data)))
        else:
            # 对于新合约：使用前几个数据点
            print(f"      新合约使用前部数据点")
            # 取前2小时的数据
            cutoff_time = reference_time + timedelta(hours=2)
            early_data = data[data['datetime'] <= cutoff_time]
            
            if not early_data.empty:
                # 取前10个数据点或全部（取较少者）
                result = early_data.head(min(10, len(early_data)))
                print(f"      使用前{len(result)}个数据点")
                return result
            else:
                # 如果还是没有数据，使用所有可用数据
                print(f"      使用所有可用数据: {len(data)}个点")
                return data.head(min(10, len(data)))

    def _calculate_settlement_prices(self) -> Tuple[float, float]:
        """计算结算价 - 使用DataFrame"""
        if self._cached_settlement_prices is not None:
            return self._cached_settlement_prices
            
        print(f"  结算价计算:")
        
        # 使用正确的时间窗口
        if self.previous_day_end and not self.old_data.empty:
            # 旧合约：从最后一个数据点往前找窗口
            old_window_start = self.previous_day_end - self._get_window_duration()
            old_window_data = self._get_data_in_window(self.old_data, old_window_start, self.previous_day_end)
            print(f"    旧合约窗口: {old_window_start} 到 {self.previous_day_end}, {len(old_window_data)}个点")
        else:
            old_window_data = pd.DataFrame()
            print(f"    旧合约无previous_day_end时间点或无数据")
        
        if self.current_day_start and not self.new_data.empty:
            # 新合约：从第一个数据点往后找窗口
            new_window_end = self.current_day_start + self._get_window_duration()
            new_window_data = self._get_data_in_window(self.new_data, self.current_day_start, new_window_end)
            print(f"    新合约窗口: {self.current_day_start} 到 {new_window_end}, {len(new_window_data)}个点")
        else:
            new_window_data = pd.DataFrame()
            print(f"    新合约无current_day_start时间点或无数据")
        
        # 使用增强方法计算结算价
        old_settlement = self._calculate_settlement_price(old_window_data, self.previous_day_end, is_old=True) if self.previous_day_end else 0.0
        new_settlement = self._calculate_settlement_price(new_window_data, self.current_day_start, is_old=False) if self.current_day_start else 0.0
        
        # 如果旧合约结算价仍然为0，使用最后可用价格
        if old_settlement == 0 and not old_window_data.empty:
            print(f"    旧合约结算价仍为0，使用最后价格")
            old_settlement = old_window_data['close'].iloc[-1]
            print(f"    使用最后价格: {old_settlement:.4f}")
        
        # 如果新合约结算价为0，使用第一个可用价格
        if new_settlement == 0 and not new_window_data.empty:
            print(f"    新合约结算价为0，使用第一个价格")
            new_settlement = new_window_data['close'].iloc[0]
            print(f"    使用第一个价格: {new_settlement:.4f}")
        
        print(f"    最终结算价: 旧={old_settlement:.4f}, 新={new_settlement:.4f}")
        
        self._cached_settlement_prices = (old_settlement, new_settlement)
        return self._cached_settlement_prices

    def _get_data_in_window(self, data: pd.DataFrame, 
                        start: datetime, end: datetime) -> pd.DataFrame:
        """获取窗口内的完整数据 - 使用DataFrame"""
        if data.empty:
            return pd.DataFrame()
        
        # 确保时间窗口合理
        if start > end:
            start, end = end, start
        
        window_mask = (data['datetime'] >= start) & (data['datetime'] <= end)
        window_data = data[window_mask]
        
        # 如果窗口内数据太少，适当扩大窗口
        if len(window_data) < 5 and len(data) > 0:
            print(f"      窗口内数据较少({len(window_data)}个点)，适当扩大窗口")
            # 扩大窗口到包含更多数据
            expanded_start = data['datetime'].min()
            expanded_end = data['datetime'].max()
            expanded_mask = (data['datetime'] >= expanded_start) & (data['datetime'] <= expanded_end)
            window_data = data[expanded_mask]
        
        return window_data

    # 更新其他辅助方法
    def _calculate_single_point_prices(self) -> Tuple[float, float]:
        """计算单点价格（用于非窗口策略）"""
        if self._cached_single_point_prices is not None:
            return self._cached_single_point_prices
            
        # 从DataFrame中提取价格
        old_price = self._get_price_near_time(self.old_data, self.rollover_datetime)
        new_price = self._get_price_near_time(self.new_data, self.rollover_datetime)
        
        self._cached_single_point_prices = (old_price, new_price)
        return self._cached_single_point_prices
    
    def _get_price_near_time(self, data: pd.DataFrame, target_time: datetime) -> float:
        """获取最接近目标时间的价格 - 使用DataFrame"""
        if data.empty:
            return 0.0
            
        # 计算时间差
        time_diffs = (data['datetime'] - target_time).abs()
        closest_idx = time_diffs.idxmin()
        return data.loc[closest_idx, 'close']

    def has_negative_prices(self) -> bool:
        """检查是否有负价格"""
        if self._cached_negative_check is not None:
            return self._cached_negative_check
            
        # 检查所有价格列
        price_columns = ['open', 'high', 'low', 'close']
        all_negative = False
        
        for df in [self.old_data, self.new_data]:
            if not df.empty:
                for col in price_columns:
                    if col in df.columns and (df[col] < 0).any():
                        all_negative = True
                        break
        
        self._cached_negative_check = all_negative
        return self._cached_negative_check

    def get_data_summary(self) -> Dict[str, Any]:
        """获取数据摘要"""
        return {
            'old_contract': {
                'data_points': len(self.old_data),
                'time_range': (self.old_data['datetime'].min(), self.old_data['datetime'].max()) if not self.old_data.empty else (None, None),
                'columns': list(self.old_data.columns) if not self.old_data.empty else []
            },
            'new_contract': {
                'data_points': len(self.new_data),
                'time_range': (self.new_data['datetime'].min(), self.new_data['datetime'].max()) if not self.new_data.empty else (None, None),
                'columns': list(self.new_data.columns) if not self.new_data.empty else []
            }
        }

    # def _get_data_in_window(self, data: List[Tuple[datetime, float, float, float]], 
    #                     start: datetime, end: datetime) -> List[Tuple[datetime, float, float, float]]:
    #     """获取窗口内的完整数据"""
    #     if not data:
    #         return []
        
    #     # 确保时间窗口合理
    #     if start > end:
    #         start, end = end, start
        
    #     window_data = [(ts, price, volume, position) for ts, price, volume, position in data 
    #                 if start <= ts <= end]
        
    #     # 如果窗口内数据太少，适当扩大窗口
    #     if len(window_data) < 5 and len(data) > 0:
    #         print(f"      窗口内数据较少({len(window_data)}个点)，适当扩大窗口")
    #         # 扩大窗口到包含更多数据
    #         all_times = [ts for ts, _, _, _ in data]
    #         if all_times:
    #             expanded_start = min(all_times)
    #             expanded_end = max(all_times)
    #             window_data = [(ts, price, volume, position) for ts, price, volume, position in data 
    #                         if expanded_start <= ts <= expanded_end]
        
    #     return window_data
    
    def set_custom_strategy(self, strategy_name: str):
        """设置个性化策略（覆盖配置）"""
        if self.adjustment_config is None:
            self.adjustment_config = RolloverAdjustmentConfig()
        if strategy_name not in self.adjustment_config.registered_strategies:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        self.custom_strategy = strategy_name
    
    def clear_custom_strategy(self):
        """清除个性化策略（使用配置）"""
        self.custom_strategy = None
    
    def get_applied_strategy_name(self) -> str:
        """获取实际应用的策略名称"""
        if self.adjustment_config is None:
            self.adjustment_config = RolloverAdjustmentConfig()
        if self.custom_strategy:
            return self.custom_strategy
        return self.adjustment_config.get_recommended_strategy(self)
    
    def get_applied_strategy(self) -> AdjustmentStrategy:
        """获取实际应用的策略实例"""
        if self.adjustment_config is None:
            self.adjustment_config = RolloverAdjustmentConfig()
        strategy_name = self.get_applied_strategy_name()
        strategy = self.adjustment_config.get_strategy(strategy_name)
        if strategy is None:
            raise ValueError(f"策略 '{strategy_name}' 未注册")
        return strategy
    
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
        if self.adjustment_config is None:
            self.adjustment_config = RolloverAdjustmentConfig()
        if strategy_name is None:
            strategy_name = self.get_applied_strategy_name()
        return self.adjustment_config.validate_strategy_for_rollover(strategy_name, self)
    
    def get_available_strategies(self) -> List[Dict[str, Any]]:
        """获取所有可用策略及其状态"""
        if self.adjustment_config is None:
            self.adjustment_config = RolloverAdjustmentConfig()
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
    
    def _get_strategy_description(self, strategy: Optional[AdjustmentStrategy]) -> str:
        """获取策略描述"""
        if strategy is None:
            return "未知策略"
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
    
    # def _calculate_single_point_prices(self) -> Tuple[float, float]:
    #     """计算单点价格（用于非窗口策略）"""
    #     if self._cached_single_point_prices is not None:
    #         return self._cached_single_point_prices
            
    #     # 从完整数据中提取价格
    #     old_price = self._get_price_near_time(self.old_prices, self.rollover_datetime)
    #     new_price = self._get_price_near_time(self.new_prices, self.rollover_datetime)
        
    #     self._cached_single_point_prices = (old_price, new_price)
    #     return self._cached_single_point_prices
    
    def _check_negative_prices_in_method(self, method: str) -> bool:
        """检查特定方法中是否有负价格"""
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
    
    # def has_negative_prices(self) -> bool:
    #     """检查是否有负价格"""
    #     if self._cached_negative_check is not None:
    #         return self._cached_negative_check
            
    #     # 从完整数据中提取价格
    #     all_prices = [price for _, price, _, _ in self.old_prices + self.new_prices]
    #     self._cached_negative_check = any(price < 0 for price in all_prices)
    #     return self._cached_negative_check
    
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
            print(f"      警告: 无价格数据用于结算价计算")
            return 0.0
            
        period_end = reference_time
        period_start = period_end - timedelta(minutes=self.settlement_period_minutes)
        
        settlement_prices = [price for ts, price in prices if period_start <= ts <= period_end]
        
        print(f"      结算期间: {period_start} 到 {period_end}")
        print(f"      期间内数据点: {len(settlement_prices)}")
        
        if settlement_prices:
            result = sum(settlement_prices) / len(settlement_prices)
            print(f"      使用期间平均: {result:.4f}")
            return result
        else:
            # 如果指定期间内没有数据，使用所有数据的平均值
            all_prices = [price for _, price in prices]
            if all_prices:
                result = sum(all_prices) / len(all_prices)
                print(f"      使用全部平均: {result:.4f}")
                return result
            else:
                print(f"      警告: 完全无价格数据")
                return 0.0
    
    # def _get_price_near_time(self, data: List[Tuple[datetime, float, float, float]], 
    #                     target_time: datetime) -> float:
    #     """获取最接近目标时间的价格"""
    #     if not data:
    #         return 0.0
            
    #     # 从完整数据中提取时间戳和价格
    #     price_data = [(ts, price) for ts, price, volume, position in data]
    #     closest_time = min(price_data, key=lambda x: abs((x[0] - target_time).total_seconds()))
    #     return closest_time[1]
    
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
    
class DirectRolloverReturnCalculator:
    """直接复权收益率计算器 - 处理多个合约切换事件"""
    
    def __init__(self, adjustment_config: Optional[RolloverAdjustmentConfig] = None):
        self.adjustment_config = adjustment_config or RolloverAdjustmentConfig()
        self.return_strategy: ReturnCalculationStrategy = LogReturnStrategy()
        self.adjustment_cache: Dict[datetime, Tuple[float, float]] = {}
    
    def set_return_strategy(self, strategy: ReturnCalculationStrategy):
        """设置收益率计算策略"""
        self.return_strategy = strategy
    
    def calculate_returns_with_rollovers(self, 
                                       price_data: List[Tuple[datetime, float, str]], 
                                       rollover_events: List[ContractRollover]) -> List[Tuple[datetime, float]]:
        """
        计算考虑合约切换的收益率序列
        
        Args:
            price_data: 价格数据列表 (时间戳, 价格, 合约代码)
            rollover_events: 合约切换事件列表
            
        Returns:
            收益率序列 (时间戳, 收益率)
        """
        # 按合约分组价格数据
        contract_prices = self._group_prices_by_contract(price_data)
        
        # 构建连续价格序列
        continuous_prices = self._build_continuous_prices(contract_prices, rollover_events)
        
        # 计算收益率
        returns = self.return_strategy.calculate_returns(continuous_prices)
        
        return returns
    
    def _group_prices_by_contract(self, price_data: List[Tuple[datetime, float, str]]) -> Dict[str, List[Tuple[datetime, float]]]:
        """按合约分组价格数据"""
        contract_prices = {}
        for timestamp, price, contract in price_data:
            if contract not in contract_prices:
                contract_prices[contract] = []
            contract_prices[contract].append((timestamp, price))
        
        # 对每个合约的价格按时间排序
        for contract in contract_prices:
            contract_prices[contract].sort(key=lambda x: x[0])
        
        return contract_prices
    
    def _build_continuous_prices(self, 
                               contract_prices: Dict[str, List[Tuple[datetime, float]]], 
                               rollover_events: List[ContractRollover]) -> List[Tuple[datetime, float]]:
        """构建连续价格序列"""
        if not rollover_events:
            # 如果没有切换事件，直接使用第一个合约的价格
            first_contract = next(iter(contract_prices.keys()))
            return contract_prices[first_contract]
        
        # 按时间排序切换事件
        sorted_rollovers = sorted(rollover_events, key=lambda x: x.rollover_datetime)
        
        # 构建连续价格序列
        continuous_prices = []
        cumulative_adjustment = 1.0
        cumulative_gap = 0.0
        
        # 处理第一个合约（切换前）
        first_contract = sorted_rollovers[0].old_contract
        if first_contract in contract_prices:
            for timestamp, price in contract_prices[first_contract]:
                if timestamp <= sorted_rollovers[0].rollover_datetime:
                    continuous_prices.append((timestamp, price))
        
        # 处理每个切换事件
        for i, rollover in enumerate(sorted_rollovers):
            # 为切换事件设置配置
            rollover.adjustment_config = self.adjustment_config
            
            # 计算调整参数
            adjustment_factor = rollover.get_adjustment_factor()
            price_gap = rollover.get_price_gap()
            
            # 更新累积调整
            cumulative_adjustment *= adjustment_factor
            cumulative_gap = cumulative_gap * adjustment_factor + price_gap
            
            # 缓存调整参数
            self.adjustment_cache[rollover.rollover_datetime] = (cumulative_adjustment, cumulative_gap)
            
            # 添加新合约的价格（应用调整）
            new_contract = rollover.new_contract
            if new_contract in contract_prices:
                for timestamp, price in contract_prices[new_contract]:
                    if timestamp >= rollover.rollover_datetime:
                        # 应用累积调整
                        adjusted_price = price * cumulative_adjustment + cumulative_gap
                        continuous_prices.append((timestamp, adjusted_price))
        
        # 按时间排序连续价格序列
        continuous_prices.sort(key=lambda x: x[0])
        
        return continuous_prices
    
    def get_adjustment_history(self) -> List[Dict[str, Any]]:
        """获取调整历史"""
        history = []
        for timestamp, (adjustment, gap) in sorted(self.adjustment_cache.items()):
            history.append({
                'timestamp': timestamp,
                'cumulative_adjustment': adjustment,
                'cumulative_gap': gap
            })
        return history
    
    def calculate_statistics(self, returns: List[Tuple[datetime, float]]) -> Dict[str, Any]:
        """计算收益率统计信息"""
        if not returns:
            return {}
        
        return_values = [ret for _, ret in returns]
        
        return {
            'count': len(returns),
            'mean': np.mean(return_values),
            'std': np.std(return_values),
            'min': min(return_values),
            'max': max(return_values),
            'sharpe_ratio': np.mean(return_values) / np.std(return_values) if np.std(return_values) > 0 else 0,
            'total_return': np.prod([1 + ret for ret in return_values]) - 1
        }
    
class RolloverDetector:
    """合约切换点检测器"""
    
    def __init__(self, 
                 min_volume: float = 0, 
                 price_change_threshold: float = 0.5,
                 adjustment_config: Optional[RolloverAdjustmentConfig] = None,
                 window_size_hours: float = 2.0,
                 settlement_period_minutes: int = 15):
        self.min_volume = min_volume
        self.price_change_threshold = price_change_threshold  # 价格变化阈值，用于验证切换点
        self.adjustment_config = adjustment_config or RolloverAdjustmentConfig()
        self.window_size_hours = window_size_hours
        self.settlement_period_minutes = settlement_period_minutes
    
    def detect_rollover_points(self, data: pd.DataFrame) -> List[ContractRollover]:
        """检测合约切换点"""
        if data is None:
            raise ValueError("数据未加载")
            
        # 按时间排序
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # 检测symbol变化点
        data['symbol_change'] = data['symbol'] != data['symbol'].shift(1)
        change_indices = data[data['symbol_change']].index.tolist()
        
        # 移除第一个点（可能是数据开始）
        if change_indices and change_indices[0] == 0:
            change_indices = change_indices[1:]
        
        print(f"初步检测到 {len(change_indices)} 个合约切换点")
        print(f"数据时间范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
        
        # 构建切换事件并进行验证
        rollover_points = []
        for idx in change_indices:
            try:
                # 获取切换时间点
                rollover_time = data.iloc[idx]['datetime']
                old_contract = data.iloc[idx - 1]['symbol']
                new_contract = data.iloc[idx]['symbol']
                
                print(f"\n处理切换点 {rollover_time}: {old_contract} -> {new_contract}")
                
                # 构建价格窗口数据
                old_prices = self._get_price_window(data, old_contract, rollover_time, is_old=True)
                new_prices = self._get_price_window(data, new_contract, rollover_time, is_old=False)
                
                # 如果数据不足，尝试扩大窗口
                if len(old_prices) < 3:
                    print(f"  旧合约数据不足，尝试扩大窗口...")
                    # 扩大窗口到4小时
                    self.window_size_hours = 4.0
                    old_prices = self._get_price_window(data, old_contract, rollover_time, is_old=True)
                    # 恢复原窗口大小
                    self.window_size_hours = 2.0
                
                if len(new_prices) < 3:
                    print(f"  新合约数据不足，尝试扩大窗口...")
                    # 扩大窗口到4小时
                    self.window_size_hours = 4.0
                    new_prices = self._get_price_window(data, new_contract, rollover_time, is_old=False)
                    # 恢复原窗口大小
                    self.window_size_hours = 2.0
                
                # 创建切换事件
                rollover = ContractRollover(
                    rollover_datetime=rollover_time,
                    old_contract=old_contract,
                    new_contract=new_contract,
                    old_prices=old_prices,  # 现在是包含完整数据的列表
                    new_prices=new_prices,  # 现在是包含完整数据的列表
                    window_type="time_based",
                    window_size_hours=self.window_size_hours,
                    settlement_period_minutes=self.settlement_period_minutes,
                    adjustment_config=self.adjustment_config
                )
                
                # 验证切换点
                is_valid = self._validate_rollover(rollover, data, idx)
                
                if is_valid:
                    rollover_points.append(rollover)
                    print(f"✓ 有效切换点 {rollover.rollover_datetime}: {rollover.old_contract} -> {rollover.new_contract}")
                else:
                    print(f"✗ 可疑切换点 {rollover.rollover_datetime}: {rollover.old_contract} -> {rollover.new_contract} - 跳过")
                    
            except Exception as e:
                print(f"处理切换点 {idx} 时出错: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        print(f"\n最终有效切换点: {len(rollover_points)}/{len(change_indices)}")
        
        return rollover_points
    
    def detect_futures_rollovers(self, data: pd.DataFrame) -> List[ContractRollover]:
        """专门针对期货合约的切换点检测"""
        
        data_sorted = data.sort_values('datetime')
        data_sorted['contract_change'] = data_sorted['symbol'] != data_sorted['symbol'].shift(1)
        change_points = data_sorted[data_sorted['contract_change']]
        
        print(f"找到 {len(change_points)} 个合约变化点")
        
        rollover_events = []
        
        for idx, change_row in change_points.iterrows():
            try:
                change_time = change_row['datetime']
                new_contract = change_row['symbol']
                
                # 找到前一个合约
                prev_data = data_sorted[data_sorted['datetime'] < change_time]
                if len(prev_data) == 0:
                    continue
                    
                old_contract = prev_data.iloc[-1]['symbol']
                
                print(f"\n处理切换: {old_contract} -> {new_contract} 在 {change_time}")
                
                # 获取价格数据
                old_prices, previous_day_end = self._get_futures_settlement_price(data, old_contract, change_time)
                new_prices, current_day_start = self._get_futures_opening_price(data, new_contract, change_time)
                
                # 创建切换事件
                rollover = ContractRollover(
                    rollover_datetime=change_time,
                    old_contract=old_contract,
                    new_contract=new_contract,
                    previous_day_end=previous_day_end,
                    current_day_start=current_day_start,
                    old_prices=old_prices,
                    new_prices=new_prices,
                    window_type="time_based",
                    window_size_hours=self.window_size_hours,
                    settlement_period_minutes=self.settlement_period_minutes,
                    adjustment_config=self.adjustment_config
                )
                
                rollover_events.append(rollover)
                print(f"✓ 创建切换事件: 旧合约结束于 {previous_day_end}, 新合约开始于 {current_day_start}")
                
            except Exception as e:
                print(f"处理切换点时出错: {e}")
                continue
        
        return rollover_events

    def _get_futures_settlement_price(self, data: pd.DataFrame, contract: str, rollover_time: datetime) -> Tuple[List[Tuple[datetime, float, float, float]], Optional[datetime]]:
        """获取期货合约的结算价数据并返回最后一个数据点时间"""
        contract_data = data[data['symbol'] == contract]
        
        # 找切换时间前最后一个交易日
        previous_data = contract_data[contract_data['datetime'] < rollover_time]
        if len(previous_data) == 0:
            return [], None
        
        last_trading_day = previous_data['datetime'].max().date()
        last_day_data = contract_data[contract_data['datetime'].dt.date == last_trading_day]
        
        if len(last_day_data) == 0:
            return [], None
        
        # 获取最后一个数据点时间
        previous_day_end = last_day_data['datetime'].max()
        
        # 取最后2小时的数据作为窗口
        window_start = previous_day_end - timedelta(hours=2)
        window_data = last_day_data[last_day_data['datetime'] >= window_start]
        
        # 返回完整数据
        result = []
        for _, row in window_data.iterrows():
            close = row['close'] if pd.notna(row['close']) else 0.0
            volume = row['volume'] if 'volume' in row and pd.notna(row['volume']) else 0.0
            position = row['position'] if 'position' in row and pd.notna(row['position']) else 0.0
            
            result.append((row['datetime'], close, volume, position))
        
        print(f"  旧合约 {contract}: 最后交易日 {last_trading_day}, 结束时间 {previous_day_end}, {len(result)}个数据点")
        return result, previous_day_end

    def _get_futures_opening_price(self, data: pd.DataFrame, contract: str, rollover_time: datetime) -> Tuple[List[Tuple[datetime, float, float, float]], Optional[datetime]]:
        """获取期货合约的开盘价数据并返回第一个数据点时间"""
        contract_data = data[data['symbol'] == contract]
        
        # 找切换时间当天的数据
        current_day_data = contract_data[contract_data['datetime'].dt.date == rollover_time.date()]
        
        if len(current_day_data) == 0:
            # 如果当天没有数据，找之后第一个交易日
            future_data = contract_data[contract_data['datetime'] > rollover_time]
            if len(future_data) == 0:
                return [], None
            first_trading_day = future_data['datetime'].min().date()
            current_day_data = contract_data[contract_data['datetime'].dt.date == first_trading_day]
        
        if len(current_day_data) == 0:
            return [], None
        
        # 获取第一个数据点时间
        current_day_start = current_day_data['datetime'].min()
        
        # 取前2小时的数据作为窗口
        window_end = current_day_start + timedelta(hours=2)
        window_data = current_day_data[current_day_data['datetime'] <= window_end]
        
        # 返回完整数据
        result = []
        for _, row in window_data.iterrows():
            close = row['close'] if pd.notna(row['close']) else 0.0
            volume = row['volume'] if 'volume' in row and pd.notna(row['volume']) else 0.0
            position = row['position'] if 'position' in row and pd.notna(row['position']) else 0.0
            
            result.append((row['datetime'], close, volume, position))
        
        print(f"  新合约 {contract}: 开始时间 {current_day_start}, {len(result)}个数据点")
        return result, current_day_start

    # 在 RolloverDetector 中修改数据获取
    def _get_price_window(self, data: pd.DataFrame, contract: str, rollover_time: datetime, is_old: bool) -> List[Tuple[datetime, float, float, float]]:
        """
        获取切换点附近的完整交易数据 - 修正版本
        
        Returns:
            List of (datetime, close, volume, position)
        """
        # 筛选指定合约的数据
        contract_data = data[data['symbol'] == contract].copy()
        
        if len(contract_data) == 0:
            print(f"  警告: 合约 {contract} 在数据中不存在")
            return []
        
        if is_old:
            # 旧合约：获取切换时间前一天的数据
            previous_day = rollover_time - timedelta(days=1)
            previous_day_start = datetime(previous_day.year, previous_day.month, previous_day.day, 0, 0, 0)
            previous_day_end = datetime(previous_day.year, previous_day.month, previous_day.day, 23, 59, 59)
            
            window_data = contract_data[
                (contract_data['datetime'] >= previous_day_start) & 
                (contract_data['datetime'] <= previous_day_end)
            ]
            
            print(f"  旧合约 {contract}: 前一天 {previous_day.date()} 的数据")
            print(f"  时间范围 {previous_day_start} 到 {previous_day_end}")
            print(f"  找到 {len(window_data)} 个数据点")
            
            # 如果前一天没有数据，尝试往前找
            if len(window_data) == 0:
                print(f"  前一天无数据，尝试寻找最近的数据...")
                # 找切换时间之前最近的几个交易日的数据
                window_data = contract_data[contract_data['datetime'] < rollover_time].tail(100)  # 取最近100个点
            
        else:
            # 新合约：取切换时间当天的数据
            current_day_start = datetime(rollover_time.year, rollover_time.month, rollover_time.day, 0, 0, 0)
            current_day_end = datetime(rollover_time.year, rollover_time.month, rollover_time.day, 23, 59, 59)
            
            window_data = contract_data[
                (contract_data['datetime'] >= current_day_start) & 
                (contract_data['datetime'] <= current_day_end)
            ]
            
            print(f"  新合约 {contract}: 当天 {rollover_time.date()} 的数据")
            print(f"  时间范围 {current_day_start} 到 {current_day_end}")
            print(f"  找到 {len(window_data)} 个数据点")
            
            # 如果当天没有数据，尝试往后找
            if len(window_data) == 0:
                print(f"  当天无数据，尝试寻找后续的数据...")
                # 找切换时间之后最近的数据
                window_data = contract_data[contract_data['datetime'] > rollover_time].head(100)  # 取最近100个点
        
        # 返回完整数据 (时间, 收盘价, 成交量, 持仓量)
        result = []
        for _, row in window_data.iterrows():
            close = row['close'] if pd.notna(row['close']) else 0.0
            volume = row['volume'] if 'volume' in row and pd.notna(row['volume']) else 0.0
            position = row['position'] if 'position' in row and pd.notna(row['position']) else 0.0
            
            result.append((row['datetime'], close, volume, position))
        
        print(f"  最终获取: {contract} {len(result)}个数据点")
        return result
    
    def _validate_rollover(self, rollover: ContractRollover, data: pd.DataFrame, change_index: int) -> bool:
        """
        验证切换点是否有效
        
        Args:
            rollover: 切换事件
            data: 完整数据
            change_index: 变化点的索引
            
        Returns:
            是否有效
        """
        # 1. 检查数据量是否足够（对于日内数据，有几个点就够）
        if len(rollover.old_prices) == 0 or len(rollover.new_prices) == 0:
            print(f"  数据完全缺失: 旧合约{len(rollover.old_prices)}个点, 新合约{len(rollover.new_prices)}个点")
            return False
        
        # 2. 对于期货合约切换，价格可能会有较大跳空，这是正常的
        try:
            # 获取切换前后的价格
            old_row = data.iloc[change_index - 1]
            new_row = data.iloc[change_index]
            
            price_change = abs(new_row['close'] - old_row['close']) / old_row['close'] if old_row['close'] > 0 else float('inf')
            
            # 放宽价格变化阈值到200%，因为期货合约切换可能有较大价差
            if price_change > 2.0:  # 200%
                print(f"  价格跳空较大: {price_change:.2%}，但期货切换可能正常")
                # 不因为价格跳空而拒绝，只是警告
            
        except Exception as e:
            print(f"  价格连续性检查失败: {e}")
            # 不因为检查失败而拒绝
        
        # 3. 检查策略有效性
        try:
            is_valid, status = rollover.is_strategy_valid()
            if not is_valid:
                print(f"  策略状态: {status.value}")
                # 即使策略无效也接受，使用默认策略
        except Exception as e:
            print(f"  策略验证失败: {e}")
        
        # 主要检查是否有基本数据
        if len(rollover.old_prices) >= 1 and len(rollover.new_prices) >= 1:
            print(f"  ✓ 基本数据满足: 旧合约{len(rollover.old_prices)}点, 新合约{len(rollover.new_prices)}点")
            return True
        else:
            print(f"  ✗ 基本数据不足")
            return False
    
    def analyze_rollover_events(self, rollover_events: List[ContractRollover]) -> pd.DataFrame:
        """
        分析切换事件统计信息
        
        Args:
            rollover_events: 切换事件列表
            
        Returns:
            统计分析结果
        """
        if not rollover_events:
            return pd.DataFrame()
        
        analysis_data = []
        for i, event in enumerate(rollover_events):
            # 获取窗口信息
            window_info = event.get_window_info()
            
            # 获取策略信息
            strategy_name = event.get_applied_strategy_name()
            adjustment = event.get_adjustment_factor()
            gap = event.get_price_gap()
            
            # 验证状态
            is_valid, status = event.is_strategy_valid()
            
            analysis_data.append({
                'event_id': i + 1,
                'datetime': event.rollover_datetime,
                'old_contract': event.old_contract,
                'new_contract': event.new_contract,
                'strategy': strategy_name,
                'adjustment_factor': adjustment,
                'price_gap': gap,
                'is_valid': is_valid,
                'validity_status': status.value,
                'old_prices_count': len(event.old_prices),
                'new_prices_count': len(event.new_prices),
                'old_settlement': window_info['old_settlement'],
                'new_settlement': window_info['new_settlement'],
                'has_negative_prices': event.has_negative_prices()
            })
        
        return pd.DataFrame(analysis_data)
    
    def detect_with_advanced_validation(self, data: pd.DataFrame, 
                                      volume_threshold: float = 1000,
                                      gap_threshold: float = 0.1) -> List[ContractRollover]:
        """
        使用高级验证检测切换点
        
        Args:
            data: 价格数据
            volume_threshold: 成交量阈值
            gap_threshold: 价格差距阈值
            
        Returns:
            验证通过的切换事件列表
        """
        # 基础检测
        # basic_rollovers = self.detect_rollover_points(data)
        basic_rollovers = self.detect_futures_rollovers(data)
        
        # 高级验证
        validated_rollovers = []
        for rollover in basic_rollovers:
            if self._advanced_validation(rollover, data, volume_threshold, gap_threshold):
                validated_rollovers.append(rollover)
        
        print(f"高级验证后剩余切换点: {len(validated_rollovers)}/{len(basic_rollovers)}")
        return validated_rollovers
    
    def _advanced_validation(self, rollover: ContractRollover, data: pd.DataFrame,
                           volume_threshold: float, gap_threshold: float) -> bool:
        """高级验证逻辑"""
        # 1. 检查价格稳定性
        # rollover.old_prices 和 rollover.new_prices 中的每一项为 (datetime, price, volume, position)
        old_prices = [price for _, price, _, _ in rollover.old_prices]
        new_prices = [price for _, price, _, _ in rollover.new_prices]
        
        if len(old_prices) > 1 and len(new_prices) > 1:
            old_volatility = np.std(old_prices) / np.mean(old_prices) if np.mean(old_prices) > 0 else float('inf')
            new_volatility = np.std(new_prices) / np.mean(new_prices) if np.mean(new_prices) > 0 else float('inf')
            
            if old_volatility > 0.1 or new_volatility > 0.1:  # 10% 波动率阈值
                print(f"  价格波动过大: 旧合约{old_volatility:.2%}, 新合约{new_volatility:.2%}")
                return False
        
        # 2. 检查成交量（如果可用）
        if 'volume' in data.columns:
            # 获取切换点附近的成交量
            rollover_time = rollover.rollover_datetime
            time_window = timedelta(hours=1)
            
            old_volume_data = data[
                (data['symbol'] == rollover.old_contract) & 
                (data['datetime'] >= rollover_time - time_window) & 
                (data['datetime'] <= rollover_time)
            ]
            
            new_volume_data = data[
                (data['symbol'] == rollover.new_contract) & 
                (data['datetime'] >= rollover_time) & 
                (data['datetime'] <= rollover_time + time_window)
            ]
            
            if len(old_volume_data) > 0 and len(new_volume_data) > 0:
                old_avg_volume = old_volume_data['volume'].mean()
                new_avg_volume = new_volume_data['volume'].mean()
                
                if old_avg_volume < volume_threshold or new_avg_volume < volume_threshold:
                    print(f"  平均成交量不足: 旧合约{old_avg_volume:.0f}, 新合约{new_avg_volume:.0f}")
                    return False
        
        return True

class DataQualityChecker:
    """数据质量检查器 - 考虑合约切换点"""
    
    def __init__(self, rollover_points: Optional[List[ContractRollover]] = None):
        self.issues = []
        self.rollover_points = rollover_points or []
        self.rollover_times = [rp.rollover_datetime for rp in self.rollover_points]## if rp.is_valid]
    
    def check_missing_values(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """检查缺失值"""
        issues = []
        
        # 检查整体缺失情况
        missing_stats = data.isnull().sum()
        total_rows = len(data)
        
        for column, missing_count in missing_stats.items():
            if missing_count > 0:
                missing_pct = (missing_count / total_rows) * 100
                severity = "high" if missing_pct > 5 else "medium" if missing_pct > 1 else "low"
                
                issue = DataQualityIssue(
                    issue_type="missing_value",
                    description=f"字段 {column} 有 {missing_count} 个缺失值 ({missing_pct:.2f}%)",
                    timestamp=data['datetime'].min() if 'datetime' in data.columns else datetime.now(),
                    contract="all",
                    severity=severity,
                    action_taken="待处理"
                )
                issues.append(issue)
        
        # 检查时间连续性 - 确保datetime列是datetime类型
        if 'datetime' in data.columns:
            # 确保datetime列是datetime类型
            if data['datetime'].dtype == 'object':
                try:
                    data = data.copy()
                    data['datetime'] = pd.to_datetime(data['datetime'], format='%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"无法转换datetime列: {e}")
                    return issues
            
            # 现在可以安全地进行时间差计算
            time_gaps = data['datetime'].diff()
            # 寻找异常大的时间间隔（排除正常交易间隔）
            normal_trading_hours_mask = self._get_normal_trading_hours_mask(data['datetime'])
            abnormal_gaps = time_gaps[~normal_trading_hours_mask & (time_gaps > timedelta(hours=4))]
            
            for gap in abnormal_gaps.unique():
                count = (time_gaps == gap).sum()
                issue = DataQualityIssue(
                    issue_type="time_gap",
                    description=f"异常时间间隔: {gap}, 出现 {count} 次",
                    timestamp=data['datetime'].min(),
                    contract="all",
                    severity="medium",
                    action_taken="待处理"
                )
                issues.append(issue)
        
        self.issues.extend(issues)
        return issues

    def _get_normal_trading_hours_mask(self, datetimes: pd.Series) -> pd.Series:
        """判断时间点是否在正常交易时间内"""
        # 确保输入是datetime类型
        if datetimes.dtype == 'object':
            try:
                datetimes = pd.to_datetime(datetimes, format='%Y-%m-%d %H:%M:%S')
            except:
                # 如果无法转换，返回全False
                return pd.Series(False, index=datetimes.index)
        
        mask = pd.Series(False, index=datetimes.index)
        for i in range(1, len(datetimes)):
            time_diff = datetimes.iloc[i] - datetimes.iloc[i-1]
            # 假设正常交易时间间隔小于4小时
            if time_diff <= timedelta(hours=4):
                mask.iloc[i] = True
        return mask
    
    def check_price_anomalies(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """检查价格异常 - 排除合约切换点"""
        issues = []
        
        price_columns = ['open', 'high', 'low', 'close']
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            
            for price_col in price_columns:
                if price_col not in symbol_data.columns:
                    continue
                
                prices = symbol_data[price_col]
                datetimes = symbol_data['datetime']
                
                # 检查负价格
                negative_prices = prices < 0
                for idx in symbol_data[negative_prices].index:
                    # 排除合约切换点附近的数据
                    if not self._is_near_rollover(datetimes.loc[idx]):
                        issue = DataQualityIssue(
                            issue_type="negative_price",
                            description=f"{price_col} 为负值: {prices[idx]}",
                            timestamp=datetimes.loc[idx],
                            contract=symbol,
                            severity="high",
                            action_taken="待处理"
                        )
                        issues.append(issue)
                
                # 检查价格跳空（排除合约切换点）
                price_changes = prices.pct_change().abs()
                large_jumps = price_changes > 0.1  # 10%以上的跳空
                
                # 排除第一个点和合约切换点附近的数据
                for idx in symbol_data[large_jumps].index[1:]:
                    if not self._is_near_rollover(datetimes.loc[idx]):
                        issue = DataQualityIssue(
                            issue_type="price_jump",
                            description=f"{price_col} 异常跳空: {price_changes[idx]:.2%}",
                            timestamp=datetimes.loc[idx],
                            contract=symbol,
                            severity="medium",
                            action_taken="待处理"
                        )
                        issues.append(issue)
                
                # 检查OHLC关系合理性
                if all(col in symbol_data.columns for col in ['open', 'high', 'low', 'close']):
                    invalid_ohlc = (
                        (symbol_data['high'] < symbol_data['low']) |
                        (symbol_data['high'] < symbol_data['open']) |
                        (symbol_data['high'] < symbol_data['close']) |
                        (symbol_data['low'] > symbol_data['open']) |
                        (symbol_data['low'] > symbol_data['close'])
                    )
                    
                    for idx in symbol_data[invalid_ohlc].index:
                        if not self._is_near_rollover(datetimes.loc[idx]):
                            issue = DataQualityIssue(
                                issue_type="invalid_ohlc",
                                description="OHLC价格关系不合理",
                                timestamp=datetimes.loc[idx],
                                contract=symbol,
                                severity="high",
                                action_taken="待处理"
                            )
                            issues.append(issue)
        
        self.issues.extend(issues)
        return issues
    
    def _is_near_rollover(self, timestamp: datetime) -> bool:
        """判断时间点是否在合约切换点附近"""
        for rollover_time in self.rollover_times:
            if abs((timestamp - rollover_time).total_seconds()) < 3600:  # 1小时内
                return True
        return False
    
    def check_volume_anomalies(self, data: pd.DataFrame) -> List[DataQualityIssue]:
        """检查成交量异常"""
        issues = []
        
        if 'volume' not in data.columns:
            return issues
        
        for symbol in data['symbol'].unique():
            symbol_data = data[data['symbol'] == symbol]
            volumes = symbol_data['volume']
            datetimes = symbol_data['datetime']
            
            # 检查零成交量（排除合约切换点附近）
            zero_volume = volumes == 0
            for idx in symbol_data[zero_volume].index:
                if not self._is_near_rollover(datetimes.loc[idx]):
                    issue = DataQualityIssue(
                        issue_type="zero_volume",
                        description="成交量为零",
                        timestamp=datetimes.loc[idx],
                        contract=symbol,
                        severity="medium",
                        action_taken="待处理"
                    )
                    issues.append(issue)
            
            # 检查异常大成交量
            if len(volumes) > 10:  # 需要有足够数据计算
                volume_mean = volumes.mean()
                volume_std = volumes.std()
                
                if volume_std > 0:
                    z_scores = (volumes - volume_mean) / volume_std
                    extreme_volumes = z_scores.abs() > 5  # Z-score大于5
                    
                    for idx in symbol_data[extreme_volumes].index:
                        if not self._is_near_rollover(datetimes.loc[idx]):
                            issue = DataQualityIssue(
                                issue_type="extreme_volume",
                                description=f"异常成交量: Z-score = {z_scores[idx]:.2f}",
                                timestamp=datetimes.loc[idx],
                                contract=symbol,
                                severity="low",
                                action_taken="待处理"
                            )
                            issues.append(issue)
        
        self.issues.extend(issues)
        return issues
    
    def generate_quality_report(self) -> Dict[str, Any]:
        """生成数据质量报告"""
        if not self.issues:
            return {
                "status": "excellent", 
                "issues_count": 0,  # 添加issues_count
                "total_issues": 0,  # 添加total_issues
                "issue_types": {},
                "severity_breakdown": {"low": 0, "medium": 0, "high": 0},
                "issues": []
            }
        
        issue_types = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        
        for issue in self.issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
            severity_counts[issue.severity] += 1
        
        total_issues = len(self.issues)
        overall_status = "good" if severity_counts["high"] == 0 else "poor"
        
        return {
            "status": overall_status,
            "issues_count": total_issues,  # 保持向后兼容
            "total_issues": total_issues,  # 添加total_issues
            "issue_types": issue_types,
            "severity_breakdown": severity_counts,
            "issues": self.issues
        }
    
class ExtendedDataCleaner:
    """扩展的数据清洗器 - 专门处理连续全零数据"""
    
    def __init__(self, rollover_points: Optional[List[ContractRollover]] = None):
        self.cleaning_log = []
        self.rollover_points = rollover_points or []
        self.rollover_times = [rp.rollover_datetime for rp in self.rollover_points]## if rp.is_valid]
    
    def _is_near_rollover(self, timestamp: datetime) -> bool:
        """判断时间点是否在合约切换点附近"""
        for rollover_time in self.rollover_times:
            if abs((timestamp - rollover_time).total_seconds()) < 3600:  # 1小时内
                return True
        return False
    
    def handle_missing_values(self, data: pd.DataFrame, method: str = "interpolate") -> pd.DataFrame:
        """处理缺失值 - 修复版本（使用现代pandas语法）"""
        cleaned_data = data.copy()
        
        # 记录原始缺失情况
        original_missing = cleaned_data.isnull().sum().sum()
        
        if method == "interpolate":
            # 对数值列进行线性插值
            numeric_columns = cleaned_data.select_dtypes(include=[np.number]).columns
            
            # 先按合约分组处理，避免跨合约插值
            grouped_data = []
            for symbol in cleaned_data['symbol'].unique():
                symbol_data = cleaned_data[cleaned_data['symbol'] == symbol].copy()
                
                # 确保该合约有数据
                if len(symbol_data) == 0:
                    continue
                    
                # 插值处理 - 使用现代pandas语法
                symbol_data[numeric_columns] = symbol_data[numeric_columns].interpolate(method='linear')
                symbol_data[numeric_columns] = symbol_data[numeric_columns].ffill()  # 替换 fillna(method='ffill')
                symbol_data[numeric_columns] = symbol_data[numeric_columns].bfill()  # 替换 fillna(method='bfill')
                grouped_data.append(symbol_data)
            
            # 检查是否有数据可以拼接
            if grouped_data:
                cleaned_data = pd.concat(grouped_data).sort_values('datetime').reset_index(drop=True)
            else:
                print("警告: 所有合约数据都为空，无法处理缺失值")
                return cleaned_data
            
        elif method == "drop":
            # 删除包含缺失值的行
            cleaned_data = cleaned_data.dropna()
        
        # 记录处理结果
        final_missing = cleaned_data.isnull().sum().sum()
        resolved_count = original_missing - final_missing
        
        self.cleaning_log.append({
            "action": "handle_missing_values",
            "method": method,
            "original_missing": original_missing,
            "resolved_count": resolved_count,
            "remaining_missing": final_missing
        })
        
        return cleaned_data
    
    def handle_price_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理价格异常 - 排除合约切换点"""
        cleaned_data = data.copy()
        corrections_made = 0
        
        price_columns = ['open', 'high', 'low', 'close']
        
        for symbol in cleaned_data['symbol'].unique():
            symbol_mask = cleaned_data['symbol'] == symbol
            symbol_data = cleaned_data[symbol_mask]
            
            for price_col in price_columns:
                if price_col not in symbol_data.columns:
                    continue
                
                # 创建该合约该列数据的掩码
                col_mask = symbol_mask & cleaned_data[price_col].notna()
                
                # 处理负价格（排除切换点附近）
                negative_mask = col_mask & (cleaned_data[price_col] < 0)
                # 排除切换点附近
                for idx in cleaned_data[negative_mask].index:
                    if not self._is_near_rollover(cleaned_data.loc[idx, 'datetime']):
                        # 用前一个有效值替换负价格
                        prev_valid = cleaned_data.loc[:idx-1, price_col][cleaned_data.loc[:idx-1, price_col] > 0]
                        if len(prev_valid) > 0:
                            cleaned_data.loc[idx, price_col] = prev_valid.iloc[-1]
                            corrections_made += 1
                
                # 处理极端价格跳空（排除切换点）
                price_pct_change = cleaned_data[price_col].pct_change().abs()
                extreme_jumps = col_mask & (price_pct_change > 0.2)  # 20%以上的跳空
                
                # 排除第一个点和切换点附近
                extreme_jumps_indices = []
                for idx in cleaned_data[extreme_jumps].index[1:]:
                    if not self._is_near_rollover(cleaned_data.loc[idx, 'datetime']):
                        extreme_jumps_indices.append(idx)
                
                if extreme_jumps_indices:
                    # 标记异常点但不立即修正，需要更复杂的处理
                    cleaned_data.loc[extreme_jumps_indices, f'{price_col}_suspicious'] = True
                    corrections_made += len(extreme_jumps_indices)
        
        self.cleaning_log.append({
            "action": "handle_price_anomalies",
            "corrections_made": corrections_made
        })
        
        return cleaned_data
    
    def handle_volume_anomalies(self, data: pd.DataFrame) -> pd.DataFrame:
        """处理成交量异常 - 排除合约切换点"""
        cleaned_data = data.copy()
        
        if 'volume' not in cleaned_data.columns:
            return cleaned_data
        
        corrections_made = 0
        
        for symbol in cleaned_data['symbol'].unique():
            symbol_mask = cleaned_data['symbol'] == symbol
            symbol_data = cleaned_data[symbol_mask]
            
            # 处理零成交量（排除切换点附近）
            zero_volume_mask = symbol_mask & (cleaned_data['volume'] == 0)
            zero_volume_indices = []
            for idx in cleaned_data[zero_volume_mask].index:
                if not self._is_near_rollover(cleaned_data.loc[idx, 'datetime']):
                    zero_volume_indices.append(idx)
            
            if zero_volume_indices:
                # 用移动平均值替换（排除零值）
                for idx in zero_volume_indices:
                    # 获取附近非零成交量的平均值
                    nearby_data = symbol_data[
                        (symbol_data['datetime'] >= cleaned_data.loc[idx, 'datetime'] - timedelta(hours=2)) &
                        (symbol_data['datetime'] <= cleaned_data.loc[idx, 'datetime'] + timedelta(hours=2)) &
                        (symbol_data['volume'] > 0)
                    ]
                    if len(nearby_data) > 0:
                        cleaned_data.loc[idx, 'volume'] = nearby_data['volume'].mean()
                        corrections_made += 1
        
        self.cleaning_log.append({
            "action": "handle_volume_anomalies",
            "zero_volume_corrected": corrections_made
        })
        
        return cleaned_data
    
    def get_cleaning_summary(self) -> Dict[str, Any]:
        """获取清洗摘要"""
        return {
            "total_actions": len(self.cleaning_log),
            "log": self.cleaning_log
        }
    
    def detect_zero_sequences(self, data: pd.DataFrame, zero_threshold: float = 0.001) -> List[Dict]:
        """
        检测连续的全零数据序列
        """
        zero_sequences = []
        
        # 检查主要数值列是否全为零
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        available_columns = [col for col in numeric_columns if col in data.columns]
        
        if not available_columns:
            return zero_sequences
        
        # 确保数据已按时间排序
        if 'datetime' not in data.columns:
            print("警告: 数据中没有datetime列，无法检测零序列")
            return zero_sequences
        
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # 创建全零掩码
        zero_mask = pd.Series(True, index=data.index)
        for col in available_columns:
            # 处理可能的NaN值
            col_data = data[col].fillna(0)
            zero_mask = zero_mask & (col_data.abs() <= zero_threshold)
        
        # 如果没有全零数据，直接返回
        if not zero_mask.any():
            return zero_sequences
        
        # 找到连续的零序列
        zero_diff = zero_mask.astype(int).diff()
        sequence_starts = zero_diff[zero_diff == 1].index.tolist()
        sequence_ends = zero_diff[zero_diff == -1].index.tolist()
        
        # 处理边界情况
        if zero_mask.iloc[0]:
            sequence_starts = [data.index[0]] + sequence_starts
        if zero_mask.iloc[-1]:
            sequence_ends = sequence_ends + [data.index[-1]]
        
        # 构建序列信息
        for start_idx, end_idx in zip(sequence_starts, sequence_ends):
            # 确保索引有效
            if start_idx > end_idx or start_idx >= len(data) or end_idx >= len(data):
                continue
                
            sequence_data = data.loc[start_idx:end_idx]
            
            # 确保序列数据不为空
            if len(sequence_data) == 0:
                continue
                
            sequence_info = {
                'start_index': start_idx,
                'end_index': end_idx,
                'start_time': sequence_data['datetime'].iloc[0],
                'end_time': sequence_data['datetime'].iloc[-1],
                'duration': len(sequence_data),
                'is_trading_day': self._is_likely_trading_day(sequence_data),
                'surrounding_volume': self._get_surrounding_volume(data, start_idx, end_idx)
            }
            
            zero_sequences.append(sequence_info)
        
        return zero_sequences
    
    def _is_likely_trading_day(self, sequence_data: pd.DataFrame) -> bool:
        """判断零序列是否可能是交易日"""
        # 检查时间特征
        if len(sequence_data) == 0:
            return False
        
        # 如果数据覆盖了典型的交易时间段（如9:00-15:00），则可能是交易日
        times = sequence_data['datetime'].dt.time
        typical_start = pd.Timestamp('09:00:00').time()
        typical_end = pd.Timestamp('15:00:00').time()
        
        has_typical_hours = any(
            typical_start <= t <= typical_end for t in times
        )
        
        return has_typical_hours
    
    def _get_surrounding_volume(self, data: pd.DataFrame, start_idx: int, end_idx: int, window: int = 5) -> Dict:
        """获取零序列前后的成交量信息"""
        before_start = max(0, start_idx - window)
        after_end = min(len(data) - 1, end_idx + window)
        
        before_volume = data.loc[before_start:start_idx-1, 'volume'].mean() if start_idx > 0 else 0
        after_volume = data.loc[end_idx+1:after_end, 'volume'].mean() if end_idx < len(data) - 1 else 0
        
        return {
            'before_avg': before_volume,
            'after_avg': after_volume,
            'is_low_volume_around': before_volume < 10 and after_volume < 10  # 阈值可根据实际情况调整
        }
    
    def handle_zero_sequences(self, data: pd.DataFrame, strategy: str = "interpolate") -> pd.DataFrame:
        """
        处理连续全零数据序列
        
        Parameters:
        -----------
        data : pd.DataFrame
            原始数据
        strategy : str
            处理策略: "interpolate", "forward_fill", "remove", "mark_only"
        
        Returns:
        --------
        pd.DataFrame : 处理后的数据
        """
        cleaned_data = data.copy()
        zero_sequences = self.detect_zero_sequences(cleaned_data)
        
        if not zero_sequences:
            print("未检测到连续全零数据序列")
            return cleaned_data
        
        print(f"检测到 {len(zero_sequences)} 个连续全零数据序列")
        
        total_zero_points = 0
        for i, seq in enumerate(zero_sequences):
            print(f"序列 {i+1}: {seq['start_time']} 到 {seq['end_time']}, "
                  f"时长: {seq['duration']} 个数据点, "
                  f"类型: {'交易日' if seq['is_trading_day'] else '非交易日/间隔期'}")
            
            total_zero_points += seq['duration']
            
            if strategy == "remove":
                # 直接删除这些行
                cleaned_data = cleaned_data.drop(
                    range(seq['start_index'], seq['end_index'] + 1)
                )
                
            elif strategy == "interpolate":
                # 对数值列进行插值
                self._interpolate_zero_sequence(cleaned_data, seq)
                
            elif strategy == "forward_fill":
                # 使用前一个有效值填充
                self._forward_fill_zero_sequence(cleaned_data, seq)
                
            elif strategy == "mark_only":
                # 仅标记，不修改数据
                cleaned_data.loc[seq['start_index']:seq['end_index'], 'is_zero_sequence'] = True
        
        # 重置索引
        cleaned_data = cleaned_data.reset_index(drop=True)
        
        self.cleaning_log.append({
            "action": "handle_zero_sequences",
            "strategy": strategy,
            "sequences_count": len(zero_sequences),
            "total_zero_points": total_zero_points
        })
        
        print(f"处理完成: 共处理 {total_zero_points} 个零数据点")
        return cleaned_data
    
    def _interpolate_zero_sequence(self, data: pd.DataFrame, sequence: Dict):
        """对零序列进行插值处理"""
        start_idx, end_idx = sequence['start_index'], sequence['end_index']
        
        # 获取序列前后的有效数据点
        prev_valid_idx = start_idx - 1
        next_valid_idx = end_idx + 1
        
        # 确保前后都有有效数据
        if prev_valid_idx < 0 or next_valid_idx >= len(data):
            print(f"  警告: 序列边界无法插值，使用前向填充")
            self._forward_fill_zero_sequence(data, sequence)
            return
        
        # 对每个数值列进行线性插值
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'position']
        available_columns = [col for col in numeric_columns if col in data.columns]
        
        for col in available_columns:
            prev_value = data.loc[prev_valid_idx, col]
            next_value = data.loc[next_valid_idx, col]
            
            # 线性插值
            if col in ['volume', 'amount', 'position']:
                # 对于成交量相关列，使用更保守的插值方法
                interpolated_values = np.linspace(prev_value, next_value, sequence['duration'] + 2)[1:-1]
            else:
                # 对于价格列，使用线性插值
                interpolated_values = np.linspace(prev_value, next_value, sequence['duration'] + 2)[1:-1]
            
            data.loc[start_idx:end_idx, col] = interpolated_values
        
        # 标记已处理
        data.loc[start_idx:end_idx, 'was_zero_corrected'] = True
    
    def _forward_fill_zero_sequence(self, data: pd.DataFrame, sequence: Dict):
        """使用前向填充处理零序列"""
        start_idx, end_idx = sequence['start_index'], sequence['end_index']
        
        # 找到前一个有效值
        prev_valid_idx = start_idx - 1
        if prev_valid_idx < 0:
            print(f"  警告: 无法找到前一个有效值，使用后向填充")
            # 使用后一个有效值填充
            next_valid_idx = end_idx + 1
            if next_valid_idx < len(data):
                fill_values = data.loc[next_valid_idx]
                for col in data.columns:
                    if col not in ['datetime', 'symbol'] and col in data.columns:
                        data.loc[start_idx:end_idx, col] = fill_values[col]
            return
        
        # 使用前一个有效值填充
        fill_values = data.loc[prev_valid_idx]
        for col in data.columns:
            if col not in ['datetime', 'symbol'] and col in data.columns:
                data.loc[start_idx:end_idx, col] = fill_values[col]
        
        # 标记已处理
        data.loc[start_idx:end_idx, 'was_zero_corrected'] = True
    
    def analyze_zero_sequences_patterns(self, data: pd.DataFrame) -> Dict:
        """分析零序列的模式"""
        zero_sequences = self.detect_zero_sequences(data)
        
        if not zero_sequences:
            return {
                "status": "no_zero_sequences",
                "total_sequences": 0,
                "total_zero_points": 0,
                "sequence_lengths": [],
                "trading_day_sequences": 0,
                "non_trading_sequences": 0,
                "avg_sequence_length": 0,
                "max_sequence_length": 0,
                "sequences_by_contract": {}
            }
        
        analysis = {
            "status": "has_zero_sequences",  # 添加状态字段
            "total_sequences": len(zero_sequences),
            "total_zero_points": sum(seq['duration'] for seq in zero_sequences),
            "sequence_lengths": [seq['duration'] for seq in zero_sequences],
            "trading_day_sequences": sum(1 for seq in zero_sequences if seq['is_trading_day']),
            "non_trading_sequences": sum(1 for seq in zero_sequences if not seq['is_trading_day']),
            "avg_sequence_length": np.mean([seq['duration'] for seq in zero_sequences]) if zero_sequences else 0,
            "max_sequence_length": max([seq['duration'] for seq in zero_sequences]) if zero_sequences else 0,
            "sequences_by_contract": {}
        }
        
        # 按合约分析
        for seq in zero_sequences:
            seq_data = data.loc[seq['start_index']:seq['end_index']]
            contracts = seq_data['symbol'].unique()
            
            for contract in contracts:
                if contract not in analysis['sequences_by_contract']:
                    analysis['sequences_by_contract'][contract] = 0
                analysis['sequences_by_contract'][contract] += 1
        
        return analysis

class FuturesDataProcessor:
    """增强版期货数据处理类 - 集成零序列处理"""
    
    def __init__(
        self, 
        data_path: Optional[str] = None,
        auto_detect_rollovers: bool = True,
        auto_clean: bool = True, 
        zero_handling_strategy: str = "interpolate",
        adjustment_config: Optional[RolloverAdjustmentConfig] = None,
        rollover_detector_config: Optional[Dict[str, Any]] = None,
        silent_mode: bool = False,
        # 新增配置选项
        use_percentage_adjustment: bool = True,  # 使用百分比复权
        use_window_adjustment: bool = False,     # 是否使用窗口调整（默认不使用）
        default_adjustment_method: str = "forward"  # 默认复权方法: "forward" 或 "backward"
    ):
        """
        初始化期货数据处理器
        
        Args:
            data_path: 数据文件路径，如果提供则在初始化时自动加载
            auto_detect_rollovers: 是否自动检测切换点
            auto_clean: 是否自动清洗数据
            zero_handling_strategy: 零序列处理策略 ("interpolate", "forward_fill", "remove", "mark_only")
            adjustment_config: 复权配置对象
            rollover_detector_config: 切换点检测器配置字典
            silent_mode: 是否启用静默模式
            use_percentage_adjustment: 是否使用百分比复权（默认True）
            use_window_adjustment: 是否使用窗口调整（默认False，即使用单点调整）
            default_adjustment_method: 默认复权方法 "forward"(前复权) 或 "backward"(后复权)
        """
        self.data_path = data_path
        self.auto_detect_rollovers = auto_detect_rollovers
        self.auto_clean = auto_clean
        self.zero_handling_strategy = zero_handling_strategy
        self.silent_mode = silent_mode
        self.use_percentage_adjustment = use_percentage_adjustment
        self.use_window_adjustment = use_window_adjustment
        self.default_adjustment_method = default_adjustment_method
        
        # 数据存储
        self.raw_data = None
        self.cleaned_data = None
        self.rollover_points = []
        self.continuous_data = None
        
        # 配置对象 - 根据参数自动创建或使用传入的配置
        self.adjustment_config = self._create_adjustment_config(adjustment_config)
        
        # 创建切换点检测器
        detector_config = rollover_detector_config or {
            'min_volume': 100,
            'price_change_threshold': 0.5,
            'window_size_hours': 2.0,
            'settlement_period_minutes': 15
        }
        self.rollover_detector = RolloverDetector(
            min_volume=detector_config.get('min_volume', 100),
            price_change_threshold=detector_config.get('price_change_threshold', 0.5),
            adjustment_config=self.adjustment_config,
            window_size_hours=detector_config.get('window_size_hours', 2.0),
            settlement_period_minutes=detector_config.get('settlement_period_minutes', 15)
        )
        
        # 如果提供了数据路径，自动加载数据
        if data_path:
            self.load_data_from_csv(data_path)
            
            # 如果启用了自动清洗，自动清洗数据
            if auto_clean and self.raw_data is not None:
                self.process_data_with_zero_handling()
            
            # 如果启用了自动检测，自动检测切换点
            if auto_detect_rollovers and self.cleaned_data is not None:
                self.detect_rollover_points()
    
    def _create_adjustment_config(self, existing_config: Optional[RolloverAdjustmentConfig]) -> RolloverAdjustmentConfig:
        """根据参数创建或修改调整配置"""
        if existing_config is not None:
            # 如果传入了现有配置，根据参数修改它
            config = existing_config
        else:
            # 创建新的配置
            config = RolloverAdjustmentConfig()
        
        # 设置默认策略
        if self.use_percentage_adjustment:
            if self.use_window_adjustment:
                default_strategy = "percentage_window"
                fallback_strategy = "spread_window"
            else:
                default_strategy = "percentage"
                fallback_strategy = "spread"
        else:
            if self.use_window_adjustment:
                default_strategy = "spread_window"
                fallback_strategy = "percentage_window"
            else:
                default_strategy = "spread"
                fallback_strategy = "percentage"
        
        config.set_global_default(default_strategy)
        config.set_global_fallback(fallback_strategy)
        
        return config
    
    def create_continuous_contract(
        self, 
        method: Optional[str] = None, 
        adjustment_type: str = "percentage"
    ) -> pd.DataFrame:
        """创建连续合约
        
        Args:
            method: 复权方法 - "forward"(前复权) 或 "backward"(后复权)，如果为None则使用默认方法
            adjustment_type: 调整类型 - "percentage"(百分比调整) 或 "spread"(价差调整)
        """
        # 如果没有指定方法，使用默认方法
        method_to_use = method or self.default_adjustment_method
        
        data_to_use = self.cleaned_data if self.cleaned_data is not None else self.raw_data
        
        if data_to_use is None:
            raise ValueError("请先处理数据")
            
        # 只使用有效的切换点
        valid_rollovers = []
        for rp in self.rollover_points:
            is_valid, status = rp.is_strategy_valid()
            if is_valid:
                valid_rollovers.append(rp)
        
        if not valid_rollovers:
            self._print("没有有效的合约切换点，无法创建连续合约")
            # 返回原始数据，但添加必要的调整列
            continuous_data = data_to_use.copy()
            continuous_data['close_adj'] = continuous_data['close']
            continuous_data['cumulative_adjustment'] = 1.0
            continuous_data['cumulative_gap'] = 0.0
            self.continuous_data = continuous_data
            return continuous_data

        continuous_data = data_to_use.copy()
        sorted_rollovers = sorted(valid_rollovers, key=lambda x: x.rollover_datetime)

        self._print(f"使用 {len(valid_rollovers)} 个有效切换点创建连续合约")
        self._print(f"复权方法: {'前复权' if method_to_use == 'forward' else '后复权'}")
        
        # 调试信息：显示每个切换点的调整参数
        for i, rollover in enumerate(sorted_rollovers):
            adjustment = rollover.get_adjustment_factor()
            gap = rollover.get_price_gap()
            strategy = rollover.get_applied_strategy_name()
            self._print(f"切换点 {i+1}: {rollover.old_contract} -> {rollover.new_contract}, "
                      f"策略: {strategy}, 调整因子: {adjustment:.6f}, 价差: {gap:.6f}")

        if method_to_use == "forward":
            # 前复权：当前价格不变，调整历史价格
            if adjustment_type == "percentage":
                continuous_data = self._forward_percentage_adjustment(continuous_data, sorted_rollovers)
            else:  # spread
                continuous_data = self._forward_spread_adjustment(continuous_data, sorted_rollovers)
                
        elif method_to_use == "backward":
            # 后复权：历史价格不变，调整未来价格
            if adjustment_type == "percentage":
                continuous_data = self._backward_percentage_adjustment(continuous_data, sorted_rollovers)
            else:  # spread
                continuous_data = self._backward_spread_adjustment(continuous_data, sorted_rollovers)
        
        else:
            raise ValueError("method 参数必须是 'forward' 或 'backward'")

        # 确保必要的列存在
        required_columns = ['close_adj', 'cumulative_adjustment', 'cumulative_gap']
        for col in required_columns:
            if col not in continuous_data.columns:
                self._print(f"警告: 列 '{col}' 未生成，使用默认值")
                if col == 'close_adj':
                    continuous_data[col] = continuous_data['close']
                elif col == 'cumulative_adjustment':
                    continuous_data[col] = 1.0
                elif col == 'cumulative_gap':
                    continuous_data[col] = 0.0

        self.continuous_data = continuous_data
        self._print(f"{'前复权' if method_to_use == 'forward' else '后复权'}连续合约创建完成")
        self._print(f"调整类型: {'百分比调整' if adjustment_type == 'percentage' else '价差调整'}")
        self._print(f"数据形状: {continuous_data.shape}")
        
        return continuous_data
    
    def get_processor_info(self) -> Dict[str, Any]:
        """获取处理器信息"""
        return {
            "data_path": self.data_path,
            "auto_detect_rollovers": self.auto_detect_rollovers,
            "auto_clean": self.auto_clean,
            "zero_handling_strategy": self.zero_handling_strategy,
            "silent_mode": self.silent_mode,
            "use_percentage_adjustment": self.use_percentage_adjustment,
            "use_window_adjustment": self.use_window_adjustment,
            "default_adjustment_method": self.default_adjustment_method,
            "data_loaded": self.raw_data is not None,
            "data_rows": len(self.raw_data) if self.raw_data is not None else 0,
            "rollover_points_count": len(self.rollover_points),
            "continuous_data_created": self.continuous_data is not None,
            "adjustment_config": {
                "default_strategy": self.adjustment_config.global_default_strategy,
                "fallback_strategy": self.adjustment_config.global_fallback_strategy,
                "available_strategies": self.adjustment_config.get_available_strategies()
            },
            "rollover_detector_config": {
                "min_volume": self.rollover_detector.min_volume,
                "price_change_threshold": self.rollover_detector.price_change_threshold,
                "window_size_hours": self.rollover_detector.window_size_hours,
                "settlement_period_minutes": self.rollover_detector.settlement_period_minutes
            }
        }
    
    def _print(self, *args, **kwargs):
        """自定义打印函数，支持静默模式"""
        if not self.silent_mode:
            print(*args, **kwargs)
    
    def load_data_from_csv(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """加载数据，如果未提供文件路径则使用初始化时的路径"""
        path_to_use = file_path or self.data_path
        if not path_to_use:
            raise ValueError("未提供数据文件路径")
        
        # 调用现有的数据加载逻辑
        # 这里需要将您现有的 load_data_from_csv 方法整合进来
        try:
            self.raw_data = pd.read_csv(path_to_use)
            print(f"数据加载成功: {len(self.raw_data)} 行")
            
            # 数据预处理
            self._ensure_data_types()
            
            return self.raw_data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            return None
    
    def detect_rollover_points(self, data: Optional[pd.DataFrame] = None) -> List[ContractRollover]:
        """检测合约切换点"""
        data_to_use = data or self.raw_data
        if data_to_use is None:
            raise ValueError("请先加载数据")
        
        print("检测合约切换点...")
        self.rollover_points = self.rollover_detector.detect_rollover_points(data_to_use)
        
        valid_count = len([rp for rp in self.rollover_points])
        print(f"检测完成: 找到 {valid_count} 个有效切换点")
        
        return self.rollover_points
    
    def set_adjustment_config(self, config: RolloverAdjustmentConfig):
        """设置复权配置"""
        self.adjustment_config = config
    
    def update_rollover_detector_config(self, **kwargs):
        """更新切换点检测器配置"""
        # 重新创建检测器
        current_config = {
            'min_volume': self.rollover_detector.min_volume,
            'price_change_threshold': self.rollover_detector.price_change_threshold,
            'window_size_hours': self.rollover_detector.window_size_hours,
            'settlement_period_minutes': self.rollover_detector.settlement_period_minutes
        }
        current_config.update(kwargs)
        
        self.rollover_detector = RolloverDetector(
            min_volume=current_config['min_volume'],
            price_change_threshold=current_config['price_change_threshold'],
            adjustment_config=self.adjustment_config,
            window_size_hours=current_config['window_size_hours'],
            settlement_period_minutes=current_config['settlement_period_minutes']
        )
    
    def process_data_with_zero_handling(self) -> pd.DataFrame:
        """完整的数据处理流程 - 包含零序列处理"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        print("\n" + "="*50)
        print("步骤1: 数据类型检查和预处理")
        print("="*50)
        
        # 第一步：确保数据类型正确
        self._ensure_data_types()
        
        print("\n" + "="*50)
        print("步骤2: 检测合约切换点")
        print("="*50)
        
        # 第二步：检测合约切换点
        # self.rollover_points = self.rollover_detector.detect_rollover_points(self.raw_data)
        self.rollover_points = self.rollover_detector.detect_futures_rollovers(self.raw_data)
        valid_rollovers = [rp for rp in self.rollover_points]## if rp.is_valid]
        
        print(f"\n有效合约切换点: {len(valid_rollovers)} 个")
        
        print("\n" + "="*50)
        print("步骤3: 零序列分析")
        print("="*50)
        
        # 第三步：零序列分析
        zero_analyzer = ExtendedDataCleaner(valid_rollovers)
        zero_analysis = zero_analyzer.analyze_zero_sequences_patterns(self.raw_data)
        
        self._print_zero_analysis(zero_analysis)
        
        print("\n" + "="*50)
        print("步骤4: 数据质量检查")
        print("="*50)
        
        # 第四步：数据质量检查
        # quality_checker = DataQualityChecker(valid_rollovers)
        # quality_checker.check_missing_values(self.raw_data)
        # quality_checker.check_price_anomalies(self.raw_data)
        # quality_checker.check_volume_anomalies(self.raw_data)
        
        # quality_report = quality_checker.generate_quality_report()
        # self._print_quality_report(quality_report)
        
        # 第五步：数据清洗（包含零序列处理）
        if self.auto_clean:
            print("\n" + "="*50)
            print("步骤5: 数据清洗（包含零序列处理）")
            print("="*50)
            
            data_cleaner = ExtendedDataCleaner(valid_rollovers)
            
            # 先处理零序列
            self.cleaned_data = data_cleaner.handle_zero_sequences(
                self.raw_data, 
                strategy=self.zero_handling_strategy
            )
            
            # 然后处理其他数据质量问题
            self.cleaned_data = data_cleaner.handle_missing_values(self.cleaned_data)
            # self.cleaned_data = data_cleaner.handle_volume_anomalies(self.cleaned_data)
            # self.cleaned_data = data_cleaner.handle_price_anomalies(self.cleaned_data)
            
            cleaning_summary = data_cleaner.get_cleaning_summary()
            print(f"数据清洗完成，执行了 {cleaning_summary['total_actions']} 个清洗操作")
        else:
            self.cleaned_data = self.raw_data.copy()
            print("跳过数据清洗（auto_clean=False）")
        
        return self.cleaned_data

    def _ensure_data_types(self):
        """确保数据列具有正确的数据类型"""
        if self.raw_data is None:
            return
        
        # 确保datetime列是datetime类型
        if 'datetime' in self.raw_data.columns and self.raw_data['datetime'].dtype == 'object':
            print("转换datetime列为datetime类型...")
            self.raw_data['datetime'] = pd.to_datetime(self.raw_data['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
            # 检查转换结果
            null_count = self.raw_data['datetime'].isnull().sum()
            if null_count > 0:
                print(f"警告: {null_count} 个datetime值无法解析，已设为NaT")
        
        # 确保数值列是数值类型
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'position']
        for col in numeric_columns:
            if col in self.raw_data.columns and self.raw_data[col].dtype == 'object':
                print(f"转换{col}列为数值类型...")
                self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
                # 检查转换结果
                null_count = self.raw_data[col].isnull().sum()
                if null_count > 0:
                    print(f"警告: {null_count} 个{col}值无法转换为数值，已设为NaN")
        
        print("数据类型检查完成")
    
    def _print_zero_analysis(self, analysis: Dict):
        """打印零序列分析结果"""
        # 确保analysis字典包含必要的键
        if "status" not in analysis:
            print("零序列分析数据格式错误")
            return
            
        if analysis["status"] == "no_zero_sequences":
            print("未发现连续全零数据序列")
            return
        
        print(f"零序列分析结果:")
        print(f"- 总零序列数: {analysis.get('total_sequences', 0)}")
        print(f"- 总零数据点: {analysis.get('total_zero_points', 0)}")
        print(f"- 交易日零序列: {analysis.get('trading_day_sequences', 0)}")
        print(f"- 非交易日零序列: {analysis.get('non_trading_sequences', 0)}")
        print(f"- 平均序列长度: {analysis.get('avg_sequence_length', 0):.1f} 个数据点")
        print(f"- 最大序列长度: {analysis.get('max_sequence_length', 0)} 个数据点")
        
        sequences_by_contract = analysis.get('sequences_by_contract', {})
        if sequences_by_contract:
            print("- 各合约零序列分布:")
            for contract, count in sequences_by_contract.items():
                print(f"  - {contract}: {count} 个序列")
    
    def _print_quality_report(self, report: Dict[str, Any]):
        """打印质量报告"""
        # 确保report字典包含必要的键
        if "status" not in report:
            print("数据质量报告格式错误")
            return
            
        print(f"数据质量状态: {report['status']}")
        
        # 使用get方法安全地获取值
        total_issues = report.get('total_issues', report.get('issues_count', 0))
        print(f"总问题数: {total_issues}")
        
        if total_issues > 0:
            issue_types = report.get('issue_types', {})
            if issue_types:
                print("问题类型分布:")
                for issue_type, count in issue_types.items():
                    print(f"  - {issue_type}: {count}")
            
            severity_breakdown = report.get('severity_breakdown', {})
            if severity_breakdown:
                print("严重程度分布:")
                for severity, count in severity_breakdown.items():
                    print(f"  - {severity}: {count}")
    
    def _forward_percentage_adjustment(self, data: pd.DataFrame, rollovers: List[ContractRollover]) -> pd.DataFrame:
        """前复权 - 百分比调整"""
        data = data.copy()
        data['cumulative_adjustment'] = 1.0
        data['cumulative_gap'] = 0.0
        
        print("开始前复权百分比调整...")
        
        # 按时间倒序处理切换点（从最新到最旧）
        for i, rollover in enumerate(reversed(rollovers)):
            adjustment_factor = rollover.get_adjustment_factor()
            price_gap = rollover.get_price_gap()
            
            print(f"处理切换点 {i+1}: {rollover.old_contract} -> {rollover.new_contract}")
            print(f"  调整因子: {adjustment_factor:.6f}, 价差: {price_gap:.6f}")
            print(f"  切换时间: {rollover.rollover_datetime}")
            
            # 调整切换点之前的所有历史数据
            mask = data['datetime'] < rollover.rollover_datetime
            affected_rows = mask.sum()
            
            if affected_rows > 0:
                data.loc[mask, 'cumulative_adjustment'] = data.loc[mask, 'cumulative_adjustment'] * adjustment_factor
                data.loc[mask, 'cumulative_gap'] = data.loc[mask, 'cumulative_gap'] * adjustment_factor + price_gap
                print(f"  影响了 {affected_rows} 行数据")
            else:
                print(f"  没有影响任何数据")
        
        # 应用价格调整
        data['open_adj'] = data['open'] * data['cumulative_adjustment'] + data['cumulative_gap']
        data['high_adj'] = data['high'] * data['cumulative_adjustment'] + data['cumulative_gap']
        data['low_adj'] = data['low'] * data['cumulative_adjustment'] + data['cumulative_gap']
        data['close_adj'] = data['close'] * data['cumulative_adjustment'] + data['cumulative_gap']
        
        # 调整成交量（反向）
        data['volume_adj'] = data['volume'] / data['cumulative_adjustment']
        if 'amount' in data.columns:
            data['amount_adj'] = data['amount'] / data['cumulative_adjustment']
        if 'position' in data.columns:
            data['position_adj'] = data['position'] / data['cumulative_adjustment']
        
        print("前复权百分比调整完成")
        return data

    # 其他调整方法也需要类似的调试信息...
    def _forward_spread_adjustment(self, data: pd.DataFrame, rollovers: List[ContractRollover]) -> pd.DataFrame:
        """前复权 - 价差调整"""
        data = data.copy()
        data['cumulative_gap'] = 0.0
        data['cumulative_adjustment'] = 1.0  # 为了保持列的一致性
        
        print("开始前复权价差调整...")
        
        # 按时间倒序处理切换点（从最新到最旧）
        for i, rollover in enumerate(reversed(rollovers)):
            price_gap = rollover.get_price_gap()
            
            print(f"处理切换点 {i+1}: {rollover.old_contract} -> {rollover.new_contract}")
            print(f"  价差: {price_gap:.6f}")
            
            # 调整切换点之前的所有历史数据
            mask = data['datetime'] < rollover.rollover_datetime
            affected_rows = mask.sum()
            
            if affected_rows > 0:
                data.loc[mask, 'cumulative_gap'] += price_gap
                print(f"  影响了 {affected_rows} 行数据")
            else:
                print(f"  没有影响任何数据")
        
        # 应用价格调整（只加价差）
        data['open_adj'] = data['open'] + data['cumulative_gap']
        data['high_adj'] = data['high'] + data['cumulative_gap']
        data['low_adj'] = data['low'] + data['cumulative_gap']
        data['close_adj'] = data['close'] + data['cumulative_gap']
        
        # 成交量不需要调整（价差调整不影响成交量）
        data['volume_adj'] = data['volume']
        if 'amount' in data.columns:
            data['amount_adj'] = data['amount']
        if 'position' in data.columns:
            data['position_adj'] = data['position']
        
        print("前复权价差调整完成")
        return data

    def _backward_percentage_adjustment(self, data: pd.DataFrame, rollovers: List[ContractRollover]) -> pd.DataFrame:
        """后复权 - 百分比调整"""
        data = data.copy()
        data['cumulative_adjustment'] = 1.0
        data['cumulative_gap'] = 0.0
        
        # 按时间正序处理切换点（从最旧到最新）
        for rollover in rollovers:
            strategy = rollover.get_applied_strategy()
            adjustment_factor, price_gap = strategy.calculate_adjustment(rollover)
            
            # 调整切换点及之后的所有未来数据
            mask = data['datetime'] >= rollover.rollover_datetime
            data.loc[mask, 'cumulative_adjustment'] /= adjustment_factor
            data.loc[mask, 'cumulative_gap'] = data.loc[mask, 'cumulative_gap'] / adjustment_factor - price_gap
        
        # 应用价格调整
        data['open_adj'] = data['open'] * data['cumulative_adjustment'] + data['cumulative_gap']
        data['high_adj'] = data['high'] * data['cumulative_adjustment'] + data['cumulative_gap']
        data['low_adj'] = data['low'] * data['cumulative_adjustment'] + data['cumulative_gap']
        data['close_adj'] = data['close'] * data['cumulative_adjustment'] + data['cumulative_gap']
        
        # 调整成交量（反向）
        data['volume_adj'] = data['volume'] / data['cumulative_adjustment']
        data['amount_adj'] = data['amount'] / data['cumulative_adjustment']
        if 'position' in data.columns:
            data['position_adj'] = data['position'] / data['cumulative_adjustment']
        
        return data

    def _backward_spread_adjustment(self, data: pd.DataFrame, rollovers: List[ContractRollover]) -> pd.DataFrame:
        """后复权 - 价差调整"""
        data = data.copy()
        data['cumulative_gap'] = 0.0
        
        # 按时间正序处理切换点（从最旧到最新）
        for rollover in rollovers:
            price_gap = rollover.get_price_gap()
            
            # 调整切换点及之后的所有未来数据
            mask = data['datetime'] >= rollover.rollover_datetime
            data.loc[mask, 'cumulative_gap'] += price_gap
        
        # 应用价格调整（只加价差）
        data['open_adj'] = data['open'] + data['cumulative_gap']
        data['high_adj'] = data['high'] + data['cumulative_gap']
        data['low_adj'] = data['low'] + data['cumulative_gap']
        data['close_adj'] = data['close'] + data['cumulative_gap']
        
        # 成交量不需要调整（价差调整不影响成交量）
        data['volume_adj'] = data['volume']
        data['amount_adj'] = data['amount']
        if 'position' in data.columns:
            data['position_adj'] = data['position']
        
        return data

    def get_adjustment_history(self) -> pd.DataFrame:
        """获取调整历史记录"""
        if not hasattr(self, 'continuous_data') or self.continuous_data is None:
            raise ValueError("请先创建连续合约")
        
        # 提取调整点信息
        adjustment_points = []
        for rollover in self.rollover_points:
            is_valid, status = rollover.is_strategy_valid()
            if is_valid:
                adjustment_points.append({
                    'datetime': rollover.rollover_datetime,
                    'old_contract': rollover.old_contract,
                    'new_contract': rollover.new_contract,
                    'strategy': rollover.get_applied_strategy_name(),
                    'adjustment_factor': rollover.get_adjustment_factor(),
                    'price_gap': rollover.get_price_gap(),
                    'validity_status': status.value
                })
        
        return pd.DataFrame(adjustment_points)

    def analyze_continuous_contract(self) -> dict:
        """分析连续合约质量"""
        if not hasattr(self, 'continuous_data') or self.continuous_data is None:
            raise ValueError("请先创建连续合约")
        
        analysis = {}
        
        # 基本统计
        analysis['total_data_points'] = len(self.continuous_data)
        analysis['total_rollovers'] = len(self.rollover_points)
        
        # 有效切换点统计
        valid_rollovers = [rp for rp in self.rollover_points if rp.is_strategy_valid()[0]]
        analysis['valid_rollovers'] = len(valid_rollovers)
        analysis['valid_ratio'] = len(valid_rollovers) / len(self.rollover_points) if self.rollover_points else 0
        
        # 价格连续性分析
        if 'close_adj' in self.continuous_data.columns:
            returns = self.continuous_data['close_adj'].pct_change().dropna()
            analysis['return_stats'] = {
                'mean': returns.mean(),
                'std': returns.std(),
                'min': returns.min(),
                'max': returns.max(),
                'sharpe': returns.mean() / returns.std() if returns.std() > 0 else 0
            }
            
            # 检查价格跳跃
            large_jumps = returns.abs() > 0.1  # 10%以上的跳跃
            analysis['large_jump_count'] = large_jumps.sum()
            analysis['large_jump_ratio'] = large_jumps.mean()
        
        # 策略使用统计
        strategy_counts = {}
        for rollover in valid_rollovers:
            strategy = rollover.get_applied_strategy_name()
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        analysis['strategy_distribution'] = strategy_counts
        
        return analysis
    
    def plot_price_comparison(self, start_date=None, end_date=None, max_points=5000, sample_step=None):
        """绘制原始价格和复权后价格的对比图 - 修复版本"""
        if self.continuous_data is None:
            self.create_continuous_contract("forward")
        
        if self.continuous_data is None:
            raise ValueError("连续合约数据未创建，无法绘图")
        
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            
            # 设置中文字体
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'Heiti TC', 'STHeiti', 'PingFang SC']
            plt.rcParams['axes.unicode_minus'] = False

            # 筛选时间范围
            plot_data = self.continuous_data.copy()
            if start_date:
                plot_data = plot_data[plot_data['datetime'] >= start_date]
            if end_date:
                plot_data = plot_data[plot_data['datetime'] <= end_date]
            
            if len(plot_data) == 0:
                print("指定时间范围内无数据")
                return
            
            # 抽样以减少数据点
            step = max(len(plot_data) // max_points, 1)
            plot_sampled = plot_data.iloc[::(sample_step if sample_step else step)]

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))

            # 两条价格线 - 直接使用datetime对象，matplotlib会自动处理
            ax.plot(plot_sampled['datetime'], plot_sampled['close'], label='原始收盘价', color='blue', alpha=0.7, linewidth=1)
            ax.plot(plot_sampled['datetime'], plot_sampled['close_adj'], label='前复权收盘价', color='green', alpha=0.7, linewidth=1)

            # 标记切换点
            for rollover in self.rollover_points:
                rv = rollover.rollover_datetime
                
                # 检查切换点是否在绘图范围内
                if start_date and rv < start_date:
                    continue
                if end_date and rv > end_date:
                    continue
                    
                # 修复：将datetime转换为matplotlib数值格式
                rv_num = float(mdates.date2num(rv))
                ax.axvline(rv_num, color='red', linestyle='--', alpha=0.5)
                
                # 获取当前y轴范围来放置文本
                ylim = ax.get_ylim()
                ax.text(
                    rv_num, ylim[1],
                    f'{rollover.old_contract}→{rollover.new_contract}',
                    rotation=90, va='top', ha='right', fontsize=8, color='red'
                )

            # 智能设置x轴刻度
            self._set_smart_date_ticks(ax, plot_sampled['datetime'])

            ax.set_title('原始与前复权连续合约价格')
            ax.set_xlabel('时间')
            ax.set_ylabel('价格')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("警告: 未安装matplotlib，无法显示图表")
            print("请安装: pip install matplotlib")
        except Exception as e:
            print(f"绘图失败: {e}")

    def _set_smart_date_ticks(self, ax, datetimes):
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
            formatter = mdates.DateFormatter('%H:%M\n%m-%d')
        elif hours > 6:  # 超过6小时
            locator = mdates.HourLocator(interval=2)
            formatter = mdates.DateFormatter('%H:%M\n%m-%d')
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

def debug_data_loading(file_path: str):
    """调试数据加载过程"""
    print("=== 数据加载调试 ===")
    
    # 1. 直接读取CSV，不进行日期解析
    data_raw = pd.read_csv(file_path)
    print(f"1. 原始数据信息:")
    print(f"   - 列名: {list(data_raw.columns)}")
    print(f"   - 形状: {data_raw.shape}")
    print(f"   - datetime列类型: {data_raw['datetime'].dtype}")
    print(f"   - datetime列前5个值: {list(data_raw['datetime'].head())}")
    
    # 2. 尝试解析日期
    try:
        data_parsed = pd.read_csv(file_path, parse_dates=['datetime'], dayfirst=True)
        print(f"\n2. 解析日期后:")
        print(f"   - datetime列类型: {data_parsed['datetime'].dtype}")
        print(f"   - datetime列前5个值: {list(data_parsed['datetime'].head())}")
    except Exception as e:
        print(f"\n2. 日期解析失败: {e}")
        # 尝试手动解析
        data_parsed = data_raw.copy()
        data_parsed['datetime'] = pd.to_datetime(data_parsed['datetime'], format='%Y-%m-%d %H:%M:%S', errors='coerce')
        print(f"   - 手动解析后datetime列类型: {data_parsed['datetime'].dtype}")
        print(f"   - 手动解析后datetime列前5个值: {list(data_parsed['datetime'].head())}")
    
    # 3. 检查数值列
    numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'position']
    print(f"\n3. 数值列类型:")
    for col in numeric_columns:
        if col in data_parsed.columns:
            print(f"   - {col}: {data_parsed[col].dtype}")
            print(f"     前5个值: {list(data_parsed[col].head())}")

def analyze_data_coverage(self, data: pd.DataFrame):
    """分析数据覆盖情况"""
    print("数据覆盖分析:")
    print(f"总数据点数: {len(data)}")
    print(f"时间范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
    print(f"合约数量: {data['symbol'].nunique()}")
    
    # 按合约统计
    contract_stats = data.groupby('symbol').agg({
        'datetime': ['min', 'max', 'count'],
        'close': 'mean'
    }).round(2)
    
    contract_stats.columns = ['开始时间', '结束时间', '数据点数', '平均价格']
    print("\n各合约统计:")
    print(contract_stats)
    
    # 检查时间连续性
    data_sorted = data.sort_values('datetime')
    time_gaps = data_sorted['datetime'].diff().dt.total_seconds().dropna()
    avg_gap = time_gaps.mean()
    max_gap = time_gaps.max()
    
    print(f"\n时间间隔分析:")
    print(f"平均时间间隔: {avg_gap/60:.1f} 分钟")
    print(f"最大时间间隔: {max_gap/3600:.1f} 小时")
    
    return contract_stats

def main():
    """主函数示例"""
    # print("=== 数据加载调试 ===")
    # debug_data_loading("data/FG.csv")
    
    # 创建处理器
    processor = FuturesDataProcessor(
        data_path="data/FG.csv",
        auto_detect_rollovers=True,
        auto_clean=True, 
        zero_handling_strategy="interpolate",
        default_adjustment_method="backward",
        use_window_adjustment=True
    )
    
    # 显示数据清洗结果
    print("\n处理完成!")
    if processor.cleaned_data is not None:
        print(f"处理后数据行数: {len(processor.cleaned_data)}")
    
    if processor.cleaned_data is not None and 'was_zero_corrected' in processor.cleaned_data.columns:
        corrected_count = processor.cleaned_data['was_zero_corrected'].sum()
        print(f"修正的零数据点: {corrected_count}")
    
    # 创建连续合约
    continuous_data = processor.create_continuous_contract(adjustment_type="percentage")

    # 显示前几行结果
    if continuous_data is not None:
        print("\n前复权结果预览:")
        cols_to_show = ['datetime', 'symbol', 'close', 'close_adj', 'cumulative_adjustment']
        print(continuous_data[cols_to_show].head(10))
    else:
        print("\n错误: 无法创建连续合约")
    
    print("\n=== 调整后价格范围统计 ===")
    for col in ['open_adj', 'high_adj', 'low_adj', 'close_adj']:
        max_val = continuous_data[col].max()
        min_val = continuous_data[col].min()
        print(f"{col}:")
        print(f"  最小值 = {min_val:.4f}")
        print(f"  最大值 = {max_val:.4f}")
    print(f"累计调整因子范围: {continuous_data['cumulative_adjustment'].min():.6f} ~ {continuous_data['cumulative_adjustment'].max():.6f}")

    # 绘制价格对比图
    processor.plot_price_comparison(max_points=5000)

if __name__ == "__main__":
    main()
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any, Callable
import warnings
from enum import Enum

from ContractRollover import ContractRollover
from RolloverDetector import FuturesRolloverDetector_MainTick_MainCloseLast


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
        if len(rollover.old_data) == 0 or len(rollover.new_data) == 0:
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
        old_avg = self._calculate_weighted_average(rollover.old_data)
        new_avg = self._calculate_weighted_average(rollover.new_data)
        
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
        if len(rollover.old_data) == 0 or len(rollover.new_data) == 0:
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
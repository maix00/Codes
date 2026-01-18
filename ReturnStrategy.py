'''
本模块实现了多种收益率计算策略及其在合约切换（复权）场景下的应用。
主要功能包括：
1. 定义了收益率计算策略的抽象基类（ReturnCalculationStrategy），并实现了简单收益率（SimpleReturnStrategy）和对数收益率（LogReturnStrategy）两种常用策略。
2. 提供了直接复权收益率计算器（DirectRolloverReturnCalculator），用于处理多个合约切换事件，自动调整价格序列，实现连续收益率的计算。
3. 支持自定义收益率计算策略的切换，能够灵活适应不同的收益率计算需求。
4. 提供了收益率序列的统计分析功能，包括均值、标准差、夏普比率、总收益等指标。
输入定义：
- 价格数据（price_data）：由时间戳、价格、合约代码组成的列表。
- 合约切换事件（rollover_events）：包含切换时间、旧合约、新合约等信息的事件列表。
输出定义：
- 收益率序列：由时间戳和收益率组成的列表。
- 调整历史：每次合约切换后的累计调整因子和价格差。
- 统计信息：收益率序列的统计指标字典。
适用场景：
本模块适用于期货、可转债等存在合约切换的金融品种，能够自动处理合约切换带来的价格不连续问题，输出连续的收益率序列，便于后续量化分析和策略开发。
'''

from abc import ABC, abstractmethod
import numpy as np
from datetime import datetime
from typing import List, Optional, Tuple, Dict, Any


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

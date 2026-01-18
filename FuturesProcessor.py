"""
本模块提供期货合约切换与复权处理的基础框架，包含合约切换事件的数据结构、数据表管理器、以及多种复权策略的实现。
主要内容包括：
- ContractRollover：用于描述期货合约切换事件及其相关数据。
- DataFrameManager：数据表管理与标准化的基类，支持列名映射、数据表校验等功能。
- FuturesProcessorBase：期货主力合约处理的基类，定义切换点检测、复权因子计算等接口。
- 多种复权策略（AdjustmentStrategy 及其子类）：包括百分比复权、价差复权、加权平均复权和手动覆盖策略，支持窗口均值、加权等多种方式。
- 相关辅助枚举类型和静态方法。
本模块适用于期货主力合约序列的生成、合约切换点检测、以及历史行情的复权处理等场景，便于扩展和自定义不同的复权策略。
"""

from abc import ABC, abstractmethod
import pandas as pd
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict
from StrategySelector import ProductPeriodStrategySelector
from enum import Enum
from dataclasses import dataclass, field

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
    new_contract_end_datetime: Optional[datetime] = None    # 新合约结束时间点
    new_contract_end_date: Optional[date] = None            # 新合约结束日期
    old_contract_start_datetime: Optional[datetime] = None  # 旧合约开始时间点
    old_contract_start_date: Optional[date] = None          # 旧合约开始日期
    old_contract_end_datetime: Optional[datetime] = None    # 旧合约结束时间点
    old_contract_end_date: Optional[date] = None            # 旧合约结束日期

class DataFrameManager:
    """合约切换点检测器基类"""
    
    # 期望的数据表变量名（基类不定义，要求子类实现）
    @property
    def EXPECTED_TABLE_NAMES(self) -> List[str]:
        raise NotImplementedError("请在子类中定义 EXPECTED_TABLE_NAMES 属性")
    
    # 期望的列结构（基类不定义，要求子类实现）
    @property
    def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
        raise NotImplementedError("请在子类中定义 EXPECTED_COLUMNS 属性")
    
    # 期望的必需列结构（基类不定义，要求子类实现）
    @property
    def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
        raise NotImplementedError("请在子类中定义 REQUIRED_COLUMNS 属性")
    
    def __init__(self, 
                 column_mapping: Optional[Dict[str, Dict[str, str]]] = None,
                 data_tables: Optional[Dict[str, pd.DataFrame]] = None):
        """
        初始化DataFrameManager

        Args:
            column_mapping: 每个数据表的列名映射字典
            data_tables: 直接提供的数据表字典
        """
        # 设置每个表的列名映射，默认为空字典
        self.column_mapping = column_mapping or {}
        # 存储数据表
        self.data_tables: Dict[str, pd.DataFrame] = {}
        
        # 根据提供的data_tables设置数据表
        if data_tables:
            self._load_data_tables(data_tables)
    
    def _load_data_tables(self, data_tables: Dict[str, pd.DataFrame]):
        """
        加载数据表
        
        Args:
            data_tables: 数据表字典，键为标准表名，值为DataFrame
        """
        for standard_table_name in self.EXPECTED_TABLE_NAMES:
            # 尝试从提供的数据表中获取数据
            if standard_table_name in data_tables:
                raw_data = data_tables[standard_table_name]
                if isinstance(raw_data, pd.DataFrame):
                    # 映射列名（如果已有映射）
                    mapped_data = self._map_input_columns(raw_data, standard_table_name)
                    self.data_tables[standard_table_name] = mapped_data
                    print(f"加载数据表 '{standard_table_name}'")
                    # 自动检查列要求
                    self._check_table_column_requirements(standard_table_name)
                else:
                    print(f"警告: 数据表 '{standard_table_name}' 不是DataFrame类型")
            else:
                print(f"警告: 未提供数据表 '{standard_table_name}'")
    
    def _map_input_columns(self, data: pd.DataFrame, table_type: str) -> pd.DataFrame:
        """
        将输入数据的列名映射为标准列名
        
        Args:
            data: 输入数据
            table_type: 数据表类型（'main_tick', 'date_main_sub', 'date_main_close_last'）
            
        Returns:
            列名已标准化的数据
            
        Raises:
            ValueError: 当必需的列不存在时抛出异常
        """
        # 获取特定表的列名映射
        table_column_mapping = self.column_mapping.get(table_type, {})
        
        # 只有在需要映射时才创建副本，否则直接返回原数据
        if not table_column_mapping:
            mapped_data = data
        else:
            mapped_data = data.copy()
            
            # 应用列名映射
            for original_col, expected_col in table_column_mapping.items():
                if original_col in mapped_data.columns:
                    mapped_data.rename(columns={original_col: expected_col}, inplace=True)
        
        # 检查必需的列是否存在
        required_columns = self.REQUIRED_COLUMNS.get(table_type, [])
        missing_columns = [col for col in required_columns if col not in mapped_data.columns]
        if missing_columns:
            raise ValueError(f"数据表 {table_type} 缺少必需的列: {missing_columns}")
            
        return mapped_data
    
    def set_column_mapping(self, table_name: str, mapping: Dict[str, str]):
        """
        为特定数据表设置列名映射，并立即应用到已存在的数据表上
        
        Args:
            table_name: 数据表名
            mapping: 列名映射字典
                    键为用户提供的列名，值为期望的列名
                    例如: {'时间': 'datetime', '合约': 'symbol', '开盘价': 'open'}
        """
        self.column_mapping[table_name] = mapping
        
        # 如果该表已存在，立即应用列名映射
        if table_name in self.data_tables:
            print(f"为数据表 '{table_name}' 应用新的列名映射")
            try:
                mapped_data = self._map_input_columns(self.data_tables[table_name], table_name)
                self.data_tables[table_name] = mapped_data
                # 自动检查列要求
                self._check_table_column_requirements(table_name)
            except ValueError as e:
                print(f"列映射应用失败: {e}")
    
    def add_data_table(self, table_name: str, data: pd.DataFrame):
        """
        添加数据表，并应用已有的列名映射
        
        Args:
            table_name: 标准数据表名
            data: 数据表内容
        """
        # 检查表名是否是我们期望的
        if table_name in self.EXPECTED_TABLE_NAMES:
            # 映射列名（如果已有映射）
            try:
                mapped_data = self._map_input_columns(data, table_name)
                self.data_tables[table_name] = mapped_data
                # print(f"数据表 '{table_name}' 已添加并应用列名映射")
                # 自动检查列要求
                self._check_table_column_requirements(table_name)
            except ValueError as e:
                print(f"数据表添加失败: {e}")
        else:
            # 如果不是期望的表名，仍然可以添加，但给出警告
            self.data_tables[table_name] = data.copy()
            print(f"警告: 数据表 '{table_name}' 不在期望的表名列表中")
            print(f"期望的表名: {self.EXPECTED_TABLE_NAMES}")
    
    def get_mapped_table(self, expected_table_name: str) -> pd.DataFrame:
        """
        获取映射后的数据表
        
        Args:
            expected_table_name: 期望的数据表名
            
        Returns:
            对应的数据表，如果不存在则返回空的DataFrame
        """
        # 验证表是否可用
        is_valid, missing_tables = self.validate_required_tables([expected_table_name])
        if not is_valid:
            # 如果表不可用，返回空的DataFrame，但保持预期的列结构
            expected_columns = self.EXPECTED_COLUMNS.get(expected_table_name, [])
            print(f"警告: 数据表 '{expected_table_name}' 不可用: {missing_tables}")
            return pd.DataFrame(columns=expected_columns)
            
        # 获取表数据
        table = self.data_tables.get(expected_table_name)
        if table is None:
            # 如果表不存在，返回空的DataFrame，但保持预期的列结构
            expected_columns = self.EXPECTED_COLUMNS.get(expected_table_name, [])
            print(f"警告: 数据表 '{expected_table_name}' 不存在")
            return pd.DataFrame(columns=expected_columns)
        elif len(table) == 0:
            print(f"警告: 数据表 '{expected_table_name}' 为空")
            
        return table
    
    def list_available_tables(self) -> Dict[str, str]:
        """
        列出所有可用的数据表
        
        Returns:
            字典，键为标准表名，值为状态信息
        """
        available_tables = {}
        
        for standard_name in self.EXPECTED_TABLE_NAMES:
            if standard_name in self.data_tables and len(self.data_tables[standard_name]) > 0:
                available_tables[standard_name] = "[已加载]"
            else:
                available_tables[standard_name] = "[缺失]"
                
        return available_tables
    
    def check_table_availability(self) -> List[str]:
        """
        检查哪些期望的数据表是可用的（非空的）
        
        Returns:
            可用的期望数据表列表
        """
        available_tables = []
        for table_name in self.EXPECTED_TABLE_NAMES:
            if table_name in self.data_tables and len(self.data_tables[table_name]) > 0:
                available_tables.append(table_name)
        return available_tables
    
    def validate_required_tables(self, required_tables: List[str]) -> Tuple[bool, List[str]]:
        """
        验证所需的表是否都已提供且非空
        
        Args:
            required_tables: 必需的表名列表
            
        Returns:
            (是否全部满足, 缺失的表名列表)
        """
        missing_tables = []
        for table_name in required_tables:
            if table_name not in self.data_tables or len(self.data_tables[table_name]) == 0:
                missing_tables.append(table_name)
        
        is_available = len(missing_tables) == 0
        if not is_available:
            raise ValueError(f"缺少必需的数据表: {missing_tables}")
                
        return is_available, missing_tables
    
    def _check_table_column_requirements(self, table_name: str) -> bool:
        """
        检查特定数据表的列要求是否满足
        
        Args:
            table_name: 数据表名
            
        Returns:
            是否满足列要求
        """
        if table_name not in self.data_tables:
            print(f"  数据表 {table_name} 不存在")
            return False
            
        data = self.data_tables[table_name]
        expected_columns = self.EXPECTED_COLUMNS.get(table_name, [])
        required_columns = self.REQUIRED_COLUMNS.get(table_name, [])
        
        # 检查必需列
        missing_required = [col for col in required_columns if col not in data.columns]
        if missing_required:
            print(f"  数据表 {table_name} 缺少必需列: {missing_required}")
            return False
            
        # 检查期望列（警告级别）
        missing_expected = [col for col in expected_columns if col not in data.columns]
        if missing_expected:
            print(f"  数据表 {table_name} 缺少期望列: {missing_expected}")
            
        # print(f"  数据表 {table_name} 列要求检查通过")
        return True
    
class FuturesProcessorBase(DataFrameManager):

    def __init__(self, 
                 column_mapping: Dict[str, Dict[str, str]] | None = None, 
                 data_tables: Dict[str, pd.DataFrame] | None = None):
        
        super().__init__(column_mapping, data_tables)
        self.rollover_points: Dict[str, Dict[str, ContractRollover]] = {}
        self.rollover_points_cache_path: Optional[str] = None
        self.product_id_list: List[str] = []
        self.strategy_selector: Optional[ProductPeriodStrategySelector] = None
    
    def detect_rollover_points(self, *args, **kwargs) -> Dict[str, Dict[str, ContractRollover]]:
        """
        检测合约切换点 - 基类方法，需要在子类中实现

        Args:
            **kwargs: 可选参数，供子类实现时使用

        Returns:
            检测到的切换点列表

        Raises:
            NotImplementedError: 基类方法需要在子类中实现
        """
        raise NotImplementedError("detect_rollover_points方法需要在子类中实现")

    def calculate_adjustment(self, strategy_selector: ProductPeriodStrategySelector, *args, **kwargs) -> pd.DataFrame:
        """
        根据ProductPeriodStrategySelector对象获取对应的adjustment_factor，并存储adjustmentstrategy信息

        Args:
            strategy_selector: ProductPeriodStrategySelector对象

        Returns:
            adjustment_factor: float
        """
        raise NotImplementedError("get_adjustment_factor方法需要在子类中实现")

    def generate_main_contract_series(self, *args, **kwargs) -> pd.DataFrame:
        """
        生成主力合约时间序列，返回DataFrame

        Returns:
            pd.DataFrame: 包含主力合约随时间变化的序列
        """
        raise NotImplementedError("generate_main_contract_series方法需要在子类中实现")
    
class ValidityStatus(Enum):
    VALID = "valid"
    INVALID_ZERO_DIVISION = "invalid_zero_division"
    INSUFFICIENT_DATA = "insufficient_data"
    LATER_INSUFFICIENT_DATA = "later_insufficient_data"
    NEGATIVE_PRICES = "negative_prices"
    MANUALLY_OVERRIDDEN_VALID = "manually_overridden_valid"
    MANUALLY_OVERRIDDEN_INVALID = "manually_overridden_invalid"

class AdjustmentDirection(Enum):
    ADJUST_OLD = 0
    ADJUST_NEW = 1

class AdjustmentOperation(Enum):
    MULTIPLICATIVE = 0
    ADDITIVE = 1

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

class AdjustmentStrategy(ABC):
    """复权策略基类"""

    def __init__(self, adjustment_direction: AdjustmentDirection = AdjustmentDirection.ADJUST_OLD):
        self.adjustment_direction = adjustment_direction
        self.adjustment_operation: AdjustmentOperation = AdjustmentOperation.MULTIPLICATIVE
        self.description: Optional[str] = None
    
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
    
    @staticmethod
    def apply_multiplicative_adjustment(adjustment: float, results: list,
                                        adjustment_backward_bool: bool = False) -> Optional[float]:
        adjustment_new = adjustment
        if adjustment is not None and len(results) > 0:
            for r in results:
                if r.get('is_valid') is not None and r['is_valid'] and r.get('val_adjust_old') is not None:
                    r['val_adjust_old'] *= adjustment
            if not adjustment_backward_bool:
                return
            max_r = max(
                results,
                key=lambda r: pd.to_datetime(
                    r.get('old_contract_start_date') 
                    if r.get('is_valid') is not None and r['is_valid'] and r.get('reference_time') is not None 
                    else pd.Timestamp.min
                )
            )
            if max_r.get('val_adjust_new') is not None:
                adjustment_new = max_r['val_adjust_new'] * adjustment
            return adjustment_new
        return adjustment_new

    @staticmethod
    def apply_additive_adjustment(adjustment: float, results: list,
                                  adjustment_backward_bool: bool = False) -> Optional[float]:
        adjustment_new = adjustment
        if adjustment is not None and len(results) > 0:
            for r in results:
                if r.get('is_valid') is not None and r['is_valid'] and r.get('val_adjust_old') is not None:
                    r['val_adjust_old'] += adjustment
            if not adjustment_backward_bool:
                return
            max_r = max(
                results,
                key=lambda r: pd.to_datetime(
                    r.get('old_contract_start_date') 
                    if r.get('is_valid') is not None and r['is_valid'] and r.get('reference_time') is not None 
                    else pd.Timestamp.min
                )
            )
            if max_r.get('val_adjust_new') is not None:
                adjustment_new = max_r['val_adjust_new'] + adjustment
            return adjustment_new
        return adjustment_new

    def apply_adjustment_to_results(self, adjustment: float, results: list, 
                                    adjustment_backward_bool: bool = False) -> Optional[float]:
        """
        基类方法：对子类开放，允许定向至不同的静态方法
        """
        if self.adjustment_operation == AdjustmentOperation.ADDITIVE:
            return self.apply_additive_adjustment(adjustment, results, adjustment_backward_bool)
        elif self.adjustment_operation == AdjustmentOperation.MULTIPLICATIVE:
            return self.apply_multiplicative_adjustment(adjustment, results, adjustment_backward_bool)
        else:
            raise ValueError(f"未知的调整操作类型: {self.adjustment_operation}")

class PercentageAdjustmentStrategy(AdjustmentStrategy):
    """百分比复权策略"""
    
    def __init__(self, use_window: bool = False, window_size: int = 120,
                 old_price_field: str = 'close', new_price_field: str = 'close',
                 new_price_old_data_bool: bool = True, description: Optional[str] = None,
                 adjustment_direction: AdjustmentDirection = AdjustmentDirection.ADJUST_OLD,
                 adjustment_backward_datetime: Optional[datetime] = None):
        super().__init__(adjustment_direction=adjustment_direction)
        self.use_window = use_window
        self.window_size = window_size
        self.old_price_field = old_price_field
        self.new_price_field = new_price_field
        self.new_price_old_data_bool = new_price_old_data_bool
        self.description = description
        self.adjustment_direction = adjustment_direction
        self.adjustment_backward_datetime = adjustment_backward_datetime
        self.name = "percentage_window" if use_window else "percentage"

    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        if self.use_window:
            old_price, new_price = calculate_settlement(
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
            old_price, new_price = calculate_settlement(
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

class SpreadAdjustmentStrategy(AdjustmentStrategy):
    """价差复权策略"""
    
    def __init__(self, use_window: bool = True, window_size: int = 120,
                 old_price_field: str = 'close', new_price_field: str = 'close',
                 new_price_old_data_bool: bool = True, description: Optional[str] = None,
                 adjustment_direction: AdjustmentDirection = AdjustmentDirection.ADJUST_OLD):
        super().__init__(adjustment_direction=adjustment_direction)
        self.adjustment_operation = AdjustmentOperation.ADDITIVE
        self.use_window = use_window
        self.window_size = window_size
        self.old_price_field = old_price_field
        self.new_price_field = new_price_field
        self.new_price_old_data_bool = new_price_old_data_bool
        self.description = description
        self.name = "spread_window" if self.use_window else "spread"
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        if self.use_window:
            old_price, new_price = calculate_settlement(
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

        gap = new_price - old_price
        return 1.0, gap
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        # 价差复权几乎总是有效的，除非数据不足
        if self.new_price_old_data_bool:
            if len(rollover.old_contract_old_data) == 0 or len(rollover.new_contract_old_data) == 0:
                return False, ValidityStatus.INSUFFICIENT_DATA
        else:
            if len(rollover.old_contract_old_data) == 0 or len(rollover.new_contract_new_data) == 0:
                return False, ValidityStatus.INSUFFICIENT_DATA
        return True, ValidityStatus.VALID
    
    def get_name(self) -> str:
        return self.name

class WeightedAverageStrategy(AdjustmentStrategy):
    """加权平均复权策略 - 用于处理异常情况"""
    
    def __init__(self, weights_by_time: Optional[Dict[str, float]] = None,
                 old_price_field: str = 'close', new_price_field: str = 'close',
                 time_field: str = 'datetime', new_price_old_data_bool: bool = True,
                 description: Optional[str] = None,
                 adjustment_direction: AdjustmentDirection = AdjustmentDirection.ADJUST_OLD):
        super().__init__(adjustment_direction=adjustment_direction)
        self.weights_by_time = weights_by_time or {}
        self.old_price_field = old_price_field
        self.new_price_field = new_price_field
        self.time_field = time_field
        self.new_price_old_data_bool = new_price_old_data_bool
        self.description = description
        self.name = "weighted_average"
    
    def calculate_adjustment(self, rollover: 'ContractRollover') -> Tuple[float, float]:
        # 基于成交量加权的平均价格计算
        old_avg = self._calculate_weighted_average(rollover.old_contract_old_data, price_field=self.old_price_field)
        if self.new_price_old_data_bool:
            new_avg = self._calculate_weighted_average(rollover.new_contract_old_data, price_field=self.new_price_field)
        else:
            new_avg = self._calculate_weighted_average(rollover.new_contract_new_data, price_field=self.new_price_field)

        if old_avg == 0:
            return 1.0, 0.0
        
        adjustment = new_avg / old_avg
        gap = new_avg - old_avg
        
        return adjustment, gap
    
    def _calculate_weighted_average(self, df: pd.DataFrame, price_field: str) -> float:
        if df.empty:
            return 0.0
        
        # 如果有成交量权重，使用加权平均
        if self.weights_by_time:
            total_weight = 0
            weighted_sum = 0
            for _, row in df.iterrows():
                ts = row[self.time_field]
                price = row[price_field]
                weight = self.weights_by_time.get(ts, 1.0)
                weighted_sum += price * weight
                total_weight += weight
            
            if total_weight > 0:
                return weighted_sum / total_weight
        
        # 否则使用简单平均
        all_prices = [row[price_field] for _, row in df.iterrows()]
        return sum(all_prices) / len(all_prices)
    
    def is_valid(self, rollover: 'ContractRollover') -> Tuple[bool, ValidityStatus]:
        if self.new_price_old_data_bool:
            if len(rollover.old_contract_old_data) == 0 or len(rollover.new_contract_old_data) == 0:
                return False, ValidityStatus.INSUFFICIENT_DATA
        else:
            if len(rollover.old_contract_old_data) == 0 or len(rollover.new_contract_new_data) == 0:
                return False, ValidityStatus.INSUFFICIENT_DATA
        return True, ValidityStatus.VALID
    
    def get_name(self) -> str:
        return self.name

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
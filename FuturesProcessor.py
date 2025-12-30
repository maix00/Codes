"""
本模块提供期货合约切换点检测的基础类 FuturesProcessorBase 及其相关工具方法，旨在为不同类型的切换点检测器提供统一的数据表管理、列名映射、数据校验和切换点提取等通用能力。

主要功能包括：
1. 数据表管理：支持多种标准数据表（如 main_tick、date_main_close_last 等），可通过列名映射适配不同来源的数据格式，便于灵活加载和处理。
2. 列结构校验：自动检查数据表是否包含必需列和期望列，确保后续检测逻辑的正确性和健壮性。
3. 切换点检测接口：定义 detect_rollover_points 等方法，要求子类实现具体的切换点检测逻辑，支持多种检测策略扩展。
4. 合约数据提取辅助：提供辅助方法，便于从主数据表中提取特定合约在切换点前后的连续数据片段，用于切换事件的构建和验证。
5. 数据表可用性检查：支持列出、校验和获取标准化后的数据表，便于调试和上层逻辑调用。

输入参数：
- column_mapping：可选，字典类型，为每个数据表指定用户列名到标准列名的映射关系。
- data_tables：可选，字典类型，直接传入各标准表名对应的 pandas.DataFrame 数据。

输出能力：
- 可通过 get_mapped_table 方法获取标准化后的数据表。
- 可通过 detect_rollover_points 方法（需子类实现）输出合约切换点列表（ContractRollover 对象）。
- 提供辅助方法输出数据表的可用性、列结构校验结果等信息。

适用场景：
本基类适用于期货主力合约切换点的自动检测，便于扩展不同数据结构和检测策略的子类实现，提升数据处理的灵活性和可维护性。
"""

import pandas as pd
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict
# from ContractRollover import ContractRollover
from StrategySelector import ProductPeriodStrategySelector
from tqdm import tqdm
import os
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

    # def _extract_contract_data(self, main_data: pd.DataFrame, contract: str, 
    #                          reference_time: datetime, is_old: bool) -> pd.DataFrame:
    #     """
    #     从数据中提取特定合约的数据，通过向上或向下遍历确保连续性
        
    #     Args:
    #         main_data: 数据
    #         contract: 合约代码
    #         reference_time: 参考时间点
    #         is_old: 是否为旧合约（True表示向上遍历，False表示向下遍历）
            
    #     Returns:
    #         特定合约的连续数据
            
    #     Raises:
    #         ValueError: 当找不到参考时间点的数据时
    #     """
    #     if main_data is None or main_data.empty:
    #         return pd.DataFrame()
            
    #     # 按时间排序
    #     main_data = main_data.sort_values('datetime').reset_index(drop=True)
        
    #     # 直接寻找对应参考时间点的数据
    #     exact_match = main_data[main_data['datetime'] == reference_time]
    #     if exact_match.empty:
    #         raise ValueError(f"在数据中找不到时间点 {reference_time} 的数据")
            
    #     reference_idx = exact_match.index[0]
        
    #     # 确认参考点是正确的合约
    #     if main_data.loc[reference_idx, 'symbol'] != contract:
    #         raise ValueError(f"时间点 {reference_time} 的数据合约 {main_data.loc[reference_idx, 'symbol']} 与目标合约 {contract} 不匹配")
        
    #     if is_old:
    #         # 对于旧合约，从参考时间点向上遍历（向过去遍历）直到symbol变化
    #         # 向上遍历找到该合约的起始位置
    #         start_idx = reference_idx
    #         while start_idx >= 0 and main_data.loc[start_idx, 'symbol'] == contract:
    #             start_idx -= 1
    #         start_idx += 1  # 回到第一个匹配的索引
            
    #         # 从起始位置到参考时间点就是我们需要的旧合约数据
    #         result_data = main_data.iloc[start_idx:reference_idx+1].copy()
            
    #     else:
    #         # 对于新合约，从参考时间点向下遍历（向未来遍历）直到symbol变化
    #         # 向下遍历找到该合约的结束位置
    #         end_idx = reference_idx
    #         while end_idx < len(main_data) and main_data.loc[end_idx, 'symbol'] == contract:
    #             end_idx += 1
                
    #         # 从参考时间点到结束位置就是我们需要的新合约数据
    #         result_data = main_data.iloc[reference_idx:end_idx].copy()
            
    #     if result_data.empty:
    #         print(f"  警告: 合约 {contract} 在指定方向上无连续数据")
    #         return pd.DataFrame()
            
    #     print(f"  提取到 {len(result_data)} 条{'旧' if is_old else '新'}合约 {contract} 的数据")
    #     return result_data.reset_index(drop=True)

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
    
    @staticmethod
    def save_data_frame_to_path(data: pd.DataFrame, save_path: Optional[str] = None):
        '''
        将DataFrame保存到指定路径的文件中，支持csv和parquet格式。
        
        参数:
            data (pd.DataFrame): 需要保存的DataFrame数据。
            save_path (str): 保存文件的完整路径，支持以.csv或.parquet结尾的文件名。
        '''
        if save_path is not None:
            if not os.path.exists(os.path.dirname(save_path)):
                os.makedirs(os.path.dirname(save_path))
            if save_path.endswith('.csv'):
                data.to_csv(save_path, index=False)
            elif save_path.endswith('.parquet'):
                # Convert Enum columns to string before saving to parquet
                for col in data.columns:
                    if data[col].dtype == 'object' and data[col].apply(lambda x: hasattr(x, 'name')).any():
                        data = data.copy()
                        data[col] = data[col].apply(lambda x: x.name if hasattr(x, 'name') else x)
                data.to_parquet(save_path, index=False)
            else:
                raise ValueError("仅支持保存为csv或parquet格式文件")
        
    def generate_main_contract_series_adjusted(self, 
                                               data: Optional[pd.DataFrame] = None, save_path: Optional[str] = None,
                                               uid_col: str = 'unique_instrument_id', time_col: str = 'trade_time', rollover_idx_col: str = 'trading_day',
                                               price_cols: List[str] = ['open_price', 'highest_price', 'lowest_price', 'close_price'],
                                               report_bool: bool = True, report_save_path: Optional[str] = None,
                                               plot_bool: bool = False, plot_col_name: str = 'close_price', plot_save_path: Optional[str] = None,
                                               plot_start_date: Optional[datetime|str] = None, plot_end_date: Optional[datetime|str] = None,
                                               plot_sample_step: Optional[int] = None, plot_max_points: int = 5000) -> pd.DataFrame:
        '''
        生成主力合约序列的复权行情数据（主力合约连续行情调整后数据）。

        本方法基于已有的主力合约行情数据表（'main_contract_series'），结合价格调整因子（adjustment_mul, adjustment_add），对指定的行情价格字段（如开盘价、最高价、最低价、收盘价）进行复权处理，生成调整后的主力合约连续行情序列。支持输出每个产品的复权数据统计报告，并可选地绘制复权前后的价格对比图。最终结果以DataFrame形式返回，并存入self.data_tables['main_contract_series_adjusted']。

        参数:
            data (Optional[pd.DataFrame]): 输入的主力合约行情数据表，若为None则自动读取self.data_tables['main_contract_series']。
            save_path (Optional[str]): 结果保存路径，支持csv或parquet格式。
            price_cols (List[str]): 需要进行复权调整的行情价格字段列表，默认为['open_price', 'highest_price', 'lowest_price', 'close_price']。
            report_bool (bool): 是否输出每个产品的复权后行情统计报告，默认为True。
            report_save_path (Optional[str]): 复权统计报告保存路径。
            plot_bool (bool): 是否绘制复权前后价格对比图，默认为False。
            plot_col_name (str): 绘图时使用的价格字段，默认为'close_price'。
            plot_save_path (Optional[str]): 绘图文件保存文件夹路径。
            plot_start_date (Optional[datetime|str]): 绘图起始日期，可选。
            plot_end_date (Optional[datetime|str]): 绘图结束日期，可选。
            plot_sample_step (Optional[int]): 绘图采样步长，减少点数以加快绘图速度。
            plot_max_points (int): 单张图最大采样点数，默认为5000。

        返回值:
            pd.DataFrame: 返回复权调整后的主力合约连续行情数据表。每行包括原始行情数据及对应的复权行情字段（如open_price_adjusted等）。

        异常:
            ValueError:
            - 当未找到'main_contract_series'数据表且未传入data参数时抛出。
            AssertionError:
            - 当输入数据缺少adjustment_mul或adjustment_add字段时抛出。
            - 结果表未成功写入self.data_tables时抛出。

        注意事项:
            - 输入数据需包含adjustment_mul和adjustment_add两列，分别为价格的乘法和加法调整因子。
            - 复权处理方式为：adjusted_price = 原价 * adjustment_mul + adjustment_add。
            - 若report_bool为True，将按unique_instrument_id分组输出每个产品的复权行情统计信息（最大值、最小值、有效行数等），并可保存到指定路径。
            - 若plot_bool为True，将为每个产品绘制复权前后价格对比图，自动标注合约切换点。
            - 结果表会自动写入self.data_tables['main_contract_series_adjusted']，便于后续分析与调用。
        '''
        if data is None:
            if 'main_contract_series' not in self.data_tables:
                raise ValueError("'main_contract_series'数据表不存在，请先运行generate_main_contract_series方法")
            data = self.data_tables['main_contract_series']
        
        assert 'adjustment_mul' in data.columns and 'adjustment_add' in data.columns

        adjusted_data = data.copy()

        # 删除所有adjusted列全为NaN的行
        adjusted_col_names = ['adjustment_mul', 'adjustment_add']
        if adjusted_col_names:
            adjusted_data = adjusted_data.dropna(subset=adjusted_col_names, how='all')

        assert uid_col in data.columns
        assert uid_col in adjusted_data.columns
        
        # 计算删除行前后的product_id列表差异
        removed_product_ids = sorted(list(set(data[uid_col].unique()) - set(adjusted_data[uid_col].unique())))
        if removed_product_ids:
            print("删除没有adjustment值的行后减少的product_id列表:", removed_product_ids)

        for col in price_cols:
            if col in adjusted_data.columns:
                adjusted_col_name = col + '_adjusted'
                adjusted_data[adjusted_col_name] = adjusted_data[col] * adjusted_data['adjustment_mul'] + adjusted_data['adjustment_add']

        if report_bool:
            adjusted_data_grouped = adjusted_data.groupby(uid_col)
            all_reports = []
            for product_id, group in tqdm(adjusted_data_grouped, desc="Generating adjustment reports"):
                adjusted_rows = group[price_cols].notnull().all(axis=1)
                report = pd.DataFrame({'product_id': [product_id],
                                        'num_rows': [len(group)]})
                if adjusted_rows.any():
                    first_idx = adjusted_rows.idxmax()
                    last_idx = adjusted_rows[::-1].idxmax()
                    report['adjusted_start_date'] = group.loc[first_idx, rollover_idx_col]
                    report['adjusted_end_date'] = group.loc[last_idx, rollover_idx_col]
                else:
                    report['adjusted_start_date'] = None
                    report['adjusted_end_date'] = None
                
                for col in price_cols:
                    adjusted_col_name = col + '_adjusted'
                    if adjusted_col_name in group.columns:
                        max_adjusted = group[adjusted_col_name].max()
                        min_adjusted = group[adjusted_col_name].min()
                        report[adjusted_col_name + '_max'] = max_adjusted
                        report[adjusted_col_name + '_min'] = min_adjusted
                all_reports.append(report)
            if all_reports:
                if report_save_path is not None:
                    self.save_data_frame_to_path(pd.concat(all_reports, ignore_index=True), report_save_path)

        if plot_bool:
            self.plot_adjusted(data=data, adjusted_data=adjusted_data, uid_col=uid_col, time_col=time_col,
                               rollover_idx_col=rollover_idx_col, plot_col_name=plot_col_name,
                               plot_save_path=plot_save_path, plot_start_date=plot_start_date,
                               plot_end_date=plot_end_date, plot_sample_step=plot_sample_step,
                               plot_max_points=plot_max_points)

        self.add_data_table('main_contract_series_adjusted', adjusted_data)
        assert 'main_contract_series_adjusted' in self.data_tables
        assert not self.data_tables['main_contract_series_adjusted'].empty
        self.save_data_frame_to_path(self.data_tables['main_contract_series_adjusted'], save_path)
        return self.data_tables['main_contract_series_adjusted']
    
    def plot_adjusted(self, data: pd.DataFrame, adjusted_data: pd.DataFrame, uid_col: str = 'unique_instrument_id',
                      time_col: str = 'trade_time', rollover_idx_col: str = 'trading_day',
                      plot_col_name: str = 'close_price', plot_save_path: Optional[str] = None,
                      plot_start_date: Optional[datetime|str] = None, plot_end_date: Optional[datetime|str] = None,
                      plot_sample_step: Optional[int] = None, plot_max_points: int = 5000):
        '''绘制复权前后的价格对比图，标注合约切换点。'''
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates

        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Microsoft YaHei', 'SimHei', 'Heiti TC', 'STHeiti', 'PingFang SC']
        plt.rcParams['axes.unicode_minus'] = False

        adjusted_data_grouped = adjusted_data.groupby(uid_col)

        if plot_start_date:
            plot_start_date = pd.to_datetime(plot_start_date)
        if plot_end_date:
            plot_end_date = pd.to_datetime(plot_end_date)

        for uid, plot_data in tqdm(adjusted_data_grouped, desc="Plotting adjusted prices"):

            assert type(uid) == str
            
            plot_data[time_col] = pd.to_datetime(plot_data[time_col])
            if plot_start_date:
                plot_data = plot_data[plot_data[time_col] >= plot_start_date]
            if plot_end_date:
                plot_data = plot_data[plot_data[time_col] <= plot_end_date]

            # 确保plot_sampled中包含所有rollover_points的old_contract_end_date和new_contract_start_date
            if not self.rollover_points or uid not in self.rollover_points:
                self.detect_rollover_points()
            rollover_dates = []
            for rollover in self.rollover_points[uid].values():
                assert rollover.old_contract_end_date is not None and rollover.new_contract_start_date is not None
                rollover_dates.append(pd.to_datetime(rollover.old_contract_end_date))
                rollover_dates.append(pd.to_datetime(rollover.new_contract_start_date))
            rollover_dates = [d for d in rollover_dates if not pd.isnull(d)]
            # 取所有需要保留的索引
            must_keep_idx = plot_data[plot_data[rollover_idx_col].isin(rollover_dates)].index.tolist()

            # 抽样以减少数据点，但保留所有rollover点
            step = max(len(plot_data) // plot_max_points, 1)
            sampled_idx = set(plot_data.iloc[::(plot_sample_step if plot_sample_step else step)].index.tolist())
            all_idx = sorted(set(sampled_idx).union(must_keep_idx))
            plot_sampled = plot_data.loc[all_idx]

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))

            # 先画切换点的线和文本（在价格线下方）
            # 先获取y轴范围（先画一条线获取ylim）
            ax.plot(plot_sampled[time_col], plot_sampled[plot_col_name], alpha=0)
            ylim = ax.get_ylim()
            for rollover in self.rollover_points[uid].values():
                assert rollover.new_contract_start_date is not None
                rv = pd.to_datetime(rollover.new_contract_start_date)

                # 检查切换点是否在绘图范围内
                if plot_start_date and rv < plot_start_date:
                    continue
                if plot_end_date and rv > plot_end_date:
                    continue
                
                rv_num = float(mdates.date2num(rv))
                ax.axvline(rv_num, color="#FF9999AB", linestyle='--', alpha=0.5, zorder=1)
                # ax.text(
                #     rv, ylim[1],
                #     f'{rollover.old_contract}→{rollover.new_contract}',
                #     rotation=90, va='top', ha='right', fontsize=8, color="#FF9999AB", alpha=0.6, zorder=1
                # )

            # 再画两条价格线（在切换点标记之上）
            ax.plot(plot_sampled[time_col], plot_sampled[plot_col_name], label='Original', color='blue', alpha=0.7, linewidth=1, zorder=2)
            ax.plot(plot_sampled[time_col], plot_sampled[plot_col_name + '_adjusted'], label='Adjusted', color='green', alpha=0.7, linewidth=1, zorder=2)

            # 智能设置x轴刻度
            self._set_smart_date_ticks(ax, plot_sampled[time_col])

            ax.set_title(f'{uid}: 原始与前复权连续合约价格 ({plot_col_name})')
            ax.set_xlabel('时间')
            ax.set_ylabel('价格')
            ax.legend()
            ax.grid(True, linestyle='--', alpha=0.5)

            # plt.tight_layout()
            if plot_save_path is not None:
                if not os.path.exists(plot_save_path):
                    os.makedirs(plot_save_path)
                plt.savefig(os.path.join(plot_save_path, f'{uid}_adjusted_prices.png'), dpi=200, bbox_inches='tight')
            plt.close()
    
    @staticmethod
    def _set_smart_date_ticks(ax, datetimes):
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
            formatter = mdates.DateFormatter('%H:%M')
        elif hours > 6:  # 超过6小时
            locator = mdates.HourLocator(interval=2)
            formatter = mdates.DateFormatter('%H:%M')
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

# class FuturesRolloverDetector_MainTick(FuturesProcessorBase):
#     """基于main_tick数据表的合约切换点检测器"""

#     @property
#     def EXPECTED_TABLE_NAMES(self) -> List[str]:
#         return ['main_tick']
    
#     @property
#     def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
#         return {
#             'main_tick': ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'position']
#         }

#     @property
#     def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
#         return {
#             'main_tick': ['datetime', 'symbol', 'close']
#         }
    
#     def detect_rollover_points(self) -> List[ContractRollover]:
#         """
#         基于main_tick数据表的切换点检测方法
        
#         Args:
#             data: 输入数据（用于检测symbol变化点）
            
#         Returns:
#             检测到的切换点列表
#         """
            
#         # 按时间排序
#         data = self.get_mapped_table('main_tick').sort_values('datetime').reset_index(drop=True)
        
#         # 检测symbol变化点
#         data['symbol_change'] = data['symbol'] != data['symbol'].shift(1)
#         change_indices = data[data['symbol_change']].index.tolist()
        
#         # 移除第一个点（因为是数据开始）
#         if change_indices and change_indices[0] == 0:
#             change_indices = change_indices[1:]
        
#         print(f"检测到 {len(change_indices)} 个合约切换点")
#         print(f"数据时间范围: {data['datetime'].min()} 到 {data['datetime'].max()}")
        
#         # 构建切换事件并进行验证
#         rollover_points = []
#         for idx in change_indices:
#             try:
#                 # 获取切换时间点
#                 rollover_time = data.iloc[idx]['datetime']
#                 old_contract = data.iloc[idx - 1]['symbol']
#                 new_contract = data.iloc[idx]['symbol']
                
#                 print(f"\n处理切换点 {rollover_time}: {old_contract} -> {new_contract}")
                
#                 # 从main_tick数据中提取旧合约和新合约的数据
#                 # 向上遍历旧合约数据直到symbol变化
#                 old_contract_data = self._extract_contract_data(data, old_contract, rollover_time, is_old=True)
#                 # 向下遍历新合约数据直到symbol变化
#                 new_contract_data = self._extract_contract_data(data, new_contract, rollover_time, is_old=False)
                
#                 # 创建切换事件
#                 rollover = ContractRollover(
#                     rollover_datetime=rollover_time,
#                     old_contract=old_contract,
#                     new_contract=new_contract,
#                     old_contract_old_data=old_contract_data,
#                     old_contract_new_data=pd.DataFrame(),
#                     new_contract_old_data=pd.DataFrame(),
#                     new_contract_new_data=new_contract_data,
#                 )
                
#                 rollover.is_valid = rollover.validate_data_tables(['old_contract_old_data', 'new_contract_new_data'])
                
#                 if rollover.is_valid:
#                     rollover_points.append(rollover)
#                 else:
#                     # 直接报错
#                     raise ValueError(f"切换点 {rollover.rollover_datetime} 无效")
                    
#             except Exception as e:
#                 print(f"处理切换点 {idx} 时出错: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         print(f"\n有效切换点: {len(rollover_points)}/{len(change_indices)}")
        
#         return rollover_points

# class FuturesRolloverDetector_MainTick_MainCloseLast(FuturesProcessorBase):
#     """使用main_tick和date_main_close_last两个表格的合约切换点检测器"""
    
#     @property
#     def EXPECTED_TABLE_NAMES(self) -> List[str]:
#         return ['main_tick', 'date_main_close_last']
    
#     @property
#     def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
#         return {
#             'main_tick': ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'position'],
#             'date_main_close_last': ['datetime', 'symbol', 'close', 'last_close']
#         }

#     @property
#     def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
#         return {
#             'main_tick': ['datetime', 'symbol', 'close'],
#             'date_main_close_last': ['datetime', 'symbol', 'close']
#         }

#     def detect_rollover_points(self) -> List[ContractRollover]:
#         """
#         增强切换点检测方法，使用main_tick和date_main_close_last两个表格
        
#         Returns:
#             检测到的切换点列表
#         """
#         # 验证必需的数据表是否可用
#         main_tick_data = self.get_mapped_table('main_tick')
#         date_main_close_last_data = self.get_mapped_table('date_main_close_last')
        
#         # 按时间排序
#         main_tick_data = main_tick_data.sort_values('datetime').reset_index(drop=True)
#         date_main_close_last_data = date_main_close_last_data.sort_values('datetime').reset_index(drop=True)

        
#         # 检测symbol变化点
#         main_tick_data['symbol_change'] = main_tick_data['symbol'] != main_tick_data['symbol'].shift(1)
#         change_indices = main_tick_data[main_tick_data['symbol_change']].index.tolist()
        
#         # 移除第一个点（因为是数据开始）
#         if change_indices and change_indices[0] == 0:
#             change_indices = change_indices[1:]
        
#         print(f"检测到 {len(change_indices)} 个合约切换点")
#         print(f"数据时间范围: {main_tick_data['datetime'].min()} 到 {main_tick_data['datetime'].max()}")
        
#         # 构建切换事件并进行验证
#         rollover_points = []
#         for idx in change_indices:
#             try:
#                 # 获取切换时间点
#                 rollover_time = main_tick_data.iloc[idx]['datetime']
#                 old_contract = main_tick_data.iloc[idx - 1]['symbol']
#                 new_contract = main_tick_data.iloc[idx]['symbol']
                
#                 print(f"\n处理切换点 {rollover_time}: {old_contract} -> {new_contract}")
                
#                 # 从main_tick数据中提取旧合约和新合约的数据
#                 # 向上遍历旧合约数据直到symbol变化
#                 old_contract_data = self._extract_contract_data(main_tick_data, old_contract, rollover_time, is_old=True)
#                 # 向下遍历新合约数据直到symbol变化
#                 new_contract_data = self._extract_contract_data(main_tick_data, new_contract, rollover_time, is_old=False)
                
#                 # 创建切换事件
#                 rollover = ContractRollover(
#                     rollover_datetime=rollover_time,
#                     old_contract=old_contract,
#                     new_contract=new_contract,
#                     old_contract_old_data=old_contract_data,
#                     old_contract_new_data=pd.DataFrame(),
#                     new_contract_old_data=pd.DataFrame(),
#                     new_contract_new_data=new_contract_data,
#                 )

#                 # Check rollover.new_contract_start_date is date not None
#                 if rollover.new_contract_start_date is None:
#                     raise ValueError(f"切换点 {rollover.rollover_datetime} 无效: 新合约 {rollover.new_contract} 的开始时间不能为None")

#                 rollover.new_contract_old_data = self._extract_new_contract_old_data(date_main_close_last_data, new_contract, rollover.new_contract_start_date)

#                 rollover.is_valid = rollover.validate_data_tables(['old_contract_old_data', 'new_contract_old_data', 'new_contract_new_data'])
                
#                 if rollover.is_valid:
#                     rollover_points.append(rollover)
#                 else:
#                     # 直接报错
#                     raise ValueError(f"切换点 {rollover.rollover_datetime} 无效")
                    
#             except Exception as e:
#                 print(f"处理切换点 {idx} 时出错: {e}")
#                 import traceback
#                 traceback.print_exc()
#                 continue
        
#         print(f"\n有效切换点: {len(rollover_points)}/{len(change_indices)}")
        
#         return rollover_points
    
#     def _extract_new_contract_old_data(self, date_data: pd.DataFrame, new_contract: str, 
#                                      reference_date: date) -> pd.DataFrame:
#         """
#         从date_main_close_last数据中提取新合约的旧数据
        
#         Args:
#             date_data: date_main_close_last数据
#             new_contract: 新合约代码
#             rollover_time: 切换时间点
            
#         Returns:
#             新合约的旧数据（前一日数据）
#         """
        
#         # 确保'datetime'列为pandas的datetime类型
#         if not pd.api.types.is_datetime64_any_dtype(date_data['datetime']):
#             date_data['datetime'] = pd.to_datetime(date_data['datetime'])
        
#         # 在date_main_close_last中查找对应日期和合约的数据
#         filtered_data = date_data[
#             (date_data['symbol'] == new_contract) & 
#             (date_data['datetime'].dt.date == pd.to_datetime(reference_date).date())
#         ]
        
#         if filtered_data.empty:
#             print(f"  警告: 未找到 {new_contract} 在 {reference_date} 的数据")
#             return pd.DataFrame()
            
#         print(f"  找到 {len(filtered_data)} 条 {new_contract} 在 {reference_date} 的数据")
#         return filtered_data.copy().reset_index(drop=True)

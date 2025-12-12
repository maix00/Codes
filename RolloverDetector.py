'''
本模块定义了期货合约切换点检测的基类 FuturesRolloverDetectorBase 及其相关方法，旨在为不同类型的合约切换点检测器提供统一的数据表管理、列名映射、数据校验和切换点提取的基础功能。
主要功能包括：
1. 数据表的加载与管理：支持多种标准数据表（如 main_tick、date_main_close_last 等），并可通过列名映射适配不同来源的数据格式。
2. 列结构校验：自动检查数据表是否包含所需的必需列和期望列，确保后续检测逻辑的正确性。
3. 合约切换点检测接口：定义了 detect_rollover_points 方法，要求子类实现具体的切换点检测逻辑。
4. 合约数据提取：提供辅助方法从主数据表中提取特定合约在切换点前后的连续数据片段，用于切换点事件的构建和验证。
5. 数据表可用性与完整性检查：支持列出、校验和获取标准化后的数据表，便于上层逻辑调用和调试。
输入：
- column_mapping: 可选，字典类型，为每个数据表指定用户列名到标准列名的映射关系。
- data_tables: 可选，字典类型，直接传入各标准表名对应的 pandas.DataFrame 数据。
输出：
- 支持通过 get_mapped_table 方法获取标准化后的数据表。
- 支持通过 detect_rollover_points 方法（需子类实现）输出合约切换点列表（ContractRollover 对象列表）。
- 支持辅助方法输出数据表的可用性、列结构校验结果等信息。
适用场景：
本基类适用于期货主力合约切换点的自动检测，便于扩展不同数据结构和检测策略的子类实现，提升数据处理的灵活性和健壮性。
'''

import pandas as pd
from datetime import datetime, date
from typing import List, Optional, Tuple, Dict
from ContractRollover import ContractRollover
    
class FuturesRolloverDetectorBase:
    """合约切换点检测器基类"""
    
    # 期望的数据表变量名（基类不定义，要求子类实现）
    # EXPECTED_TABLE_NAMES = ['main_tick', 'date_main_sub', 'date_main_close_last', 'product_contract_start_end', 'contract_tick']
    @property
    def EXPECTED_TABLE_NAMES(self) -> List[str]:
        raise NotImplementedError("请在子类中定义 EXPECTED_TABLE_NAMES 属性")
    
    # 期望的列结构（基类不定义，要求子类实现）
    # EXPECTED_COLUMNS = {
    #     'main_tick': ['datetime', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'position'],
    #     'date_main_sub': ['datetime', 'symbol', 'sub_symbol'],
    #     'date_main_close_last': ['datetime', 'symbol', 'close', 'last_close'],
    #     'product_contract_start_end': ['product', 'symbol', 'start_date', 'end_date'],
    #     'contract_tick': ['datetime', 'symbol']
    # }
    @property
    def EXPECTED_COLUMNS(self) -> Dict[str, List[str]]:
        raise NotImplementedError("请在子类中定义 EXPECTED_COLUMNS 属性")
    
    # 期望的必需列结构（基类不定义，要求子类实现）
    # REQUIRED_COLUMNS = {
    #     'main_tick': ['datetime', 'symbol', 'close'],
    #     'date_main_sub': ['datetime', 'symbol', 'sub_symbol'],
    #     'date_main_close_last': ['datetime', 'symbol', 'close'],
    #     'product_contract_start_end': ['product', 'symbol', 'start_date', 'end_date'],
    #     'contract_tick' : ['datetime', 'symbol']
    # }
    @property
    def REQUIRED_COLUMNS(self) -> Dict[str, List[str]]:
        raise NotImplementedError("请在子类中定义 REQUIRED_COLUMNS 属性")
    
    def __init__(self, 
                 column_mapping: Optional[Dict[str, Dict[str, str]]] = None,
                 data_tables: Optional[Dict[str, pd.DataFrame]] = None):
        """
        初始化FuturesRolloverDetectorBase

        Args:
            column_mapping: 每个数据表的列名映射字典
            data_tables: 直接提供的数据表字典
        """
        # 设置每个表的列名映射，默认为空字典
        self.column_mapping = column_mapping or {}
        # 存储所有合约的数据
        self.contract_data: Dict[str, pd.DataFrame] = {}
        # 存储数据表
        self.data_tables: Dict[str, pd.DataFrame] = {}
        # 存储Rollover点列表
        self.rollover_points: List[ContractRollover] = []
        
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
    
    def set_column_mapping(self, table_type: str, mapping: Dict[str, str]):
        """
        为特定数据表设置列名映射，并立即应用到已存在的数据表上
        
        Args:
            table_type: 数据表类型
            mapping: 列名映射字典
                    键为用户提供的列名，值为期望的列名
                    例如: {'时间': 'datetime', '合约': 'symbol', '开盘价': 'open'}
        """
        self.column_mapping[table_type] = mapping
        
        # 如果该表已存在，立即应用列名映射
        if table_type in self.data_tables:
            print(f"为数据表 '{table_type}' 应用新的列名映射")
            try:
                mapped_data = self._map_input_columns(self.data_tables[table_type], table_type)
                self.data_tables[table_type] = mapped_data
                # 自动检查列要求
                self._check_table_column_requirements(table_type)
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
    
    def detect_rollover_points(self, *args, **kwargs) -> List[ContractRollover]:
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

    def _extract_contract_data(self, main_data: pd.DataFrame, contract: str, 
                             reference_time: datetime, is_old: bool) -> pd.DataFrame:
        """
        从数据中提取特定合约的数据，通过向上或向下遍历确保连续性
        
        Args:
            main_data: 数据
            contract: 合约代码
            reference_time: 参考时间点
            is_old: 是否为旧合约（True表示向上遍历，False表示向下遍历）
            
        Returns:
            特定合约的连续数据
            
        Raises:
            ValueError: 当找不到参考时间点的数据时
        """
        if main_data is None or main_data.empty:
            return pd.DataFrame()
            
        # 按时间排序
        main_data = main_data.sort_values('datetime').reset_index(drop=True)
        
        # 直接寻找对应参考时间点的数据
        exact_match = main_data[main_data['datetime'] == reference_time]
        if exact_match.empty:
            raise ValueError(f"在数据中找不到时间点 {reference_time} 的数据")
            
        reference_idx = exact_match.index[0]
        
        # 确认参考点是正确的合约
        if main_data.loc[reference_idx, 'symbol'] != contract:
            raise ValueError(f"时间点 {reference_time} 的数据合约 {main_data.loc[reference_idx, 'symbol']} 与目标合约 {contract} 不匹配")
        
        if is_old:
            # 对于旧合约，从参考时间点向上遍历（向过去遍历）直到symbol变化
            # 向上遍历找到该合约的起始位置
            start_idx = reference_idx
            while start_idx >= 0 and main_data.loc[start_idx, 'symbol'] == contract:
                start_idx -= 1
            start_idx += 1  # 回到第一个匹配的索引
            
            # 从起始位置到参考时间点就是我们需要的旧合约数据
            result_data = main_data.iloc[start_idx:reference_idx+1].copy()
            
        else:
            # 对于新合约，从参考时间点向下遍历（向未来遍历）直到symbol变化
            # 向下遍历找到该合约的结束位置
            end_idx = reference_idx
            while end_idx < len(main_data) and main_data.loc[end_idx, 'symbol'] == contract:
                end_idx += 1
                
            # 从参考时间点到结束位置就是我们需要的新合约数据
            result_data = main_data.iloc[reference_idx:end_idx].copy()
            
        if result_data.empty:
            print(f"  警告: 合约 {contract} 在指定方向上无连续数据")
            return pd.DataFrame()
            
        print(f"  提取到 {len(result_data)} 条{'旧' if is_old else '新'}合约 {contract} 的数据")
        return result_data.reset_index(drop=True)

# class FuturesRolloverDetector_MainTick(FuturesRolloverDetectorBase):
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

# class FuturesRolloverDetector_MainTick_MainCloseLast(FuturesRolloverDetectorBase):
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

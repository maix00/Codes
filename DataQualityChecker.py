import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Callable, Optional, Tuple

from ContractRollover import ContractRollover

class DataQualityChecker:
    """
    数据质量检查器类，用于处理带有symbol字段的DataFrame数据。
    如果存在symbol字段，将对同一symbol的连续数据段分别处理。
    """
    
    def __init__(self, df: Optional[pd.DataFrame] = None, window_size_multiple: float = 3.0):
        """
        初始化数据质量检查器
        
        Args:
            df: 输入的DataFrame，用于在初始化时判断列的存在性和计算最小时间间隔
            window_size_multiple: 窗口大小倍数，用于判断零值序列是否过长，默认值为3（即3倍最小时间间隔）
        """
        self.window_size_multiple = window_size_multiple
        self.has_datetime = False  # 记录是否包含datetime列
        self.has_date = False      # 记录是否包含date列
        self.has_symbol = False    # 记录是否包含symbol列
        self.unique_symbol = True
        self.time_col: Optional[str] = None
        self.min_time_interval: Optional[timedelta] = None  # 记录最小时间间隔（timedelta对象）
        self.window_size_seconds : Optional[float] = None
        self.zero_handler: Optional[Callable[[pd.DataFrame, str, int, int], pd.DataFrame]] = None
        self.long_zero_handler: Optional[Callable[[pd.DataFrame, str, int, int], pd.DataFrame]] = None
        
        # 如果提供了DataFrame，在初始化时就判断列的存在性和计算最小时间间隔
        if df is not None:
            self.has_datetime = 'datetime' in df.columns
            self.has_date = 'date' in df.columns
            self.has_symbol = 'symbol' in df.columns
            self.unique_symbol = df['symbol'].nunique() == 1
            self.time_col = 'datetime' if self.has_datetime else 'date' if self.has_date else None
            self.min_time_interval = self._calculate_min_time_interval(df)
            if self.min_time_interval:
                self.window_size_seconds = self.min_time_interval.total_seconds() * self.window_size_multiple
            else:
                self.window_size_seconds = None
    
    def _calculate_min_time_interval(self, df: pd.DataFrame) -> Optional[timedelta]:
        """
        计算最小时间间隔（timedelta对象）
        
        Args:
            df: DataFrame
            
        Returns:
            最小时间间隔（timedelta对象），如果无法计算则返回None
        """
        if not self.time_col:
            return None
            
        try:
            # 尝试转换为datetime
            if df[self.time_col].dtype == 'object':
                dt_series = pd.to_datetime(df[self.time_col])
            else:
                dt_series = df[self.time_col]
                
            # 计算时间间隔
            if len(dt_series) > 1:
                time_diffs = dt_series.diff().dropna()
                print("唯一时间间隔：", time_diffs.unique())
                min_diff_seconds = time_diffs.dt.total_seconds().min()
                return timedelta(seconds=min_diff_seconds) if len(time_diffs) > 0 else None
                
            return None
        except:
            return None
        
    def process_dataframe(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        处理DataFrame数据
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 检查是否有symbol列
        if 'symbol' in df.columns:
            # 按symbol分组处理
            return self._process_with_symbol(df)
        else:
            # 直接处理整个DataFrame
            return self._process_without_symbol(df)
    
    def _process_with_symbol(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        按symbol分段处理数据
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            处理后的DataFrame
        """
        # 先按时间排序（优先使用datetime，如果没有则使用date）
        if self.time_col:
            df = df.sort_values(self.time_col).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
            
        # 根据连续的相同symbol进行分组
        # 创建一个组标识符，当symbol发生变化时增加
        df['_group_id'] = (df['symbol'] != df['symbol'].shift()).cumsum()
        
        # 对每个组分别处理
        processed_groups = []
        for group_id, group in df.groupby('_group_id'):
            # 删除辅助列
            group_data = group.drop('_group_id', axis=1)
            # 对每组数据应用_process_segment处理
            processed_group = self._process_segment(group_data)
            if processed_group is not None and not processed_group.empty:
                processed_groups.append(processed_group)
        
        # 删除辅助列
        df = df.drop('_group_id', axis=1)
        
        # 将所有处理完的数据拼接返回
        if processed_groups:
            return pd.concat(processed_groups, ignore_index=True)
        else:
            return pd.DataFrame()
    
    def _process_without_symbol(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        不按symbol处理数据
        
        Args:
            df: 输入的DataFrame
            
        Returns:
            处理后的DataFrame
        """
        return self._process_segment(df)
    
    def _process_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理单个数据段
        
        Args:
            df: 输入的DataFrame段
            
        Returns:
            处理后的DataFrame段
        """
        df = df.copy()
        
        # 将所有含有数值的列转化为数值类型
        for col in df.columns:
            # 如果col中包含字符串'date'或者'symbol'则跳过
            if 'date' in col.lower() or 'symbol' in col.lower():
                continue
            # 尝试将列转换为数值类型
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass
        
        # 处理数值列中的零值
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
            # 如果col中包含字符串'date'或者'symbol'则跳过
            if 'date' in col.lower() or 'symbol' in col.lower():
                continue
                
            # 检测零值序列
            zero_sequences = self._detect_zero_sequences(df[col])
            
            for start_idx, end_idx in zero_sequences:
                sequence_length = end_idx - start_idx + 1
                
                # 计算时间跨度（如果可能）
                time_span_seconds = self._calculate_time_span(df, start_idx, end_idx)
                
                # 如果是非日期数据且零值序列超过窗口大小，则删除
                if time_span_seconds and self.window_size_seconds and time_span_seconds > self.window_size_seconds:
                    # 如果有自定义零值处理函数，则调用它
                    if hasattr(self, 'long_zero_handler') and self.long_zero_handler:
                        df = self.long_zero_handler(df, col, start_idx, end_idx)
                    else:
                        # 默认处理：删除长时间的零值序列
                        df = self._handle_long_zero_sequence(df, start_idx, end_idx, col)
                else:
                    # 如果有自定义零值处理函数，则调用它
                    if hasattr(self, 'zero_handler') and self.zero_handler:
                        df = self.zero_handler(df, col, start_idx, end_idx)
                    else:
                        # 默认处理：对其他零值进行线性插值
                        df = self._interpolate_zero_sequence(df, start_idx, end_idx, col)
        
        # 处理异常值
        df = self._handle_outliers(df, numeric_columns)
        
        return df
    
    def _detect_zero_sequences(self, series: pd.Series) -> List[Tuple[int, int]]:
        """
        检测零值序列的起止索引
        
        Args:
            series: 数值序列
            
        Returns:
            零值序列的起止索引列表 [(start_idx, end_idx), ...]
        """
        zero_sequences = []
        in_sequence = False
        start_idx = 0
        
        for i, value in enumerate(series):
            if pd.isna(value):
                # 将NaN视为零值
                value = 0.0
                
            if value == 0.0:
                if not in_sequence:
                    in_sequence = True
                    start_idx = i
            else:
                if in_sequence:
                    in_sequence = False
                    zero_sequences.append((start_idx, i - 1))
                    
        # 处理序列末尾的零值
        if in_sequence:
            zero_sequences.append((start_idx, len(series) - 1))
            
        return zero_sequences
    
    def _calculate_time_span(self, df: pd.DataFrame, start_idx: int, end_idx: int) -> Optional[float]:
        """
        计算指定索引范围内的时间跨度（秒）
        
        Args:
            df: DataFrame
            start_idx: 起始索引
            end_idx: 结束索引
            
        Returns:
            时间跨度（秒），如果无法计算则返回None
        """
        if self.time_col:
            return (df.iloc[end_idx][self.time_col] - df.iloc[start_idx][self.time_col]).total_seconds()
        else:
            return None
    
    def _handle_long_zero_sequence(self, df: pd.DataFrame, start_idx: int, end_idx: int, col: str) -> pd.DataFrame:
        """
        处理长时间的零值序列（删除）
        
        Args:
            df: DataFrame
            start_idx: 起始索引
            end_idx: 结束索引
            col: 列名
            
        Returns:
            处理后的DataFrame
        """
        # 这里应该创建ContractRoller对象，但因为我们没有实际的ContractRoller.py文件，
        # 我们只是简单地删除这些行
        print(f"删除 {col} 列中从索引 {start_idx} 到 {end_idx} 的长时间零值序列")
        # 在实际实现中，这里会创建ContractRoller对象
        # contract_roller = ContractRoller(old_contract=symbol, new_contract=symbol)
        # contract_roller.process()
        
        # 删除行
        return df.drop(df.index[start_idx:end_idx+1]).reset_index(drop=True)
    
    def _interpolate_zero_sequence(self, df: pd.DataFrame, start_idx: int, end_idx: int, col: str) -> pd.DataFrame:
        """
        对零值序列进行线性插值
        
        Args:
            df: DataFrame
            start_idx: 起始索引
            end_idx: 结束索引
            col: 列名
            
        Returns:
            处理后的DataFrame
        """
        # 获取序列前后的值
        prev_idx = start_idx - 1
        next_idx = end_idx + 1
        
        if prev_idx < 0 or next_idx >= len(df):
            # 边界情况，无法插值
            return df
            
        prev_value = df.iloc[prev_idx][col]
        next_value = df.iloc[next_idx][col]
        
        # 检查前后值是否有效
        if pd.isna(prev_value) or pd.isna(next_value):
            return df
            
        # 线性插值
        sequence_length = end_idx - start_idx + 1
        for i in range(sequence_length):
            idx = start_idx + i
            # 线性插值公式
            interpolated_value = prev_value + (next_value - prev_value) * (i + 1) / (sequence_length + 1)
            df.loc[idx, col] = interpolated_value
            
        return df
    
    def _handle_outliers(self, df: pd.DataFrame, numeric_columns: List[str]) -> pd.DataFrame:
        """
        处理异常值
        
        Args:
            df: DataFrame
            numeric_columns: 数值列列表
            
        Returns:
            处理后的DataFrame
        """
        for col in numeric_columns:
            if col in ['datetime', 'symbol']:
                continue
                
            # 使用IQR方法检测异常值
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # 定义异常值边界
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # 标记异常值
            outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
            
            # 如果有异常值处理函数，则调用它
            if hasattr(self, 'outlier_handler') and self.outlier_handler:
                df = self.outlier_handler(df, col, outlier_mask)
            else:
                # 默认处理：用中位数替换异常值
                median_value = df[col].median()
                df.loc[outlier_mask, col] = median_value
                
        return df
    
    def set_outlier_handler(self, handler: Callable[[pd.DataFrame, str, pd.Series], pd.DataFrame]):
        """
        设置异常值处理函数
        
        Args:
            handler: 异常值处理函数，接受(df, column_name, outlier_mask)参数，返回处理后的DataFrame
        """
        self.outlier_handler = handler

    def set_long_zero_handler(self, handler: Callable[[pd.DataFrame, str, int, int], pd.DataFrame]):
        """
        设置长零值处理函数
        
        Args:
            handler: 零值处理函数，接受(df, column_name, start_idx, end_idx)参数，返回处理后的DataFrame
        """
        self.long_zero_handler = handler
    
    def set_zero_handler(self, handler: Callable[[pd.DataFrame, str, int, int], pd.DataFrame]):
        """
        设置零值处理函数
        
        Args:
            handler: 零值处理函数，接受(df, column_name, start_idx, end_idx)参数，返回处理后的DataFrame
        """
        self.zero_handler = handler


import pandas as pd
import numpy as np
from datetime import timedelta
from typing import Any, List, Optional, Dict, Callable, Tuple
from enum import Enum

class DataIssueLabel(Enum):
    MISSING_VALUES = "Missing Values"
    ZERO_SEQUENCE_LONG = "Zero Sequence Long"
    ZERO_SEQUENCE_SHORT = "Zero Sequence Short"
    ZERO_SEQUENCE_ALL = "Zero Sequence All"
    ZERO_SEQUENCE_AT_START = "Zero Sequence at Start"
    ZERO_SEQUENCE_AT_VERY_START = "Zero Sequence at Very Start"
    ZERO_SEQUENCE_AT_END = "Zero Sequence at End"
    OUTLIERS_MAD = "Outliers MAD"
    # DUPLICATE_ENTRIES = "Duplicate Entries"

class DataIssueSolution(Enum):
    # General Solutions
    NO_ACTION = "No Action"
    ERROR = "Error"
    DROP_ROWS = "Drop Rows"
    FORWARD_FILL = "Forward Fill"
    BACKWARD_FILL = "Backward Fill"
    LINEAR_INTERPOLATION = "Linear Interpolation"
    
    # Outliers solutions
    OUTLIERS_MED_CLIP = "Clip to Median"
    OUTLIERS_MEDIAN = "Replace with Median"
    OUTLIERS_MEAN = "Replace with Mean"
    OUTLIERS_DROP = "Drop Rows"
    OUTLIERS_CAP = "Cap at Bounds"
    
    # # Duplicate Entries solutions
    # DUPLICATE_ENTRIES_KEEP_FIRST = "Keep First"
    # DUPLICATE_ENTRIES_KEEP_LAST = "Keep Last"

class DataQualityChecker:
    """
    数据质量检查器类，用于处理带有symbol字段的DataFrame数据。
    如果存在symbol字段，将对同一symbol的连续数据段分别处理。
    """

    EXPECTED_COLUMNS = ['symbol', 'time']
    
    def __init__(self, df: Optional[pd.DataFrame] = None, window_size_multiple: float = 3.0, 
                 columns: Optional[List[str]] = None,
                 print_info: bool = False,
                 label: str = 'futures',
                 column_mapping: Optional[Dict[str, str]] = None):
        """
        初始化数据质量检查器
        
        Args:
            df (Optional[pd.DataFrame]): 输入的DataFrame，用于在初始化时判断列的存在性和计算最小时间间隔
            window_size_multiple (float): 窗口大小倍数，用于判断零值序列是否过长，默认值为3.0
            columns (Optional[List[str]]): 要处理的列名列表，默认为None（处理所有列）
            print_info (bool): 是否打印信息，默认为False
            label (str): 数据标签，默认为'futures'
        """
        if df is None:
            raise ValueError("Input DataFrame cannot be None.")
        self.df = df  # 存储DataFrames
        self.window_size_multiple = window_size_multiple
        self.columns = columns
        self.label = label

        self.issues_df: pd.DataFrame = pd.DataFrame(
            columns=['SYMBOL', 'SYMBOL_GROUP_ID', 'COLUMN', 'START_ROW', 'START_TIME', 'END_ROW', 'END_TIME', 
                     'ISSUE_LABEL', 'SOLUTION_LABEL', 'DETAILS'])  # 记录问题的DataFrame

        self.symbol_col = 'symbol' if 'symbol' in df.columns else (column_mapping['symbol'] if column_mapping and 'symbol' in column_mapping else None)
        
        time_candidates = ['time', 'day', 'date']
        self.time_col = None
        for col in time_candidates:
            matches = [c for c in df.columns if col.lower() in c.lower()]
            if matches:
                self.time_col = matches[0]
            break
        self.time_col = column_mapping['time'] if column_mapping and 'time' in column_mapping else self.time_col
        
        self.min_time_interval = self._calculate_min_time_interval(df)
        if self.min_time_interval:
            self.window_size_seconds = self.min_time_interval.total_seconds() * self.window_size_multiple
        else:
            self.window_size_seconds = None
        
        self.solution_handlers: Dict[DataIssueSolution, Callable[[pd.DataFrame, int, int, str, Optional[Dict[str, Any]]], pd.DataFrame]] = {
            DataIssueSolution.NO_ACTION: lambda df, start_idx, end_idx, col, issue=None: df,
            DataIssueSolution.ERROR: lambda df, start_idx, end_idx, col, issue: (
                (_ for _ in ()).throw(ValueError(f"Data quality error in column '{col}': {issue['ISSUE_LABEL'].name} - {issue['DETAILS']}")) if issue else df
            ),
            DataIssueSolution.DROP_ROWS: lambda df, start_row, end_row, _, issue=None: df.drop(df.index[start_row:end_row+1]).reset_index(drop=True),
            DataIssueSolution.LINEAR_INTERPOLATION: lambda df, start_idx, end_idx, col, issue=None: (
                df.assign(**{col: df[col].mask(df.index.isin(range(start_idx, end_idx+1))).interpolate(method='linear')})),
            DataIssueSolution.FORWARD_FILL: lambda df, start_idx, end_idx, col, issue=None: (
                df.assign(**{col: df[col].mask(df.index.isin(range(start_idx, end_idx+1))).ffill()})),
            DataIssueSolution.BACKWARD_FILL: lambda df, start_idx, end_idx, col, issue=None: (
                df.assign(**{col: df[col].mask(df.index.isin(range(start_idx, end_idx+1))).bfill()})),
            DataIssueSolution.OUTLIERS_MEDIAN: lambda df, start_idx, end_idx, col, issue=None: (
            df.assign(**{col: df[col].mask((df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | 
                (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))),
                df[col].median())})
            ),
        }

        # 将所有含有数值的列转化为数值类型
        for col in df.columns if self.columns is None else self.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

        # 如果DataFrame中没有symbol列，则添加一个默认的symbol列用于后续分组处理
        if not self.symbol_col:
            self.df['symbol'] = '[SYMBOL]'
            self.symbol_col = 'symbol'

        # 先按时间排序
        if self.time_col:
            self.df = self.df.sort_values(self.time_col).reset_index(drop=True)
        else:
            self.df = self.df.reset_index(drop=True)

        # 创建辅助列_group_id，用于标识同一symbol的连续数据段
        # 如果_group_id列已存在，将其重命名
        if '_group_id' in self.df.columns:
            self.df = self.df.rename(columns={'_group_id': '_group_id_original'})
        self.df['_group_id'] = (self.df[self.symbol_col] != self.df[self.symbol_col].shift()).cumsum()

        for _, group_data in self.df.groupby('_group_id'):
            # 对每组数据应用_check_segment
            self._check_segment(group_data)
    
        # 根据issue_label分配solution_label
        self._assign_solution_by_issue_label()

        # 按START_TIME倒序排列，防止删除行时索引变化
        self.issues_df = self.issues_df.sort_values('START_TIME', ascending=False).reset_index(drop=True)

        if print_info:
            self.print_info()

        self.processed_df = pd.DataFrame()

    def _assign_solution_by_issue_label(self) -> None:
        """
        根据issue_label为issues_df中的每一行分配对应的solution_label
        子类可以重写此方法以自定义分配逻辑
        """
        solution_mapping = {
            DataIssueLabel.MISSING_VALUES: DataIssueSolution.NO_ACTION,
            DataIssueLabel.ZERO_SEQUENCE_LONG: DataIssueSolution.LINEAR_INTERPOLATION,
            DataIssueLabel.ZERO_SEQUENCE_SHORT: DataIssueSolution.LINEAR_INTERPOLATION,
            DataIssueLabel.ZERO_SEQUENCE_ALL: DataIssueSolution.ERROR,
            DataIssueLabel.ZERO_SEQUENCE_AT_START: DataIssueSolution.ERROR,
            DataIssueLabel.ZERO_SEQUENCE_AT_VERY_START: DataIssueSolution.DROP_ROWS,
            DataIssueLabel.ZERO_SEQUENCE_AT_END: DataIssueSolution.ERROR,
            DataIssueLabel.OUTLIERS_MAD: DataIssueSolution.OUTLIERS_MEDIAN,
        }
        
        self.issues_df['SOLUTION_LABEL'] = self.issues_df['ISSUE_LABEL'].map(solution_mapping)

    def save_dataframe(self, filepath: str, format: str = 'csv', save_issues: bool = False) -> None:
        """
        保存处理后的数据或问题记录DataFrame到指定文件
        
        Args:
            filepath (str): 文件保存路径
            format (str): 文件格式，支持 'csv', 'excel', 'parquet'，默认为 'csv'
            save_issues (bool): 是否保存issues_df，默认为False（保存处理后的数据）
        """
        df_to_save = self.issues_df if save_issues else self.processed_df

        if save_issues:
            df_to_save["ISSUE_LABEL"] = df_to_save["ISSUE_LABEL"].apply(lambda x: x.name if hasattr(x, 'name') else x)
            df_to_save["SOLUTION_LABEL"] = df_to_save["SOLUTION_LABEL"].apply(lambda x: x.name if hasattr(x, 'name') else x)
        
        if df_to_save is None or df_to_save.empty:
            print("DataFrame为空，无法保存")
            return
        
        try:
            if format.lower() == 'csv':
                df_to_save.to_csv(filepath, index=False)
            elif format.lower() == 'excel':
                df_to_save.to_excel(filepath, index=False)
            elif format.lower() == 'parquet':
                df_to_save.to_parquet(filepath, index=False)
            else:
                raise ValueError(f"不支持的格式: {format}")
            
            print(f"数据已保存到: {filepath}")
        except Exception as e:
            print(f"保存数据时出错: {e}")

    def print_info(self) -> None:
        """
        打印最小时间间隔和问题记录DataFrame
        """
        print(f"最小时间间隔: {self.min_time_interval}")
        print("\n问题记录:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(self.issues_df.to_string(
            header=False,
            formatters={'ISSUE_LABEL': lambda x: x.name if hasattr(x, 'name') else x,
               'SOLUTION_LABEL': lambda x: x.name if hasattr(x, 'name') else x}
        ))
    
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
                df[self.time_col] = pd.to_datetime(df[self.time_col])

            dt_series = df[self.time_col]
                
            # 计算时间间隔
            if len(dt_series) > 1:
                time_diffs = dt_series.diff().dropna()
                min_diff_seconds = time_diffs.apply(lambda x: x.total_seconds()).min()
                return timedelta(seconds=min_diff_seconds) if len(time_diffs) > 0 else None
                
            return None
        except:
            return None
        
    def process_dataframe(self) -> Optional[pd.DataFrame]:
        """
        处理DataFrame数据
            
        Returns:
            处理后的DataFrame
        """
        
        # 对每个组分别处理
        processed_groups = []
        for group_id, group_data in self.df.groupby('_group_id'):
            filtered_issues = self.issues_df[self.issues_df['SYMBOL_GROUP_ID'] == group_id]
            df = group_data.copy()
            for _, issue in filtered_issues.iterrows():
                solution = issue['SOLUTION_LABEL']
                if solution in self.solution_handlers:
                    df = self.solution_handlers[solution](df, issue['START_ROW'], issue['END_ROW'], issue['COLUMN'], issue.to_dict())
            if df is not None and not df.empty:
                processed_groups.append(df)
        
        # 删除辅助列
        self.df = self.df.drop('_group_id', axis=1)
        if self.symbol_col == '[SYMBOL]':
            self.df = self.df.drop('symbol', axis=1)
        
        # 将所有处理完的数据拼接返回
        if processed_groups:
            self.processed_df = pd.concat(processed_groups, ignore_index=True)
        else:
            self.processed_df = pd.DataFrame()
        
        return self.processed_df
        
    def add_solution_handler(self, solution: DataIssueSolution, handler: Callable[[pd.DataFrame, int, int, str, Optional[Dict[str, Any]]], pd.DataFrame]) -> None:
        """
        增加或修改solution_handlers中的处理函数
        
        Args:
            solution (DataIssueSolution): 解决方案枚举值
            handler (Callable[[pd.DataFrame, int, int, str], pd.DataFrame]): 处理函数
        """
        self.solution_handlers[solution] = handler
    
    def _check_segment(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检查单个数据段是否存在零值、缺失值、异常值等问题
        
        Args:
            df: 输入的DataFrame段
            
        Returns:
            包含问题记录的DataFrame
        """
        
        numeric_columns = df[self.columns].select_dtypes(include=[np.number]).columns.tolist() if self.columns else df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numeric_columns:
                
            # 检测零值序列并记录
            zero_sequences = self._detect_zero_sequences(df[col])
            
            for start_idx, end_idx in zero_sequences:
                sequence_length = end_idx - start_idx + 1
                symbol_value = df.iloc[start_idx][self.symbol_col] if self.symbol_col else None
                symbol_group_id = df.iloc[start_idx]['_group_id'] if '_group_id' in df.columns else None
                time_span_seconds = self._calculate_time_span(df, start_idx, end_idx)
                start_time = df.iloc[start_idx][self.time_col] if self.time_col else None
                end_time = df.iloc[end_idx][self.time_col] if self.time_col else None
                details = f"长度为 {sequence_length}, 时间跨度为{time_span_seconds}s"

                # 检查是否触及边界，如果是则记录
                if start_idx == 0 or end_idx == len(df) - 1:
                    # 根据边界类型选择标签
                    if start_idx == 0 and end_idx == len(df) - 1:
                        issue_label = DataIssueLabel.ZERO_SEQUENCE_ALL
                        details = "零值序列覆盖整个数据段, " + details
                    elif start_idx == 0:
                        if '_group_id' in df.columns and df.iloc[start_idx]['_group_id'] == 1:
                            issue_label = DataIssueLabel.ZERO_SEQUENCE_AT_VERY_START
                            details = "零值序列起始于整个数据集的开始, " + details
                        else:
                            issue_label = DataIssueLabel.ZERO_SEQUENCE_AT_START
                            details = "零值序列起始于首行, " + details
                    else:
                        issue_label = DataIssueLabel.ZERO_SEQUENCE_AT_END
                        details = "零值序列结束于末行, " + details
                else:
                    # 如果是非日期数据且零值序列超过窗口大小，则记录问题
                    if time_span_seconds and self.window_size_seconds and time_span_seconds > self.window_size_seconds:
                        issue_label = DataIssueLabel.ZERO_SEQUENCE_LONG
                        details = f"超过窗口大小{self.window_size_seconds}s, " + details
                    else:
                        # 记录非边界零值序列问题，默认方案为线性插值
                        issue_label = DataIssueLabel.ZERO_SEQUENCE_SHORT

                new_issue = pd.DataFrame({
                    'SYMBOL': [symbol_value],
                    'SYMBOL_GROUP_ID': [symbol_group_id],
                    'COLUMN': [col],
                    'START_ROW': [start_idx],
                    'START_TIME': [start_time],
                    'END_ROW': [end_idx],
                    'END_TIME': [end_time],
                    'ISSUE_LABEL': [issue_label],
                    'DETAILS': [details]
                })
                # Only concatenate if new_issue is not empty and not all-NA
                if not new_issue.empty and not new_issue.isna().all(axis=None):
                    self.issues_df = pd.concat([self.issues_df, new_issue], ignore_index=True)
            
        # 检测缺失值并记录
        if self.time_col and self.min_time_interval:
            missing_sequences = self._detect_missing_sequences(df)
            
            for start_idx, end_idx in missing_sequences:
                symbol_value = df.iloc[start_idx][self.symbol_col] if self.symbol_col else None
                symbol_group_id = df.iloc[start_idx]['_group_id'] if '_group_id' in df.columns else None
                start_time = df.iloc[start_idx][self.time_col]
                end_time = df.iloc[end_idx][self.time_col]
                expected_count = int((end_time - start_time) / self.min_time_interval) + 1
                actual_count = end_idx - start_idx + 1
                missing_count = expected_count - actual_count
                
                new_issue = pd.DataFrame({
                    'SYMBOL': [symbol_value],
                    'SYMBOL_GROUP_ID': [symbol_group_id],
                    'COLUMN': [None],
                    'START_ROW': [start_idx],
                    'START_TIME': [start_time],
                    'END_ROW': [end_idx],
                    'END_TIME': [end_time],
                    'ISSUE_LABEL': [DataIssueLabel.MISSING_VALUES],
                    'DETAILS': [f"缺失 {missing_count} 行数据"]
                })
                self.issues_df = pd.concat([self.issues_df, new_issue], ignore_index=True)

            # 检测异常值并记录
            # outlier_mask = (df[col] < (df[col].quantile(0.25) - 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25)))) | \
            #                (df[col] > (df[col].quantile(0.75) + 1.5 * (df[col].quantile(0.75) - df[col].quantile(0.25))))

            # if outlier_mask.any():
            #     outlier_indices = df[outlier_mask].index.tolist()
            #     for idx in outlier_indices:
            #         new_issue = pd.DataFrame({
            #             'SYMBOL': [df.iloc[idx][self.symbol_col] if self.symbol_col else None],
            #             'SYMBOL_GROUP_ID': [df.iloc[idx]['_group_id'] if '_group_id' in df.columns else None],
            #             'COLUMN': [col],
            #             'START_ROW': [idx],
            #             'START_TIME': [df.iloc[idx][self.time_col] if self.time_col else None],
            #             'END_ROW': [idx],
            #             'END_TIME': [df.iloc[idx][self.time_col] if self.time_col else None],
            #             'ISSUE_LABEL': [DataIssueLabel.OUTLIERS],
            #             'DETAILS': [f"异常值: {df.iloc[idx][col]}"]
            #         })
            #         self.issues_df = pd.concat([self.issues_df, new_issue], ignore_index=True)
        
        return self.issues_df
    
    def _detect_missing_sequences(self, df: pd.DataFrame) -> List[Tuple[int, int]]:
        """
        根据时间间隔检测缺失数据序列
        
        Args:
            df: DataFrame
            col: 列名
            
        Returns:
            缺失序列的起止索引列表 [(start_idx, end_idx), ...]
        """
        missing_sequences = []
        
        if not self.time_col or not self.min_time_interval:
            return missing_sequences
        
        # 向量化计算时间间隔，提高速度
        time_series = pd.to_datetime(df[self.time_col])
        time_diffs = time_series.diff().fillna(pd.Timedelta(seconds=0))
        expected_interval = self.min_time_interval

        j = 0
        for i in range(len(df) - 1):
            if i < j:
                continue
            actual_interval = time_diffs.iloc[i + 1]
            current_time = time_series.iloc[i]
            next_time = time_series.iloc[i + 1]
            
            if actual_interval > expected_interval:
                # 排除前后交易日切换的情况
                current_date = current_time.date() if hasattr(current_time, 'date') else pd.to_datetime(current_time).date()
                next_date = next_time.date() if hasattr(next_time, 'date') else pd.to_datetime(next_time).date()
                if current_date != next_date:
                    continue  # 第一种情形，直接跳过
                else:
                    # 第二种情形，排除每日特定时间点切换的情况（如9点、12点）
                    current_hour = current_time.hour if hasattr(current_time, 'hour') else pd.to_datetime(current_time).hour
                    next_hour = next_time.hour if hasattr(next_time, 'hour') else pd.to_datetime(next_time).hour
                    if (current_hour < 12 and next_hour >= 12) or (current_hour < 9 and next_hour >= 9):
                        continue

                    # 第三种情形，期货在十点十五分到十点三十分休市
                    if self.label == 'futures' and current_hour == 10 and current_time.minute >= 15 and next_hour == 10 and next_time.minute <= 31:
                        continue

                    # 第四种情形，检查next_time开始的连续无missing数据的最后一个日期
                    j = i + 1
                    next_this_date = current_date  # 初始化，避免未绑定错误
                    while j < len(df) - 1:
                        # 用time_diffs直接判断间隔和日期
                        next_this_time = time_series.iloc[j + 1]
                        next_this_date = next_this_time.date() if hasattr(next_this_time, 'date') else pd.to_datetime(next_this_time).date()
                        interval = time_diffs.iloc[j + 1]
                        if interval > expected_interval or next_this_date != current_date:
                            break
                        j += 1
                    if next_this_date != current_date:
                        continue
                missing_sequences.append((i, i + 1))
        
        return missing_sequences

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
            start_time = pd.to_datetime(df.iloc[start_idx][self.time_col])
            end_time = pd.to_datetime(df.iloc[end_idx][self.time_col])
            time_span = (end_time - start_time)
            if self.min_time_interval is not None:
                time_span += self.min_time_interval
            return time_span.total_seconds()
        else:
            return None
    
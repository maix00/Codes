import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import warnings

@dataclass
class DataQualityIssue:
    """数据质量问题记录"""
    issue_type: str
    description: str
    timestamp: datetime
    contract: str
    severity: str  # low, medium, high
    action_taken: str

@dataclass
class ContractRollover:
    """合约切换事件的数据结构"""
    rollover_datetime: datetime
    old_contract: str
    new_contract: str
    old_close: float
    new_open: float
    price_gap: float
    adjustment_factor: float
    is_valid: bool = True  # 标记切换点是否有效
    
    def __post_init__(self):
        """计算价格差距和调整因子"""
        self.price_gap = self.new_open - self.old_close
        self.adjustment_factor = self.new_open / self.old_close if self.old_close != 0 else 1.0


class RolloverDetector:
    """合约切换点检测器"""
    
    def __init__(self, min_volume: float = 0, price_change_threshold: float = 0.5):
        self.min_volume = min_volume
        self.price_change_threshold = price_change_threshold  # 价格变化阈值，用于验证切换点
    
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
        
        # 构建切换事件并进行验证
        rollover_points = []
        for idx in change_indices:
            try:
                # 获取切换前后的数据
                old_row = data.iloc[idx - 1]
                new_row = data.iloc[idx]
                
                # 创建切换事件
                rollover = ContractRollover(
                    rollover_datetime=new_row['datetime'],
                    old_contract=old_row['symbol'],
                    new_contract=new_row['symbol'],
                    old_close=old_row['close'],
                    new_open=new_row['open'],
                    price_gap=0,
                    adjustment_factor=1.0
                )
                
                # 验证切换点
                rollover.is_valid = self._validate_rollover(rollover, old_row, new_row)
                
                rollover_points.append(rollover)
                
                status = "有效" if rollover.is_valid else "可疑"
                print(f"切换点 {rollover.rollover_datetime}: {rollover.old_contract} -> {rollover.new_contract}, "
                      f"价差: {rollover.price_gap:.2f}, 状态: {status}")
                      
            except Exception as e:
                print(f"处理切换点 {idx} 时出错: {e}")
                continue
        
        # 统计有效切换点
        valid_count = sum(1 for rp in rollover_points if rp.is_valid)
        print(f"有效切换点: {valid_count}/{len(rollover_points)}")
        
        return rollover_points
    
    def _validate_rollover(self, rollover: ContractRollover, old_row: pd.Series, new_row: pd.Series) -> bool:
        """验证切换点是否有效"""
        # 检查价格是否合理
        if rollover.old_close <= 0 or rollover.new_open <= 0:
            print(f"  警告: 价格异常 - 旧合约收盘: {rollover.old_close}, 新合约开盘: {rollover.new_open}")
            return False
        
        # 检查价格变化是否在合理范围内
        price_change_ratio = abs(rollover.price_gap) / rollover.old_close
        if price_change_ratio > self.price_change_threshold:
            print(f"  警告: 价格跳空过大 - 变化率: {price_change_ratio:.2%}")
            return False
        
        # 检查成交量是否足够
        if 'volume' in old_row and old_row['volume'] < self.min_volume:
            print(f"  警告: 旧合约成交量过低 - {old_row['volume']}")
            return False
        
        if 'volume' in new_row and new_row['volume'] < self.min_volume:
            print(f"  警告: 新合约成交量过低 - {new_row['volume']}")
            return False
        
        return True


class DataQualityChecker:
    """数据质量检查器 - 考虑合约切换点"""
    
    def __init__(self, rollover_points: Optional[List[ContractRollover]] = None):
        self.issues = []
        self.rollover_points = rollover_points or []
        self.rollover_times = [rp.rollover_datetime for rp in self.rollover_points if rp.is_valid]
    
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
        self.rollover_times = [rp.rollover_datetime for rp in self.rollover_points if rp.is_valid]
    
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

class EnhancedFuturesDataProcessor:
    """增强版期货数据处理类 - 集成零序列处理"""
    
    def __init__(self, auto_clean: bool = True, zero_handling_strategy: str = "interpolate"):
        self.raw_data = None
        self.cleaned_data = None
        self.rollover_points = []
        self.continuous_data = None
        self.rollover_detector = RolloverDetector(min_volume=100)
        self.auto_clean = auto_clean
        self.zero_handling_strategy = zero_handling_strategy
        
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """灵活加载数据，尝试多种日期格式"""
        try:
            # 读取CSV文件
            self.raw_data = pd.read_csv(file_path)
            print(f"原始数据行数: {len(self.raw_data)}")
            
            # 检查数据列
            print(f"数据列: {list(self.raw_data.columns)}")
            
            # 尝试多种日期解析方法
            datetime_col = self.raw_data['datetime']
            print(f"datetime列前5个值: {datetime_col.head().tolist()}")
            
            # 方法1: 尝试自动推断
            parsed_dates = pd.to_datetime(datetime_col, errors='coerce')
            valid_count = parsed_dates.notna().sum()
            
            if valid_count == 0:
                # 方法2: 尝试常见的日期格式
                print("自动推断失败，尝试常见日期格式...")
                common_formats = [
                    '%Y-%m-%d %H:%M:%S',
                    '%Y/%m/%d %H:%M:%S', 
                    '%m/%d/%Y %H:%M:%S',
                    '%d/%m/%Y %H:%M:%S',
                    '%Y-%m-%d',
                    '%Y/%m/%d',
                    '%m/%d/%Y',
                    '%d/%m/%Y'
                ]
                
                for fmt in common_formats:
                    try:
                        temp_dates = pd.to_datetime(datetime_col, format=fmt, errors='coerce')
                        temp_valid = temp_dates.notna().sum()
                        if temp_valid > 0:
                            print(f"  格式 '{fmt}' 成功解析 {temp_valid} 个日期")
                            parsed_dates = temp_dates
                            valid_count = temp_valid
                            break
                    except:
                        continue
            
            # 应用解析的日期
            self.raw_data['datetime'] = parsed_dates
            
            # 移除无法解析日期的行
            original_count = len(self.raw_data)
            self.raw_data = self.raw_data.dropna(subset=['datetime'])
            filtered_count = len(self.raw_data)
            
            print(f"移除 {original_count - filtered_count} 个无法解析日期的行")
            print(f"剩余有效数据: {filtered_count} 行")
            
            if filtered_count == 0:
                print("错误: 所有日期都无法解析，请检查数据格式")
                return None
            
            # 确保数值列是数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'position']
            for col in numeric_columns:
                if col in self.raw_data.columns:
                    self.raw_data[col] = pd.to_numeric(self.raw_data[col], errors='coerce')
            
            # 按时间排序
            self.raw_data = self.raw_data.sort_values('datetime').reset_index(drop=True)
            
            # 数据基本信息
            print(f"数据加载成功:")
            print(f"- 时间范围: {self.raw_data['datetime'].min()} 到 {self.raw_data['datetime'].max()}")
            print(f"- 合约数量: {self.raw_data['symbol'].nunique()}")
            print(f"- 总数据量: {len(self.raw_data)} 行")
            
            # 显示合约列表
            symbols = self.raw_data['symbol'].unique()
            print(f"- 包含合约: {list(symbols)}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None
    
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
        self.rollover_points = self.rollover_detector.detect_rollover_points(self.raw_data)
        valid_rollovers = [rp for rp in self.rollover_points if rp.is_valid]
        
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
            self.cleaned_data = data_cleaner.handle_volume_anomalies(self.cleaned_data)
            self.cleaned_data = data_cleaner.handle_price_anomalies(self.cleaned_data)
            
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
    
    def create_continuous_contract(self, method: str = "forward") -> pd.DataFrame:
        """创建连续合约"""
        data_to_use = self.cleaned_data if self.cleaned_data is not None else self.raw_data
        
        if data_to_use is None:
            raise ValueError("请先处理数据")
            
        # 只使用有效的切换点
        valid_rollovers = [rp for rp in self.rollover_points if rp.is_valid]
        
        if not valid_rollovers:
            print("没有有效的合约切换点，无法创建连续合约")
            return data_to_use

        continuous_data = data_to_use.copy()
        sorted_rollovers = sorted(valid_rollovers, key=lambda x: x.rollover_datetime)

        if method == "forward":
            # 前复权：当前价格不变，调整历史价格
            continuous_data['cumulative_factor'] = 1.0
            
            # 按时间倒序处理切换点
            for rollover in reversed(sorted_rollovers):
                adjustment_factor = rollover.new_open / rollover.old_close
                
                # 调整切换点之前的所有历史数据
                mask = continuous_data['datetime'] < rollover.rollover_datetime
                continuous_data.loc[mask, 'cumulative_factor'] *= adjustment_factor
                
        elif method == "backward":
            # 后复权：历史价格不变，调整未来价格
            continuous_data['cumulative_factor'] = 1.0
            
            # 按时间正序处理切换点
            for rollover in sorted_rollovers:
                adjustment_factor = rollover.old_close / rollover.new_open
                
                # 调整切换点及之后的所有未来数据
                mask = continuous_data['datetime'] >= rollover.rollover_datetime
                continuous_data.loc[mask, 'cumulative_factor'] *= adjustment_factor

        # 应用价格调整
        continuous_data['open_adj'] = continuous_data['open'] * continuous_data['cumulative_factor']
        continuous_data['high_adj'] = continuous_data['high'] * continuous_data['cumulative_factor']
        continuous_data['low_adj'] = continuous_data['low'] * continuous_data['cumulative_factor']
        continuous_data['close_adj'] = continuous_data['close'] * continuous_data['cumulative_factor']

        # 调整成交量（反向）
        continuous_data['volume_adj'] = continuous_data['volume'] / continuous_data['cumulative_factor']
        continuous_data['amount_adj'] = continuous_data['amount'] / continuous_data['cumulative_factor']
        continuous_data['position_adj'] = continuous_data['position'] / continuous_data['cumulative_factor']

        self.continuous_data = continuous_data
        print(f"{'前复权' if method == 'forward' else '后复权'}连续合约创建完成")

        return continuous_data
    
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
            
            # 抽样以减少数据点
            step = max(len(plot_data) // max_points, 1)
            plot_sampled = plot_data.iloc[::(sample_step if sample_step else step)]

            # 创建图表
            fig, ax = plt.subplots(figsize=(12, 6))

            # 两条价格线
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
                    
                ax.axvline(rv, color='red', linestyle='--', alpha=0.5)
                # 获取当前y轴范围来放置文本
                ylim = ax.get_ylim()
                ax.text(
                    rv, ylim[1],
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

# 在main函数开始时调用这个调试函数
def main():
    """主函数示例"""
    # 先调试数据加载
    file_path = "data/FG.csv"

    print("=== 数据加载调试 ===")
    debug_data_loading(file_path)
    
    # 然后运行完整流程
    print("\n" + "="*50)
    print("完整处理流程")
    print("="*50)
    
    # 创建处理器
    processor = EnhancedFuturesDataProcessor(
        auto_clean=True, 
        zero_handling_strategy="interpolate"
    )
    
    # 加载数据
    data = processor.load_data_from_csv(file_path)
    
    # 完整处理流程
    cleaned_data = processor.process_data_with_zero_handling()
    
    # 创建连续合约
    continuous_data = processor.create_continuous_contract("forward")
    
    # 显示结果
    print("\n处理完成!")
    print(f"原始数据行数: {len(data)}")
    print(f"处理后数据行数: {len(cleaned_data)}")
    
    if 'was_zero_corrected' in cleaned_data.columns:
        corrected_count = cleaned_data['was_zero_corrected'].sum()
        print(f"修正的零数据点: {corrected_count}")

    # 显示前几行结果
    print("\n前复权结果预览:")
    cols_to_show = ['datetime', 'symbol', 'close', 'close_adj', 'cumulative_factor']
    print(continuous_data[cols_to_show].head(10))
    
    print("\n=== 调整后价格范围统计 ===")
    for col in ['open_adj', 'high_adj', 'low_adj', 'close_adj']:
        max_val = continuous_data[col].max()
        min_val = continuous_data[col].min()
        print(f"{col}:")
        print(f"  最小值 = {min_val:.4f}")
        print(f"  最大值 = {max_val:.4f}")
    print(f"累计调整因子范围: {continuous_data['cumulative_factor'].min():.6f} ~ {continuous_data['cumulative_factor'].max():.6f}")

    # 绘制价格对比图
    processor.plot_price_comparison(max_points=5000)

if __name__ == "__main__":
    main()
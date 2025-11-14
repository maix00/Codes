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
        
        # 检查时间连续性
        if 'datetime' in data.columns:
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
        # 这是一个简化的实现，实际应用中需要根据具体品种的交易时间调整
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
            return {"status": "excellent", "issues_count": 0}
        
        issue_types = {}
        severity_counts = {"low": 0, "medium": 0, "high": 0}
        
        for issue in self.issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1
            severity_counts[issue.severity] += 1
        
        total_issues = len(self.issues)
        overall_status = "good" if severity_counts["high"] == 0 else "poor"
        
        return {
            "status": overall_status,
            "total_issues": total_issues,
            "issue_types": issue_types,
            "severity_breakdown": severity_counts,
            "issues": self.issues
        }


class DataCleaner:
    """数据清洗器 - 考虑合约切换点"""
    
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
        """处理缺失值"""
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
                symbol_data[numeric_columns] = symbol_data[numeric_columns].interpolate(method='linear')
                symbol_data[numeric_columns] = symbol_data[numeric_columns].fillna(method='ffill')
                symbol_data[numeric_columns] = symbol_data[numeric_columns].fillna(method='bfill')
                grouped_data.append(symbol_data)
            
            cleaned_data = pd.concat(grouped_data).sort_values('datetime').reset_index(drop=True)
        
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


class CorrectFuturesDataProcessor:
    """正确顺序的期货数据处理类"""
    
    def __init__(self, auto_clean: bool = True):
        self.raw_data = None
        self.cleaned_data = None
        self.rollover_points = []
        self.continuous_data = None
        self.rollover_detector = RolloverDetector(min_volume=100)
        self.auto_clean = auto_clean
        
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """从CSV文件加载数据"""
        try:
            # 读取CSV文件
            self.raw_data = pd.read_csv(
                file_path,
                parse_dates=['datetime'],
                dayfirst=True
            )
            
            # 按时间排序
            self.raw_data = self.raw_data.sort_values('datetime').reset_index(drop=True)
            
            # 数据基本信息
            print(f"数据加载成功:")
            print(f"- 时间范围: {self.raw_data['datetime'].min()} 到 {self.raw_data['datetime'].max()}")
            print(f"- 合约数量: {self.raw_data['symbol'].nunique()}")
            print(f"- 总数据量: {len(self.raw_data)} 行")
            print(f"- 包含合约: {list(self.raw_data['symbol'].unique())}")
            
            return self.raw_data
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            raise
    
    def process_data(self) -> pd.DataFrame:
        """完整的数据处理流程"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        print("\n" + "="*50)
        print("步骤1: 检测合约切换点")
        print("="*50)
        
        # 第一步：检测合约切换点
        self.rollover_points = self.rollover_detector.detect_rollover_points(self.raw_data)
        valid_rollovers = [rp for rp in self.rollover_points if rp.is_valid]
        
        print(f"\n有效合约切换点: {len(valid_rollovers)} 个")
        
        print("\n" + "="*50)
        print("步骤2: 数据质量检查")
        print("="*50)
        
        # 第二步：数据质量检查（考虑合约切换点）
        quality_checker = DataQualityChecker(valid_rollovers)
        quality_checker.check_missing_values(self.raw_data)
        quality_checker.check_price_anomalies(self.raw_data)
        quality_checker.check_volume_anomalies(self.raw_data)
        
        quality_report = quality_checker.generate_quality_report()
        self._print_quality_report(quality_report)
        
        # 第三步：数据清洗（考虑合约切换点）
        if self.auto_clean and quality_report["status"] != "excellent":
            print("\n" + "="*50)
            print("步骤3: 数据清洗")
            print("="*50)
            
            data_cleaner = DataCleaner(valid_rollovers)
            self.cleaned_data = data_cleaner.handle_missing_values(self.raw_data)
            self.cleaned_data = data_cleaner.handle_volume_anomalies(self.cleaned_data)
            self.cleaned_data = data_cleaner.handle_price_anomalies(self.cleaned_data)
            
            cleaning_summary = data_cleaner.get_cleaning_summary()
            print(f"数据清洗完成，执行了 {cleaning_summary['total_actions']} 个清洗操作")
        else:
            self.cleaned_data = self.raw_data.copy()
            print("跳过数据清洗（数据质量良好或auto_clean=False）")
        
        return self.cleaned_data
    
    def _print_quality_report(self, report: Dict[str, Any]):
        """打印质量报告"""
        print(f"数据质量状态: {report['status']}")
        print(f"总问题数: {report['total_issues']}")
        
        if report['total_issues'] > 0:
            print("问题类型分布:")
            for issue_type, count in report['issue_types'].items():
                print(f"  - {issue_type}: {count}")
            
            print("严重程度分布:")
            for severity, count in report['severity_breakdown'].items():
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
        """绘制原始价格和复权后价格的对比图"""
        if self.continuous_data is None:
            self.create_continuous_contract("forward")
        
        if self.continuous_data is None:
            raise ValueError("连续合约数据未创建，无法绘图")
        
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        # --- Set Chinese font support globally ---
        plt.rcParams['font.sans-serif'] = ['Heiti TC', 'STHeiti', 'PingFang SC', 'Microsoft YaHei', 'Arial Unicode MS']
        plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

                # 筛选时间范围
        plot_data = self.continuous_data.copy()
        if start_date:
            plot_data = plot_data[plot_data['datetime'] >= start_date]
        if end_date:
            plot_data = plot_data[plot_data['datetime'] <= end_date]
        
        # ↓ choose one simplification ↓
        # if sample_step:
        #     plot_data = plot_data.iloc[::sample_step]
        # elif len(plot_data) > max_points:
        #     step = len(plot_data) // max_points
        #     plot_data = plot_data.iloc[::step]
        
        step = max(len(plot_data) // max_points, 1)
        plot_sampled = plot_data.iloc[::(sample_step if sample_step else step)]

        # Convert strings to matplotlib date numbers (vectorized)
        x = mdates.datestr2num(plot_sampled['datetime'].astype(str).values)

        fig, ax = plt.subplots(figsize=(12, 6))

        # Two price series
        ax.plot(x, plot_sampled['close'], label='原始收盘价', color='blue', alpha=0.7)
        ax.plot(x, plot_sampled['close_adj'], label='前复权收盘价', color='green', alpha=0.7)

        # Mark rollover lines
        for rollover in self.rollover_points:
            rv = rollover.rollover_datetime

            # Convert rollover time to matplotlib float date (not array)
            if isinstance(rv, str):
                try:
                    rv_num = float(mdates.datestr2num(rv))  # explicitly cast single float
                except Exception:
                    continue
            else:
                rv_num = float(mdates.date2num(rv))

            # Range check (optional)
            if (start_date is not None and rv_num < mdates.datestr2num(str(start_date))) or \
            (end_date is not None and rv_num > mdates.datestr2num(str(end_date))):
                continue

            ax.axvline(rv_num, color='red', linestyle='--', alpha=0.5)
            ax.text(
                rv_num, ax.get_ylim()[1],
                f'{rollover.old_contract}→{rollover.new_contract}',
                rotation=90, va='top', ha='right', fontsize=8, color='red'
            )

        # Format x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()

        ax.set_title('原始与前复权连续合约价格')
        ax.set_xlabel('时间')
        ax.set_ylabel('价格')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.show()

def main():
    """正确的数据处理流程示例"""
    # 创建处理器
    processor = CorrectFuturesDataProcessor(auto_clean=True)
    
    # 加载数据
    data = processor.load_data_from_csv("data/SI.csv")
    
    # 完整处理流程
    cleaned_data = processor.process_data()
    
    # 创建连续合约
    continuous_data = processor.create_continuous_contract("forward")
    
    # 显示结果
    print("\n" + "="*50)
    print("处理结果")
    print("="*50)
    cols_to_show = ['datetime', 'symbol', 'close', 'close_adj', 'cumulative_factor']
    print(continuous_data[cols_to_show].head(10))
    
    print("\n=== 调整后价格范围统计 ===")
    for col in ['open_adj', 'high_adj', 'low_adj', 'close_adj']:
        if col in continuous_data.columns:
            max_val = continuous_data[col].max()
            min_val = continuous_data[col].min()
            print(f"{col}:")
            print(f"  最小值 = {min_val:.4f}")
            print(f"  最大值 = {max_val:.4f}")
    
    if 'cumulative_factor' in continuous_data.columns:
        print(f"累计调整因子范围: {continuous_data['cumulative_factor'].min():.6f} ~ {continuous_data['cumulative_factor'].max():.6f}")

    # 绘制价格对比图
    processor.plot_price_comparison(max_points=5000)
    
if __name__ == "__main__":
    main()
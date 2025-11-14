import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class ContractRollover:
    """合约切换事件的数据结构"""
    rollover_datetime: datetime  # 切换的具体时间点
    old_contract: str           # 旧合约代码
    new_contract: str           # 新合约代码
    old_close: float           # 旧合约最后的价格
    new_open: float            # 新合约开始的价格
    price_gap: float           # 价格差距
    adjustment_factor: float   # 调整因子
    
    def __post_init__(self):
        """计算价格差距和调整因子"""
        self.price_gap = self.new_open - self.old_close
        self.adjustment_factor = self.new_open / self.old_close if self.old_close != 0 else 1.0

class FuturesDataProcessor:
    """期货数据处理类 - 简化版本"""
    
    def __init__(self):
        self.raw_data = None
        self.rollover_points = []
        self.continuous_data = None
        
    def load_data_from_csv(self, file_path: str) -> pd.DataFrame:
        """从CSV文件加载数据"""
        try:
            # 读取CSV文件
            self.raw_data = pd.read_csv(
                file_path,
                parse_dates=['datetime'],
                dayfirst=True  # 处理日/月/年格式
            )
            
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
            raise
    
    def clean_data(self, method='interpolate', price_threshold=0.1, volume_threshold=5.0):
        """
        清洗数据：处理缺失值和异常值
        
        参数:
        method: 缺失值处理方法，可选 'interpolate'（插值）或 'fill'（向前填充）
        price_threshold: 价格异常阈值，超过这个比例的变化被视为异常
        volume_threshold: 成交量异常阈值，超过这个比例的倍数被视为异常
        """
        if self.raw_data is None:
            raise ValueError("请先加载数据")
        
        data = self.raw_data.copy()
        
        # 1. 处理缺失值
        # 检查是否有缺失的时间点（如果数据是固定频率的）
        # 这里假设数据已经是按时间排序的
        data = self._handle_missing_values(data, method)
        
        # 2. 处理异常值
        data = self._handle_outliers(data, price_threshold, volume_threshold)
        
        self.raw_data = data
        print("数据清洗完成")
        
        return data

    def _handle_missing_values(self, data, method):
        """处理缺失值"""
        # 检查缺失值
        print("检查缺失值...")
        missing_count = data.isnull().sum()
        if missing_count.any():
            print(f"发现缺失值: {missing_count}")
            
            # 对于时间序列，我们通常用插值或填充
            numeric_columns = data.select_dtypes(include=[np.number]).columns.tolist()
            
            if method == 'interpolate':
                data[numeric_columns] = data[numeric_columns].interpolate(method='linear')
                # 如果开头还有缺失，用向后填充
                data[numeric_columns] = data[numeric_columns].fillna(method='bfill')
            elif method == 'fill':
                data[numeric_columns] = data[numeric_columns].fillna(method='ffill').fillna(method='bfill')
        else:
            print("未发现缺失值")
            
        return data

    def _handle_outliers(self, data, price_threshold, volume_threshold):
        """处理异常值"""
        print("检查异常值...")
        
        # 价格异常：检查相邻两个时间点的价格变化率是否超过阈值
        price_columns = ['open', 'high', 'low', 'close']
        
        for col in price_columns:
            # 计算价格变化率
            price_change = data[col].pct_change().abs()
            # 标记异常值（变化率超过阈值）
            outlier_mask = price_change > price_threshold
            
            # 排除合约切换点附近的变化（因为换合约可能导致价格跳空）
            # 我们将在后面处理合约切换点，所以这里先不处理切换点附近的异常
            # 但注意：如果切换点已经被标记，我们需要排除这些点
            if len(self.rollover_points) > 0:
                for rollover in self.rollover_points:
                    # 将切换点附近的数据排除在异常检测之外
                    # 这里我们排除切换点前后各1个数据点
                    rollover_time = rollover.rollover_datetime
                    idx = data[data['datetime'] == rollover_time].index
                    if len(idx) > 0:
                        idx = idx[0]
                        # 将切换点前后各1个数据点标记为非异常
                        outlier_mask.loc[max(0, idx-1):min(len(data)-1, idx+1)] = False
            
            outlier_count = outlier_mask.sum()
            if outlier_count > 0:
                print(f"在{col}中发现{outlier_count}个价格异常点")
                # 将异常值替换为前一个有效值
                data.loc[outlier_mask, col] = np.nan
                data[col] = data[col].fillna(method='ffill')
            else:
                print(f"在{col}中未发现异常值")
        
        # 成交量异常：检查成交量是否为0或负数，或者异常大
        volume_col = 'volume'
        if volume_col in data.columns:
            # 计算成交量的Z-score（标准化得分）
            volume_mean = data[volume_col].mean()
            volume_std = data[volume_col].std()
            # 如果标准差为0，则所有值相同，无需处理
            if volume_std > 0:
                volume_zscore = (data[volume_col] - volume_mean) / volume_std
                # 标记Z-score绝对值超过volume_threshold的为异常
                outlier_mask = volume_zscore.abs() > volume_threshold
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"在{volume_col}中发现{outlier_count}个成交量异常点")
                    # 将异常值替换为前一个有效值
                    data.loc[outlier_mask, volume_col] = np.nan
                    data[volume_col] = data[volume_col].fillna(method='ffill')
                else:
                    print(f"在{volume_col}中未发现异常值")
        
        # 持仓量异常：类似成交量
        position_col = 'position'
        if position_col in data.columns:
            position_mean = data[position_col].mean()
            position_std = data[position_col].std()
            if position_std > 0:
                position_zscore = (data[position_col] - position_mean) / position_std
                outlier_mask = position_zscore.abs() > volume_threshold
                outlier_count = outlier_mask.sum()
                if outlier_count > 0:
                    print(f"在{position_col}中发现{outlier_count}个持仓量异常点")
                    data.loc[outlier_mask, position_col] = np.nan
                    data[position_col] = data[position_col].fillna(method='ffill')
                else:
                    print(f"在{position_col}中未发现异常值")
                    
        return data

    def detect_rollover_points_simple(self) -> List[ContractRollover]:
        """简化版合约切换点检测 - 直接检测symbol变化"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
            
        # 检测symbol变化点
        self.raw_data['symbol_change'] = self.raw_data['symbol'] != self.raw_data['symbol'].shift(1)
        
        # 找出所有切换点
        change_indices = self.raw_data[self.raw_data['symbol_change']].index.tolist()
        
        # 移除第一个点（可能是数据开始）
        if change_indices and change_indices[0] == 0:
            change_indices = change_indices[1:]
        
        print(f"检测到 {len(change_indices)} 个合约切换点")
        
        # 构建切换事件
        self.rollover_points = []
        for idx in change_indices:
            try:
                # 获取切换前后的数据
                old_row = self.raw_data.iloc[idx - 1]  # 旧合约最后一条数据
                new_row = self.raw_data.iloc[idx]      # 新合约第一条数据
                
                # 创建切换事件
                rollover = ContractRollover(
                    rollover_datetime=new_row['datetime'],
                    old_contract=old_row['symbol'],
                    new_contract=new_row['symbol'],
                    old_close=old_row['close'],
                    new_open=new_row['open'],
                    price_gap=0,  # 会在__post_init__中计算
                    adjustment_factor=1.0
                )
                
                self.rollover_points.append(rollover)
                
                print(f"切换点 {rollover.rollover_datetime}: {rollover.old_contract} -> {rollover.new_contract}, "
                      f"价差: {rollover.price_gap:.2f}, 调整因子: {rollover.adjustment_factor:.6f}")
                      
            except Exception as e:
                print(f"处理切换点 {idx} 时出错: {e}")
                continue
        
        return self.rollover_points
    
    def create_continuous_contract_forward(self, method: str = "forward") -> pd.DataFrame:
        """修正后的复权方法"""
        if self.raw_data is None:
            raise ValueError("请先加载数据")
            
        if not self.rollover_points:
            self.detect_rollover_points_simple()

        continuous_data = self.raw_data.copy()
        sorted_rollovers = sorted(self.rollover_points, key=lambda x: x.rollover_datetime)

        if method == "forward":
            # 前复权：当前价格不变，调整历史价格
            # 我们需要从最新数据向最老数据反向处理
            continuous_data['cumulative_factor'] = 1.0
            
            # 按时间倒序处理切换点
            for rollover in reversed(sorted_rollovers):
                # 调整因子应该是新合约价格/旧合约价格
                # 但为了保持当前价格不变，我们需要反向调整历史价格
                adjustment_factor = rollover.new_open / rollover.old_close
                
                # 调整切换点之前的所有历史数据
                mask = continuous_data['datetime'] < rollover.rollover_datetime
                continuous_data.loc[mask, 'cumulative_factor'] *= adjustment_factor
                
        elif method == "backward":
            # 后复权：历史价格不变，调整未来价格
            continuous_data['cumulative_factor'] = 1.0
            
            # 按时间正序处理切换点
            for rollover in sorted_rollovers:
                # 调整因子应该是旧合约价格/新合约价格
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
            self.create_continuous_contract_forward()
        
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
    """主函数示例"""
    # 创建处理器
    processor = FuturesDataProcessor()
    
    # 这里替换为您的实际CSV文件路径
    data = processor.load_data_from_csv("data/SI.csv")
    
    # 创建示例数据来演示
    # print("创建示例数据...")
    # example_data = create_sample_intraday_data()
    # processor.raw_data = example_data
    processor.raw_data = data
    
    # 检测切换点
    rollovers = processor.detect_rollover_points_simple()

     # 新增数据清洗步骤
    processor.clean_data(method='interpolate', price_threshold=0.1, volume_threshold=5.0)
    
    print(f"\n共检测到 {len(rollovers)} 个合约切换事件")
    
    # 创建连续合约
    continuous_data = processor.create_continuous_contract_forward()
    
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

def create_sample_intraday_data() -> pd.DataFrame:
    """创建日内示例数据"""
    # 创建时间序列（日内数据，每小时间隔）
    dates = pd.date_range('2012-12-03 09:00:00', '2012-12-10 15:00:00', freq='1H')
    # 只保留交易时间（假设9:00-15:00）
    dates = dates[((dates.hour >= 9) & (dates.hour <= 15))]
    
    data = []
    current_contract = 'FG1301'
    
    for i, dt in enumerate(dates):
        # 模拟合约切换：在特定时间点切换
        if i == 20:  # 切换到FG1305
            current_contract = 'FG1305'
        elif i == 40:  # 切换到FG1309
            current_contract = 'FG1309'
        
        # 基础价格 - 不同合约有不同基础价格
        if current_contract == 'FG1301':
            base_price = 1350
        elif current_contract == 'FG1305':
            base_price = 1370  # 新合约价格更高
        else:  # FG1309
            base_price = 1390  # 更远月合约价格更高
            
        # 添加一些随机波动
        price_move = np.random.normal(0, 5)
        
        data.append({
            'datetime': dt,
            'open': base_price + price_move - 2,
            'high': base_price + price_move + 5,
            'low': base_price + price_move - 5,
            'close': base_price + price_move,
            'volume': np.random.randint(1000, 5000),
            'amount': np.random.randint(1000000, 5000000),
            'position': np.random.randint(5000, 10000),
            'symbol': current_contract
        })
    
    return pd.DataFrame(data)

if __name__ == "__main__":
    main()
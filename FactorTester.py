from enum import Enum
import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
import os
from BackTester import Futures, FuturesContract, ProductBase
from collections import Counter

class FactorTester:
    def __init__(self, file_paths: List[str], start_date: Optional[str] = None, end_date: Optional[str] = None,
                 futures_flag: bool = True, futures_adjust_col: Optional[List[str]] = None):
        """
        file_paths: list of parquet file paths, each containing a time series for a future contract.
        """
        self.data = {}  # key: contract name, value: DataFrame
        for path in file_paths:
            df = pd.read_parquet(path)
            if df.empty:
                continue
            # Assume each file has a 'trade_time' column and is sorted
            contract = path.split('/')[-1].replace('.parquet', '')
            self.data[contract] = df.set_index('trade_time')
        self.contracts = list(self.data.keys())
        self.start_date = start_date
        self.end_date = end_date
        self.futures_flag = futures_flag
        self.futures_adjust_col = futures_adjust_col
        if self.futures_flag:
            self.futures_adjust_col = futures_adjust_col if futures_adjust_col else ['close_price']
            self.futures_adjust_col_adjusted = [col + '_adjusted' for col in self.futures_adjust_col]
            for c, df in self.data.items():
                if any(col not in df.columns for col in self.futures_adjust_col_adjusted):
                    assert 'adjustment_mul' in df.columns
                    assert 'adjustment_add' in df.columns
                    for col, col_adj in zip(self.futures_adjust_col, self.futures_adjust_col_adjusted):
                        df[col_adj] = df[col] * df['adjustment_mul'] + df['adjustment_add']
        self.factor_frequency: Optional[pd.Timedelta] = None

    def calc_returns(self, interval: int = 1, price_col: str = 'close_price_adjusted') -> pd.DataFrame:
        """
        Calculate returns for each contract.
        interval: time interval for returns (in rows, e.g., 1 for next row)
        price_col: column name for price
        Returns: DataFrame, index: datetime, columns: contracts
        """
        returns = {}
        for c, df in self.data.items():
            assert price_col in df.columns, f"{price_col} not in DataFrame columns for contract {c}"
            returns[c] = df[price_col].pct_change(periods=interval)
        return pd.DataFrame(returns)
    
    def calc_daily_return(self, price_col: str = 'open_price') -> pd.DataFrame:
        """
        计算每个合约的每日收益率。
        price_col: 价格列名
        Returns: DataFrame，索引为datetime，列为合约名称
        """
        daily_returns = {}
        for c, df in self.data.items():
            daily_returns[c] = daily_return(df, price_col=price_col)
        return pd.DataFrame(daily_returns)

    def calc_factor(self, factor_func: Callable[[pd.DataFrame], pd.Series], 
                    set_freq: bool = True, frequency: Optional[pd.Timedelta] = None) -> pd.DataFrame:
        """
        将因子函数应用于每个合约的DataFrame。
        
        参数:
            factor_func: 接收DataFrame并返回Series（以datetime为索引）的函数
            frequency: 因子频率，FactorFrequency枚举值
        
        返回:
            DataFrame: 索引为datetime，列为合约名称
        """

        # 计算因子
        factors = {}
        for c, df in self.data.items():
            factors[c] = factor_func(df)

        if set_freq:
            # 如果因子频率未指定，则尝试从数据中获取
            if factors and frequency is None:
                all_freqs = []
                for freq_series in factors.values():
                    if len(freq_series) > 0:
                        if len(freq_series.index) > 1:
                            time_diffs = pd.to_datetime(freq_series.index).copy().to_series().diff().dropna()
                            if len(time_diffs) > 0:
                                min_diff = time_diffs.min()
                                all_freqs.append(min_diff)
                if all_freqs:
                    freq_counter = Counter(all_freqs)
                    most_common_freq = freq_counter.most_common(1)[0][0]
                    self.factor_frequency = most_common_freq
                else:
                    self.factor_frequency = None
            else:
                self.factor_frequency = frequency
        
        return pd.DataFrame(factors)

    def calc_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional rank for each datetime.
        """
        return df.rank(axis=1, method='average', na_option='keep', pct=True)

    def calc_ic(self, factor_df: pd.DataFrame, return_df: pd.DataFrame, 
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> tuple[pd.Series, pd.DataFrame]:
        """
        Calculate IC (Spearman correlation) between factor and future returns for each datetime.
        All calculations stop before end_date (inclusive).
        Also return IC stats and average contract coverage (mean non-NaN count per datetime).
        end_date: str or None, e.g. '2023-12-31'. If provided, only datetimes <= end_date are used.
        """
        factor_rank = self.calc_rank(factor_df)
        return_rank = self.calc_rank(return_df)
        dt_index = factor_rank.index.intersection(return_rank.index)
        start_date = start_date if start_date is not None else self.start_date
        end_date = end_date if end_date is not None else self.end_date
        if start_date is not None:
            dt_index = dt_index[dt_index >= start_date]
        if end_date is not None:
            dt_index = dt_index[dt_index <= end_date]
        ic = []
        coverage = []
        for dt in dt_index:
            f = factor_rank.loc[dt]
            r = return_rank.loc[dt]
            valid = f.notna() & r.notna()
            coverage.append(valid.sum())
            if valid.sum() > 1:
                ic.append(pd.Series(f[valid]).corr(r[valid], method='spearman'))
            else:
                ic.append(np.nan)
        ic_series = pd.Series(ic, index=dt_index)
        avg_coverage = np.mean(coverage)
        stats_df = self.ic_stats(ic_series)
        stats_df['avg_coverage'] = avg_coverage
        return ic_series, stats_df

    def ic_stats(self, ic_series: pd.Series) -> pd.DataFrame:
        """
        Calculate mean, std, IR of IC series.
        """
        mean = ic_series.mean()
        std = ic_series.std()
        ir = mean / std if std != 0 else np.nan
        t_stat = mean / (std / np.sqrt(len(ic_series.dropna()))) if std != 0 and len(ic_series.dropna()) > 1 else np.nan
        max_ic = ic_series.max()
        min_ic = ic_series.min()
        # Return as DataFrame
        stats_df = pd.DataFrame({
            'mean': [mean],
            'std': [std],
            'IR': [ir],
            't_stat': [t_stat],
            'max': [max_ic],
            'min': [min_ic]
        })
        return stats_df

    def group_classes(self, factor_df: pd.DataFrame, n_groups: int = 5, plot_flag: bool = False, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      plot_n_group_list: Optional[List[int]] = None) -> Tuple[Dict[str, Dict[str, List[ProductBase]]], Dict[str, Dict[str, float]]]:
        """
        For each datetime, split contracts into n_groups groups.
        Each group is a dict: {datetime_str: [contract names]}.
        n_groups: number of groups to split into (default: 5)
        """
        # Calculate daily returns at next open
        assert 'open_price_adjusted' in next(iter(self.data.values())).columns, "DataFrames must contain 'open_price_adjusted' column."
        open_returns = self.calc_daily_return(price_col='open_price_adjusted')

        # Plot adjustment for group numbers
        plot_n_group_list = [n_groups + n_group if n_group < 0 else n_group for n_group in plot_n_group_list] if plot_n_group_list else None

        group_names = ['group_' + str(i) for i in range(n_groups)]
        group_names = group_names[::-1]
        groups = {name: {} for name in group_names}
        open_returns_groups = {name: {} for name in group_names}
        
        for dt, row in factor_df.iterrows():
            dt_str = str(dt)
            sorted_contracts = row.dropna().sort_values(ascending=False)
            n = len(sorted_contracts)
            idx = list(sorted_contracts.index)
            idx = [ProductBase(name) for name in idx]
            if self.futures_flag:
                idx = [Futures(product.name) for product in idx]
            
            # Split contracts into n_groups groups
            if n > 0:
                idx_array = np.asarray(idx)
                split = np.array_split(idx_array, min(n, n_groups))
                
                # Fill groups from bottom to top (ascending order of factor values)
                for i, group_contracts in enumerate(split):
                    group_idx = n_groups - 1 - i  # Reverse order: bottom group first
                    groups[group_names[group_idx]][dt_str] = list(group_contracts)
                    contract_names = [product.name for product in group_contracts]
                    open_returns_groups[group_names[group_idx]][dt_str] = np.mean(np.asarray(open_returns.loc[dt_str][contract_names].values, dtype=float))
            
            # Fill remaining groups (if n < n_groups) with empty lists
            for i in range(min(n, n_groups), n_groups):
                groups[group_names[i]][dt_str] = []
                open_returns_groups[group_names[i]][dt_str] = np.nan

        report_groups = {}
        for name in group_names:
            dates = list(open_returns_groups[name].keys()) if open_returns_groups[name] else []
            dates = [date for date in dates if start_date <= date] if start_date else dates
            dates = [date for date in dates if date <= end_date] if end_date else dates
            returns = [open_returns_groups[name][date] for date in dates]
            returns_series = pd.Series(returns).dropna()
            cumulative_returns = (1 + returns_series).cumprod()
            metrics = {
            'Total Return': (cumulative_returns.iloc[-1] - 1) * 100 if len(cumulative_returns) > 0 else 0,
            'Annual Return': ((cumulative_returns.iloc[-1]) ** (252 / len(cumulative_returns)) - 1) * 100 if len(cumulative_returns) > 1 else 0,
            'Volatility': pd.Series(returns).std() * np.sqrt(252) * 100,
            'Sharpe Ratio': (pd.Series(returns).mean() * 252) / (pd.Series(returns).std() * np.sqrt(252)) if pd.Series(returns).std() != 0 else 0,
            'Max Drawdown': ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max() * 100 if len(cumulative_returns) > 0 else 0,
            'Calmar Ratio': ((cumulative_returns.iloc[-1] ** (252 / len(cumulative_returns)) - 1) * 100) / (((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max() * 100) if ((cumulative_returns.cummax() - cumulative_returns) / cumulative_returns.cummax()).max() != 0 else 0,
            'Win Rate': (pd.Series(returns) > 0).sum() / len(pd.Series(returns)) * 100 if len(pd.Series(returns)) > 0 else 0,
            'Mean Return': pd.Series(returns).mean() * 100,
            'Skewness': pd.Series(returns).skew(),
            'Kurtosis': pd.Series(returns).kurtosis(),
            }
            name = int(name.split('_')[-1])
            report_groups[name] = pd.Series(metrics)
        
        report_df = pd.DataFrame(report_groups).T.sort_index()  # Convert to DataFrame, transpose, and sort by name
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("Group Performance Summary:\n", report_df)

        if plot_flag:
            import matplotlib.pyplot as plt
            # Plot average open returns per group over time
            plt.figure(figsize=(12, 6))
            start_date = start_date if start_date is not None else self.start_date
            end_date = end_date if end_date is not None else self.end_date
            print(start_date, end_date)
            dates = []  # Initialize dates as an empty list
            for name in group_names:
                if plot_n_group_list is not None and name.split('_')[-1] not in [str(n) for n in plot_n_group_list]:
                    continue
                dates = list(open_returns_groups[name].keys()) if open_returns_groups[name] else []
                dates = [date for date in dates if start_date <= date] if start_date else dates
                dates = [date for date in dates if date <= end_date] if end_date else dates
                returns = [open_returns_groups[name][date] for date in dates]
                cumulative_returns = []
                prev_value = 10000
                for ret in returns:
                    if not np.isnan(ret):
                        prev_value = prev_value * (1 + ret)
                    cumulative_returns.append(prev_value)
                plt.plot(dates, cumulative_returns, label=name)
            plt.xlabel('Date')
            plt.ylabel('Average Next Day Open Return')
            plt.title('Average Next Day Open Return by Factor Groups')
            plt.legend()
            # Only show every nth tick to reduce crowding
            n_ticks = 10
            assert len(dates) > 0, "No dates available for plotting."
            tick_indices = np.linspace(0, len(dates) - 1, min(n_ticks, len(dates)), dtype=int)
            plt.xticks(ticks=[dates[i] for i in tick_indices], rotation=45)
            plt.tight_layout()
            plt.show()

        return groups, open_returns_groups
    
# Example usage:
if __name__ == '__main__':
    parquet_dir = '../data/main_mink/'
    file_list = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith('.parquet') and '_S' not in f and '-S' not in f
    ]
    tester = FactorTester(file_list, end_date='2025-05-31', futures_flag=True, futures_adjust_col=['close_price'])
    returns = tester.calc_returns(interval=1)
    factor = tester.calc_factor(lambda df: df['close_price_adjusted'].rolling(5).mean())
    print(factor)
    ic_series, stats = tester.calc_ic(factor, returns)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    #     print(ic_series)
    print('rolloing_close\n', stats)
    # groups = tester.group_five_classes(factor)
    # print(groups['top'][:5])  # print top group for first 5 datetimes

#  计算每日收益率（按trading_day分组，收盘价对数收益）
def log_daily_return(df, price_col='close_price'):
    if 'trading_day' not in df.columns:
        raise ValueError("DataFrame must contain 'trading_day' column for daily grouping.")
    daily_close = df.groupby('trading_day')[price_col].last()
    return np.log(daily_close).diff()

def daily_return(df, price_col: str = 'close_price'):
    if 'trading_day' not in df.columns:
        raise ValueError("DataFrame must contain 'trading_day' column for daily grouping.")
    if price_col.startswith('close_price'):
        daily_close = df.groupby('trading_day')[price_col].last()
        # Calculate daily return as today's close price change
        return daily_close.pct_change()
    elif price_col.startswith('open_price'):
        daily_open = df.groupby('trading_day')[price_col].first()
        # Calculate daily return as next day's open price change
        return daily_open.pct_change().shift(-1)
    else:
        raise ValueError("Invalid price_col argument. Must be 'close_price' or 'open_price'.")

def integrated_ic_test_daily(factor_func: Callable, n_groups: int = 5, plot_n_group_list: Optional[List[int]] = None,):
    parquet_dir = '../data/main_mink/'
    file_list = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith('.parquet') and '_S' not in f and '-S' not in f
    ]
    tester = FactorTester(file_list, #start_date='2025-01-01', 
                      end_date='2025-05-30', 
                      futures_flag=True, futures_adjust_col=['close_price', 'open_price'])

    # Calculate inflection point factor for all contracts
    factor_series = tester.calc_factor(lambda df: factor_func(df, price_col='close_price_adjusted'))
    daily_returns = tester.calc_factor(lambda df: log_daily_return(df, price_col='open_price_adjusted'), set_freq=False)

    ic_series, stats = tester.calc_ic(factor_series, daily_returns)
    print(factor_func.__name__, 'IC Stats:\n', stats.T)
    groups, open_returns_groups = tester.group_classes(factor_series, 
                                                       plot_flag=True, 
                                                       n_groups=n_groups, 
                                                       plot_n_group_list=plot_n_group_list,
                                                       end_date='2025-12-31', 
                                                       start_date='2025-01-01')
    # # Get the earliest five dates from the 'top' group
    # earliest_dates = sorted(groups['group_0'].keys())[:5]
    # for date in earliest_dates:
    #     print(date, groups['group_0'][date])
    #     print(date, open_returns_groups['group_0'][date])
    #     pass
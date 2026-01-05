import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional
import os
from BackTester import Futures, FuturesContract, ProductBase

mappings_path = '../data/wind_mapping.parquet'

class ICTester:
    def __init__(self, file_paths: List[str], end_date: Optional[str] = None,
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

    def calc_factor(self, factor_func: Callable[[pd.DataFrame], pd.Series]) -> pd.DataFrame:
        """
        Apply a factor function to each contract's DataFrame.
        factor_func: function that takes a DataFrame and returns a Series (indexed by datetime)
        Returns: DataFrame, index: datetime, columns: contracts
        """
        factors = {}
        for c, df in self.data.items():
            factors[c] = factor_func(df)
        return pd.DataFrame(factors)

    def calc_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate cross-sectional rank for each datetime.
        """
        return df.rank(axis=1, method='average', na_option='keep', pct=True)

    def calc_ic(self, factor_df: pd.DataFrame, return_df: pd.DataFrame, end_date: Optional[str] = None) -> tuple[pd.Series, pd.DataFrame]:
        """
        Calculate IC (Spearman correlation) between factor and future returns for each datetime.
        All calculations stop before end_date (inclusive).
        Also return IC stats and average contract coverage (mean non-NaN count per datetime).
        end_date: str or None, e.g. '2023-12-31'. If provided, only datetimes <= end_date are used.
        """
        factor_rank = self.calc_rank(factor_df)
        return_rank = self.calc_rank(return_df)
        dt_index = factor_rank.index.intersection(return_rank.index)
        end_date = end_date if end_date is not None else self.end_date
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

    def group_five_classes(self, factor_df: pd.DataFrame) -> Dict[str, Dict[str, List[ProductBase]]]:
        """
        For each datetime, split contracts into five groups: 'top', '1', '2', '3', 'bottom'.
        Each group is a dict: {datetime_str: [contract names]}.
        """
        groups = {'top': {}, '1': {}, '2': {}, '3': {}, 'bottom': {}}
        for dt, row in factor_df.iterrows():
            dt_str = str(dt)
            sorted_contracts = row.dropna().sort_values(ascending=False)
            n = len(sorted_contracts)
            idx = list(sorted_contracts.index)
            idx = [ProductBase(name) for name in idx]
            if self.futures_flag:
                idx = [Futures(product.name, mappings_path=mappings_path) for product in idx]
            if n == 0:
                for k in groups:
                    groups[k][dt_str] = []
            elif n == 1:
                groups['top'][dt_str] = [idx[0]]
                groups['1'][dt_str] = []
                groups['2'][dt_str] = []
                groups['3'][dt_str] = []
                groups['bottom'][dt_str] = []
            elif n == 2:
                groups['top'][dt_str] = [idx[0]]
                groups['1'][dt_str] = []
                groups['2'][dt_str] = []
                groups['3'][dt_str] = []
                groups['bottom'][dt_str] = [idx[1]]
            elif n == 3:
                groups['top'][dt_str] = [idx[0]]
                groups['1'][dt_str] = [idx[1]]
                groups['2'][dt_str] = []
                groups['3'][dt_str] = []
                groups['bottom'][dt_str] = [idx[2]]
            elif n == 4:
                groups['top'][dt_str] = [idx[0]]
                groups['1'][dt_str] = [idx[1]]
                groups['2'][dt_str] = [idx[2]]
                groups['3'][dt_str] = []
                groups['bottom'][dt_str] = [idx[3]]
            else:
                # n >= 5, split as evenly as possible
                # Convert to numpy array to ensure compatibility with array_split
                idx_array = np.asarray(idx)
                split = np.array_split(idx_array, 5)
                groups['top'][dt_str] = list(split[0])
                groups['1'][dt_str] = list(split[1])
                groups['2'][dt_str] = list(split[2])
                groups['3'][dt_str] = list(split[3])
                groups['bottom'][dt_str] = list(split[4])
        return groups

# Example usage:
if __name__ == '__main__':
    parquet_dir = '../data/main_mink/'
    file_list = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith('.parquet') and '_S' not in f and '-S' not in f
    ]
    tester = ICTester(file_list, end_date='2025-05-31', futures_flag=True, futures_adjust_col=['close_price'])
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

# def intraday_momentum_reversal_factor(df: pd.DataFrame, price_col: str = 'close_price') -> pd.Series:
#     """
#     Calculate intraday momentum reversal factor for each trading day.
#     intraday_momentum = (extrema_first - extrema_behind) / extrema_first
#     Where extrema_first: first local extremum (min or max) in the day,
#           extrema_behind: second local extremum in the day.
#     Returns a Series indexed by trading_day.
#     """
#     if 'trading_day' not in df.columns:
#         raise ValueError("DataFrame must contain 'trading_day' column for daily grouping.")
#     result = {}
#     for day, day_df in df.groupby('trading_day'):
#         price = day_df[price_col]
#         # Find local extrema: points where price changes direction
#         direction = pd.Series(np.sign(price.diff()), index=price.index)
#         inflection = (direction != direction.shift(1)) & (direction != 0) & (direction.shift(1) != 0)
#         extrema_idx = price.index[inflection]
#         if len(extrema_idx) >= 2:
#             first_extrema = price.loc[extrema_idx[0]]
#             second_extrema = price.loc[extrema_idx[1]]
#             if first_extrema != 0:
#                 result[day] = (first_extrema - second_extrema) / first_extrema
#             else:
#                 result[day] = np.nan
#         else:
#             result[day] = np.nan
#     return pd.Series(result)

# # 计算日内动量反转因子
# intraday_momentum_factor = tester.calc_factor(lambda df: intraday_momentum_reversal_factor(df, price_col='close_price'))

# # 计算IC
# ic_series, stats = tester.calc_ic(intraday_momentum_factor, daily_returns)
# print('intraday_momentum_reversal_factor\n', stats)
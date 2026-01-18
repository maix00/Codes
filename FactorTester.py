from datetime import datetime
from enum import Enum
import pandas as pd
import numpy as np
from typing import Callable, List, Dict, Optional, Tuple
import os
from Products import Futures, FuturesContract, ProductBase
from collections import Counter
import logging

logger_dir_path_default = '../data/factor_tester_log/'

class FactorTester:
    def __init__(self, file_paths: List[str], 
                 start_date: Optional[str] = None, end_date: Optional[str] = None,
                 time_col: str|List[str] = ['trading_day', 'trade_time'], #'trade_time',#['trading_day', 'trade_time'], 
                 futures_flag: bool = True, futures_adjust_col: Optional[List[str]] = None,
                 logger_file: bool = True, logger_dir_path: str = logger_dir_path_default,
                 logger_console: bool = False):
        
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:

            self.logger.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

            if logger_console:
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(formatter)
                self.logger.addHandler(console_handler)

            if logger_file:
                if not os.path.exists(logger_dir_path):
                    os.makedirs(logger_dir_path)
                logger_file_path = os.path.join(logger_dir_path, "factor_tester_{}.log".format(datetime.now().strftime("%Y%m%d")))
                file_handler = logging.FileHandler(logger_file_path, encoding='utf-8')
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
        
        self.set_time_col(time_col)
        self.products = []
        self.data = {}
        self.data_freq = {}
        self.factor_data = {}
        self.factor_freq = {}
        self.product_mapping = {}
        self.return_data = {}
        for path in file_paths:
            product_name = path.split('/')[-1].replace('.parquet', '')
            product = Futures(product_name) if futures_flag else ProductBase(product_name)
            self.add_data(product, path, futures_adjust_col=futures_adjust_col)
        self.start_date = start_date
        self.end_date = end_date
        self.futures_flag = futures_flag
        self.futures_adjust_col = futures_adjust_col
        self.logger.info(f"FactorTester initialized with {len(self.products)} products")

    def set_time_col(self, time_col: str|List[str]):
        self.time_col = time_col
        self.time_col_num = 0
        if isinstance(time_col, str):
            self.time_col = [time_col]
        if len(self.time_col) == 1:
            self.time_col_num = 1
            self.logger.info(f"将`time_col`设置为单列: {self.time_col[0]}")
        elif len(self.time_col) == 2:
            self.time_col_num = 2
            self.logger.info(f"将`time_col`设置为双列: {self.time_col[0]} and {self.time_col[1]}")
            self.logger.info("首列假定为`trading_day`，第二列假定为`trade_time`")
        else:
            raise ValueError("`time_col`必须是一个字符串或两个字符串：交易时间，或者交易日和交易时间的组合。")
        if hasattr(self, "products"):
            for product in self.products:
                df = self.data[product]
                self.data[product] = df.reset_index().set_index(self.time_col)
                self.logger.info(f"将{product}的索引设置为{self.time_col}")

    def add_data(self, product: ProductBase, df: pd.DataFrame|str, futures_adjust_col: Optional[List[str]] = None):
        if isinstance(df, str):
            assert os.path.exists(df), "数据文件不存在。"
            self.logger.info(f"{product}的数据文件开始加载: {df}")
            if df.endswith('.parquet'):
                df = pd.read_parquet(df)
            elif df.endswith('.csv'):
                df = pd.read_csv(df)
            elif df.endswith('.xlsx'):
                df = pd.read_excel(df)
            else:
                raise ValueError(f"不支持的数据格式：{df}")
            self.logger.info(f"{product}的数据文件加载完成，数据行数: {len(df)}")
        if df.empty:
            self.logger.warning(f"{product}的数据为空，跳过添加。")
            return
        if isinstance(product, Futures):
            futures_adjust_col = futures_adjust_col or self.futures_adjust_col
            if futures_adjust_col:
                futures_adjust_col_adjusted = [col + '_adjusted' for col in futures_adjust_col]
                if any(col not in df.columns for col in futures_adjust_col_adjusted):
                    try:
                        assert 'adjustment_mul' in df.columns
                        assert 'adjustment_add' in df.columns
                    except:
                        message = f"{product}的数据中缺少调整列`adjustment_mul`和`adjustment_add`，无法进行价格调整。"
                        self.logger.error(message)
                        raise AssertionError(message)
                    for col, col_adj in zip(futures_adjust_col, futures_adjust_col_adjusted):
                        df[col_adj] = df[col] * df['adjustment_mul'] + df['adjustment_add']
                        self.logger.info(f"{product}的`{col}`列已进行价格调整, 新列名为`{col_adj}`。")
        if self.time_col_num == 1:
            try:
                assert self.time_col[0] in df.columns
            except:
                self.logger.error(f"{product}的数据中缺少时间列{self.time_col[0]}，无法设置索引。")
                raise AssertionError(f"{product}的数据中缺少时间列{self.time_col[0]}，无法设置索引。")
            try:
                df[self.time_col[0]] = pd.to_datetime(df[self.time_col[0]])
                self.logger.info(f"将{product}的时间列{self.time_col[0]}转换为datetime格式")
            except:
                self.logger.error(f"将{product}的时间列{self.time_col[0]}转换为datetime格式失败")
                raise TypeError(f"无法将{product}的{self.time_col[0]}转换为datetime格式")
        elif self.time_col_num == 2:
            try:
                assert self.time_col[0] in df.columns
            except:
                self.logger.error(f"{product}的数据中缺少时间列{self.time_col[0]}，无法设置索引。")
                raise AssertionError(f"{product}的数据中缺少时间列{self.time_col[0]}，无法设置索引。")
            try:
                df[self.time_col[0]] = pd.to_datetime(df[self.time_col[0]])
                self.logger.info(f"将{product}的时间列{self.time_col[0]}转换为datetime格式")
            except:
                self.logger.error(f"将{product}的时间列{self.time_col[0]}转换为datetime格式失败")
                raise TypeError(f"无法将{product}的{self.time_col[0]}转换为datetime格式")
            try:
                assert self.time_col[1] in df.columns
            except:
                self.logger.error(f"{product}的数据中缺少时间列{self.time_col[1]}，无法设置索引。")
                raise AssertionError(f"{product}的数据中缺少时间列{self.time_col[1]}，无法设置索引。")
            try:
                df[self.time_col[1]] = pd.to_datetime(df[self.time_col[1]])
                self.logger.info(f"将{product}的时间列{self.time_col[1]}转换为datetime格式")
            except:
                self.logger.error(f"将{product}的时间列{self.time_col[1]}转换为datetime格式失败")
                raise TypeError(f"无法将{product}的{self.time_col[1]}转换为datetime格式")
        self.data[product] = df.reset_index().set_index(self.time_col)
        self.logger.info(f"将{product}的索引设置为{self.time_col}")
        self.data_freq[product] = pd.Series(self.data[product].index.get_level_values(self.time_col_num - 1).sort_values().diff().dropna()).mode()[0]
        self.logger.info(f"{product}的数据频率为{self.data_freq[product]}")
        self.products = set(self.data.keys())
        self.product_mapping[product.name] = product

    def calc_interval_return(self, interval: int = 1, price_col: str = 'close_price_adjusted') -> pd.DataFrame:
        returns = {}
        for c, df in self.data.items():
            assert price_col in df.columns, f"{price_col} not in DataFrame columns for contract {c}"
            returns[c] = df[price_col].pct_change(periods=interval)
        return pd.DataFrame(returns)
    
    def calc_daily_return(self, price_col: str = 'open_price',
                          open_market: bool = False, close_market: bool = False) -> pd.DataFrame:
        daily_returns = {}
        for c, df in self.data.items():
            daily_returns[c] = daily_return(df, price_col=price_col, 
                                            open_market=open_market, close_market=close_market)
        return pd.DataFrame(daily_returns)

    def calc_return(self, price_col: str = 'close_price', date_index: bool = False, 
                    time_cols: Optional[str|List[str]] = ['trading_day', 'trade_time'], 
                    calc_freq: Optional[str|pd.Timedelta] = None,
                    return_freq: Optional[str|pd.Timedelta] = None, 
                    delta_return: bool = False, log_return: bool = False,
                    open_market: bool = False, close_market: bool = False) -> tuple[pd.DataFrame, bool, bool]:
        
        assert not (delta_return and log_return), "`delta_return`和`log_return`不能同时为True。"

        factor_frequency = pd.Series(self.factor_freq.values()).unique()
        factor_frequency = factor_frequency[0] if len(factor_frequency) == 1 else None
        assert factor_frequency is None or isinstance(factor_frequency, pd.Timedelta)
        data_frequency = pd.Series(self.data_freq.values()).unique()
        data_frequency = data_frequency[0] if len(data_frequency) == 1 else None
        assert data_frequency is None or isinstance(data_frequency, pd.Timedelta)

        if calc_freq is None:
            calc_freq = factor_frequency if factor_frequency is not None else data_frequency
        elif isinstance(calc_freq, str):
            calc_freq = pd.Timedelta(calc_freq)
        
        if return_freq is None:
            return_freq = factor_frequency if factor_frequency is not None else data_frequency
        elif isinstance(return_freq, str):
            return_freq = pd.Timedelta(return_freq)

        if return_freq == pd.Timedelta('1 day') and calc_freq == pd.Timedelta('1 day') and\
            date_index and (open_market or close_market):
            assert not (open_market and close_market), "`open_market`和`close_market`不能同时为True。" 
            returns_df = self.calc_daily_return(price_col=price_col, open_market=open_market, close_market=close_market)
            if not delta_return:
                returns_df = returns_df + 1
            if log_return:
                returns_df = pd.DataFrame(np.log(returns_df))
            return (returns_df, delta_return, log_return)
        
        if data_frequency is not None and calc_freq is not None and calc_freq == data_frequency and\
            return_freq is not None and return_freq.total_seconds() % data_frequency.total_seconds() == 0:
            interval = int(return_freq / data_frequency)
            returns_df = self.calc_interval_return(interval=interval, price_col=price_col)
            if not delta_return:
                returns_df = returns_df + 1
            if log_return:
                returns_df = pd.DataFrame(np.log(returns_df))
            return (returns_df, delta_return, log_return)

        returns_all = {}
    
        for product, df in self.data.items():
            if price_col not in df.columns:
                raise ValueError(f"DataFrame {product} 中缺少列 {price_col}")
            
            calc_freq = calc_freq if calc_freq is not None else self.data_freq[product]
            return_freq = return_freq if return_freq is not None else self.data_freq[product]
            
            temp_df = df.copy()
            if time_cols:
                temp_df = temp_df.reset_index()
                temp_df[time_cols[0]] = pd.to_datetime(temp_df[time_cols[0]])#.dt.date
                temp_df[time_cols[1]] = pd.to_datetime(temp_df[time_cols[1]])
                temp_df = temp_df.set_index(time_cols)
            temp_df = temp_df.sort_index()
            temp_series = temp_df[price_col].droplevel(0)

            all_dates = temp_df.index.get_level_values(0)
            indices_of_next_row_changed = np.where(all_dates[:-1] != all_dates[1:])[0]
            indices_market = indices_of_next_row_changed if close_market else indices_of_next_row_changed + 1 if open_market else None
            calc_step = int(calc_freq / pd.Timedelta('1 day'))
            return_step = int(return_freq / pd.Timedelta('1 day'))
            all_datetimes = temp_df.index.get_level_values(1)

            if calc_freq:
                if calc_freq.total_seconds() % pd.Timedelta('1 day').total_seconds() == 0 and (open_market or close_market):
                    assert indices_market is not None
                    start_datetimes = all_datetimes[0:1].append(all_datetimes[indices_market[calc_step-1::calc_step]])
                else:
                    last_t = all_datetimes[0] - calc_freq # 保证第一个点被选中
                    keep_indices = []
                    for i, t in enumerate(all_datetimes):
                        if t >= last_t + calc_freq:
                            keep_indices.append(i)
                            last_t = t
                    start_datetimes = all_datetimes[np.array(keep_indices)]
            else:
                start_datetimes = all_datetimes
            start_prices = temp_series.reindex(start_datetimes)
            assert len(start_prices) == len(start_datetimes), "start_prices 和 start_datetimes 长度不一致。"

            if return_freq is not None and\
                return_freq.total_seconds() % pd.Timedelta('1 day').total_seconds() == 0 and (open_market or close_market):
                    assert indices_market is not None
                    end_datetimes = all_datetimes[indices_market[return_step-1::calc_step]]
                    len_diff = len(start_datetimes) - len(end_datetimes)
                    assert len_diff >= 0, "start_datetimes 长度不得比 end_datetimes 短。"
                    end_datetimes = end_datetimes.append(all_datetimes[indices_market[-len_diff:]] + return_freq)
            else:
                end_datetimes = start_datetimes + return_freq

            assert len(start_datetimes) == len(end_datetimes), "start_datetimes 和 end_datetimes 长度不一致。"
            
            end_prices = temp_series.reindex(end_datetimes)
            assert len(end_prices) == len(end_datetimes), "end_prices 和 end_datetimes 长度不一致。"
            
            raw_return = end_prices.values / start_prices.values

            if date_index:
                index = temp_df.index[temp_df.index.get_level_values(1).isin(start_datetimes)].get_level_values(0)
                assert len(index) == len(start_datetimes), "index 和 start_datetimes 长度不一致。"
                final_series = pd.Series(raw_return, index=index)
            else:
                final_series = pd.Series(raw_return, index=start_datetimes)

            if delta_return:
                final_series = final_series - 1
            elif log_return:
                final_series = np.log(final_series)
                
            returns_all[product] = final_series

        returns_df = pd.DataFrame(returns_all)
        return (returns_df, delta_return, log_return)

    def calc_factor_freq(self, data: pd.DataFrame, name: Optional[str] = None) -> Optional[pd.Timedelta]:
        
        if name is None:
            name = 'Factor_' + str(len(self.factor_freq))

        all_freqs = {}

        if isinstance(data.index, pd.MultiIndex):
            # 确保第一层是datetime(date)，第二层是datetime
            data.index = pd.MultiIndex.from_arrays([
                pd.to_datetime(data.index.get_level_values(0)),
                pd.to_datetime(data.index.get_level_values(1))
            ])
        elif not isinstance(data.index, pd.DatetimeIndex):
            data.index = pd.to_datetime(data.index)

        for product_name in data.columns:
            assert product_name in self.products
            freq_series = data[product_name]
            if len(freq_series) > 0:
                if len(freq_series.index) > 1:
                    all_freqs[product_name] = pd.Series(freq_series.index.get_level_values(len(freq_series.index.names) - 1).sort_values().diff().dropna()).mode()[0]
                    self.logger.info(f"产品 {product_name} 的因子 {name} 频率为 {all_freqs[product_name]}")
                else:
                    self.logger.warning(f"产品 {product_name} 因子 {name} 只有一个数据点，无法确定频率。")
            else:
                self.logger.warning(f"产品 {product_name} 因子 {name} 没有数据。")
        if all_freqs:
            self.factor_freq[name] = pd.Series(all_freqs.values()).mode()[0]
            for product_name in all_freqs:
                if all_freqs[product_name] != self.factor_freq[name]:
                    self.logger.warning(f"产品 {product_name} 的因子 {name} 频率 {all_freqs[product_name]} 与整体因子频率 {self.factor_freq[name]} 不符。")
            return self.factor_freq[name]
        else:
            return None
        
    def calc_factor(self, factor_func: Callable[[pd.DataFrame], pd.Series], factor_name: Optional[str] = None,
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
        factors_df = pd.DataFrame(factors)
        factor_name = 'Factor_' + str(len(self.factor_data)) if factor_name is None else factor_name
        self.factor_data[factor_name] = factors_df

        if set_freq:
            # 如果因子频率未指定，则尝试从数据中获取
            if factors and frequency is None:
                self.calc_factor_freq(data=factors_df, name=factor_name)
            else:
                self.factor_freq[factor_name] = frequency

        return factors_df
    
    def calc_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rank(axis=1, method='average', na_option='keep', pct=True)

    def calc_ic(self, factor_names: str|List[str], 
                return_price_col: str = 'close_price_adjusted',
                return_open_market: bool = False, return_close_market: bool = False,
                return_freq: Optional[str|pd.Timedelta] = None,
                start_date: Optional[str] = None, end_date: Optional[str] = None) -> tuple[pd.DataFrame, pd.DataFrame]:
        if isinstance(factor_names, str):
            factor_names = [factor_names]
        ic_series = {}
        ic_stats = {}
        for factor_name in factor_names:
            factor_rank = self.calc_rank(self.factor_data[factor_name])
            return_df, delta_return, _ = self.cache_return_by_factor_name(factor_name, price_col=return_price_col,
                                                                    open_market=return_open_market, close_market=return_close_market,
                                                                    return_freq=return_freq)
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
            ic_series[factor_name] = pd.Series(ic, index=dt_index)
            avg_coverage = np.mean(coverage)
            stats_df = self.ic_stats(ic_series[factor_name])
            stats_df['avg_coverage'] = avg_coverage
            ic_stats[factor_name] = stats_df
        return pd.DataFrame(ic_series), pd.DataFrame(ic_stats)

    def ic_stats(self, ic_series: pd.Series) -> pd.Series:
        mean = ic_series.mean()
        std = ic_series.std()
        ir = mean / std if std != 0 else np.nan
        t_stat = mean / (std / np.sqrt(len(ic_series.dropna()))) if std != 0 and len(ic_series.dropna()) > 1 else np.nan
        max_ic = ic_series.max()
        min_ic = ic_series.min()
        stats_df = pd.Series({
            'mean': mean, 'std': std, 'IR': ir, 't_stat': t_stat, 'max': max_ic, 'min': min_ic
        })
        return stats_df
    
    def cache_return(self, price_col: str = 'close_price_adjusted',
                     date_index: bool = False, open_market: bool = False, close_market: bool = False,
                     return_freq: Optional[str|pd.Timedelta] = None,
                     calc_freq: Optional[str|pd.Timedelta] = None) -> tuple[pd.DataFrame, bool, bool]:
        return_label = (return_freq, calc_freq, price_col, date_index, open_market, close_market)
        if return_label in self.return_data and\
            set(self.return_data[return_label][0].columns) == self.products:
            return self.return_data[return_label]
        else:
            outcome = self.calc_return(price_col=price_col, return_freq=return_freq, calc_freq=calc_freq,
                                       date_index=date_index, open_market=open_market, close_market=close_market)
            self.return_data[return_label] = outcome
            return outcome
        
    def cache_return_by_factor_name(self, factor_name: str, price_col: str = 'close_price_adjusted',
                     date_index: Optional[bool] = None, open_market: bool = False, close_market: bool = False,
                     return_freq: Optional[str|pd.Timedelta] = None,
                     calc_freq: Optional[str|pd.Timedelta] = None) -> tuple[pd.DataFrame, bool, bool]:
        return_freq = return_freq if return_freq is not None else self.factor_freq[factor_name]
        calc_freq = calc_freq if calc_freq is not None else self.factor_freq[factor_name]
        date_index = date_index if date_index is not None else \
            True if self.factor_freq[factor_name] == pd.Timedelta('1 day') else False
        return self.cache_return(price_col=price_col, date_index=date_index, open_market=open_market,
                                 close_market=close_market, return_freq=return_freq, calc_freq=calc_freq)

    def group_classes(self, factor_name: str, n_groups: int = 5, plot_flag: bool = False, 
                      start_date: Optional[str] = None, end_date: Optional[str] = None,
                      return_price_col: str = 'close_price_adjusted',
                      return_open_market: bool = False, return_close_market: bool = False,
                      return_freq: Optional[str|pd.Timedelta] = None,
                      plot_n_group_list: Optional[List[int]] = None) -> Tuple[Dict[str, Dict[str, List[ProductBase]]], Dict[str, Dict[str, float]]]:
        """
        For each datetime, split contracts into n_groups groups.
        Each group is a dict: {datetime_str: [contract names]}.
        n_groups: number of groups to split into (default: 5)
        """
        # Calculate daily returns at next open
        returns, delta_return, _ = self.cache_return_by_factor_name(factor_name, price_col=return_price_col,
                                                                    open_market=return_open_market, close_market=return_close_market,
                                                                    return_freq=return_freq)

        if not delta_return:
            returns = returns - 1

        assert factor_name in self.factor_data, "Factor name must be in factor_data."
        factor_df = self.factor_data[factor_name]

        # Plot adjustment for group numbers
        plot_n_group_list = [n_groups + n_group if n_group < 0 else n_group for n_group in plot_n_group_list] if plot_n_group_list else None

        group_names = ['group_' + str(i) for i in range(n_groups)]
        group_names = group_names[::-1]
        groups = {name: {} for name in group_names}
        returns_groups = {name: {} for name in group_names}
        
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
                    returns_groups[group_names[group_idx]][dt_str] = np.mean(np.asarray(returns.loc[dt_str][contract_names].values, dtype=float))
            
            # Fill remaining groups (if n < n_groups) with empty lists
            for i in range(min(n, n_groups), n_groups):
                groups[group_names[i]][dt_str] = []
                returns_groups[group_names[i]][dt_str] = np.nan

        report_groups = {}
        for name in group_names:
            dates = list(returns_groups[name].keys()) if returns_groups[name] else []
            dates = [date for date in dates if start_date <= date] if start_date else dates
            dates = [date for date in dates if date <= end_date] if end_date else dates
            returns = [returns_groups[name][date] for date in dates]
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
            # print(start_date, end_date)
            dates = []  # Initialize dates as an empty list
            for name in group_names:
                if plot_n_group_list is not None and name.split('_')[-1] not in [str(n) for n in plot_n_group_list]:
                    continue
                dates = list(returns_groups[name].keys()) if returns_groups[name] else []
                dates = [date for date in dates if start_date <= date] if start_date else dates
                dates = [date for date in dates if date <= end_date] if end_date else dates
                returns = [returns_groups[name][date] for date in dates]
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

        return groups, returns_groups

def daily_return(df, price_col: str = 'close_price', open_market: bool = False, close_market: bool = False) -> pd.Series:
    daily_price = df.groupby('trading_day')[price_col]
    assert (open_market or close_market) and not (open_market and close_market)
    if open_market:
        return daily_price.first().pct_change().shift(-1)
    elif close_market:
        return daily_price.last().pct_change()
    else:
        raise ValueError("`open_market`和`close_market`有且只有一个为True。")

def integrated_ic_test_daily(factor_func: Callable, factor_name: Optional[str] = None,
                             n_groups: int = 5, plot_n_group_list: Optional[List[int]] = None,):
    
    import cProfile
    import pstats

    profiler = cProfile.Profile()
    profiler.enable()
    
    parquet_dir = '../data/main_mink/'
    file_list = [
        os.path.join(parquet_dir, f)
        for f in os.listdir(parquet_dir)
        if f.endswith('.parquet') and '_S' not in f and '-S' not in f
    ]
    tester = FactorTester(file_list, #start_date='2025-01-01', 
                      end_date='2025-05-30', 
                      futures_flag=True, futures_adjust_col=['close_price', 'open_price'])

    tester.calc_factor(lambda df: factor_func(df, price_col='close_price_adjusted'), factor_name=factor_name)
    factor_name = factor_name if factor_name is not None else list(tester.factor_data.keys())[-1]
    _, stats = tester.calc_ic(factor_names=factor_name, return_price_col='open_price_adjusted', return_open_market=True)
    print(factor_name, 'IC Stats:\n', stats)

    groups, returns_groups = tester.group_classes(factor_name, 
                                                  plot_flag=True, n_groups=n_groups, plot_n_group_list=plot_n_group_list,
                                                  start_date='2025-01-01', end_date='2025-12-31',
                                                  return_price_col='open_price_adjusted', return_open_market=True
                                                )
    # # Get the earliest five dates from the 'top' group
    # earliest_dates = sorted(groups['group_0'].keys())[:5]
    # for date in earliest_dates:
    #     print(date, groups['group_0'][date])
    #     print(date, returns_groups['group_0'][date])
    #     pass

    profiler.disable()
    # 输出分析结果
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')  # 按累计时间排序
    stats.print_stats(20)  # 显示前20个耗时最多的函数
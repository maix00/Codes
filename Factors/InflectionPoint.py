import numpy as np
import pandas as pd
from typing import Callable
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FactorTester import FactorTester, log_daily_return, integrated_ic_test_daily
from BackTester import Futures, PortfolioBackTester

def count_inflection_points(price_series: pd.Series) -> pd.Series:
    """
    Count inflection points in a price series.
    An inflection point is defined where the direction of price change reverses.
    Returns a Series indexed as price_series.
    """
    # Calculate price change direction: 1 for up, -1 for down, 0 for no change
    direction = pd.Series(np.sign(price_series.diff()), index=price_series.index)
    # Find where direction changes (excluding 0 to 0)
    inflection = (direction != direction.shift(1)) & (direction != 0) & (direction.shift(1) != 0)
    # Cumulative count of inflection points per row
    inflection_points = inflection.astype(int)
    return inflection_points.rolling(window=len(price_series), min_periods=1).sum()

def inflection_point_factor(df: pd.DataFrame, price_col: str = 'close_price_adjusted') -> pd.Series:
    """
    For a given DataFrame, calculate the inflection point factor:
    today's inflection count minus the mean of the past 5 days' inflection counts.
    """
    price = df[price_col]
    # Calculate direction of price change
    direction = pd.Series(np.sign(price.diff()), index=price.index)
    # Inflection point: direction changes (not including 0 to 0)
    inflection = (direction != direction.shift(1)) & (direction != 0) & (direction.shift(1) != 0)
    # Daily sum of inflection points (if intraday, group by date)
    if 'trading_day' in df.columns:
        daily_inflection = inflection.astype(int).groupby(df['trading_day']).sum()
    else:
        raise ValueError("DataFrame must contain 'trading_day' column for daily grouping.")
    # 5-day rolling mean (excluding today)
    rolling_mean = daily_inflection.rolling(window=5, min_periods=1).mean().shift(1)
    factor = daily_inflection - rolling_mean
    return factor

if __name__ == '__main__':
    integrated_ic_test_daily(inflection_point_factor, n_groups=5,)
    # integrated_ic_test_daily(inflection_point_factor, n_groups=20, plot_n_group_list=[0, -1])
    
import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FactorTester import integrated_ic_test_daily

def inflection_point_factor(df: pd.DataFrame, price_col: str = 'close_price_adjusted') -> pd.Series:
    price = df[price_col]
    direction = pd.Series(np.sign(price.diff()), index=price.index)
    inflection = (direction != direction.shift(1)) & (direction != 0) & (direction.shift(1) != 0)
    daily_inflection = inflection.astype(int).groupby(level='trading_day').sum()
    rolling_mean = daily_inflection.rolling(window=5, min_periods=1).mean().shift(1)
    factor = daily_inflection - rolling_mean
    return factor

if __name__ == '__main__':
    integrated_ic_test_daily(inflection_point_factor, n_groups=5, factor_name='InflectionPoint5D')
    
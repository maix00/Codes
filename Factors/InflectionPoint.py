import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FactorTester import FactorGrid, PriceColumnMapping
from typing import List, Dict, Any

class InflP(FactorGrid): # Inflection Point

    params_space: Dict[str, List[Any]] = {
        'PC': list(PriceColumnMapping.keys()),
        # 'W': [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200],
        'W': [100, 150, 200],
    }

    default_params: Dict[str, Any] = {'PC': 'CA', 'W': 5}

    def _factor_func(self, df: pd.DataFrame, PC: str = 'CA', W: int = 5) -> pd.Series:
        price = df[PriceColumnMapping[PC]]
        direction = pd.Series(np.sign(price.diff()), index=price.index)
        inflection = (direction != direction.shift(1)) & (direction != 0) & (direction.shift(1) != 0)
        daily_inflection = inflection.astype(int).groupby(level='trading_day').sum()
        rolling_mean = daily_inflection.rolling(window=W, min_periods=1).mean().shift(1)
        factor = daily_inflection - rolling_mean
        return factor

if __name__ == '__main__':
    InflP().factor_grid_test()
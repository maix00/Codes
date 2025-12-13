import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from datetime import datetime
from pathlib import Path
from typing import Sequence, Union, List, Dict, Optional

import pandas as pd

from qf_lib.common.enums.security_type import SecurityType
from qf_lib.common.enums.frequency import Frequency
from qf_lib.common.enums.price_field import PriceField
from qf_lib.common.tickers.tickers import Ticker
from qf_lib.data_providers.csv.csv_data_provider import CSVDataProvider

class MinkTicker(Ticker):
    def __init__(self, ticker: str, security_type: SecurityType = SecurityType.FUTURE, 
                 point_value: int = 1, currency: Optional[str] = "CNY"):
        super().__init__(ticker, security_type, point_value, currency)

    def from_string(self, ticker_str: str | Sequence[str]) -> Ticker | Sequence[Ticker]:
        return super().from_string(ticker_str)

    @classmethod
    def from_ticker_str(cls, ticker_str: str | Sequence[str]) -> "MinkTicker" | Sequence["MinkTicker"]:
        if isinstance(ticker_str, str):
            return MinkTicker(ticker_str)
        else:
            return [MinkTicker(t) for t in ticker_str]

data_folder = Path("../data/data_mink_product_2025_main")
file_names: Sequence[str] = [f.stem for f in data_folder.iterdir() if f.is_file() and f.suffix == ".csv"]

print("Starting data loading test...")
MinkDataProvider = CSVDataProvider(
    path="../data/data_mink_main_ticks.csv", 
    tickers=MinkTicker.from_ticker_str(file_names),
    index_col='trade_time',
    field_to_price_field_dict={
        'open_price': PriceField.Open,
        'highest_price': PriceField.High,
        'lowest_price': PriceField.Low,
        'close_price': PriceField.Close,
        'volume': PriceField.Volume,
    },
    start_date=datetime(2025, 1, 2, 9, 1),
    end_date=datetime(2025, 11, 28, 15, 0),
    frequency=Frequency.MIN_1,
    dateformat="%Y-%m-%d %H:%M:%S",
    ticker_col="product_id"
)
print("Data loading test finished.")

l = MinkDataProvider.historical_price(MinkTicker("BC"), PriceField.Close, 10, datetime(2025, 1, 10))

print(l, l.dtype)


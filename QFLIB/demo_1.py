import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from datetime import datetime
from typing import Sequence, Optional

from qf_lib.common.enums.security_type import SecurityType
from qf_lib.common.enums.frequency import Frequency
from qf_lib.common.enums.price_field import PriceField
from qf_lib.common.tickers.tickers import Ticker
from qf_lib.data_providers.csv.csv_data_provider import CSVDataProvider

from qf_lib.settings import Settings
from qf_lib.starting_dir import set_starting_dir_abs_path

from qf_lib.documents_utils.document_exporting.pdf_exporter import PDFExporter
from qf_lib.documents_utils.excel.excel_exporter import ExcelExporter
from qf_lib.backtesting.trading_session.backtest_trading_session_builder import BacktestTradingSessionBuilder

from qf_lib.backtesting.events.time_event.periodic_event.calculate_and_place_orders_event import \
    CalculateAndPlaceOrdersPeriodicEvent

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Strategies.MA import SimpleMAStrategy
from tqdm import tqdm

from demo_2 import MultipleParquetFuturesDataProvider
import glob

settings_path = "./QFLIB/settings.json"
secret_settings_path = "../data/QFLIB/secret_settings.json"
settings = Settings(settings_path, secret_settings_path)

set_starting_dir_abs_path('/Users/maxdeux/Documents/GTHT/Codes')

class ChinaFutures(Ticker):
    def __init__(self, ticker: str, security_type: SecurityType = SecurityType.FUTURE, 
                 point_value: int = 1, currency: Optional[str] = "CNY"):
        super().__init__(ticker, security_type, point_value, currency)

    def from_string(self, ticker_str: str | Sequence[str]) -> Ticker | Sequence[Ticker]:
        return super().from_string(ticker_str)

    @classmethod
    def from_ticker_str(cls, ticker_str: str | Sequence[str]) -> "ChinaFutures" | Sequence["ChinaFutures"]:
        if isinstance(ticker_str, str):
            return ChinaFutures(ticker_str)
        else:
            return [ChinaFutures(t) for t in ticker_str]


def local_data_provider_adjusted(ticker_names: str|Sequence[str], save_cache_mode: bool = False) -> MultipleParquetFuturesDataProvider:
    return MultipleParquetFuturesDataProvider(
        path="../data/main_mink_adjusted.parquet", 
        tickers=ChinaFutures.from_ticker_str(ticker_names),
        index_col='trade_time',
        field_to_price_field_dict={
            'open_price_adjusted': PriceField.Open,
            'highest_price_adjusted': PriceField.High,
            'lowest_price_adjusted': PriceField.Low,
            'close_price_adjusted': PriceField.Close,
            'volume': PriceField.Volume,
        },
        start_date=datetime(2025, 1, 1, 0, 0),
        end_date=datetime(2025, 12, 31, 23, 59),
        frequency=Frequency.MIN_1,
        dateformat="%Y-%m-%d %H:%M:%S",
        ticker_col="unique_instrument_id",
        qf_cache_path="../data/main_mink_2025_adjusted_qflib/",
        save_cache_mode=save_cache_mode
    )

def local_data_provider(ticker_names: str, save_cache_mode: bool = False) -> MultipleParquetFuturesDataProvider:
    return MultipleParquetFuturesDataProvider(
        path="../data/main_mink/" + ticker_names + ".parquet", 
        tickers=ChinaFutures.from_ticker_str(ticker_names),
        index_col='trade_time',
        field_to_price_field_dict={
            'open_price': PriceField.Open,
            'highest_price': PriceField.High,
            'lowest_price': PriceField.Low,
            'close_price': PriceField.Close,
            'volume': PriceField.Volume,
        },
        start_date=datetime(2025, 1, 1, 0, 0),
        end_date=datetime(2025, 12, 31, 23, 59),
        frequency=Frequency.MIN_1,
        dateformat="%Y-%m-%d %H:%M:%S",
        ticker_col="unique_instrument_id",
        qf_cache_path="../data/main_mink_qflib/",
        save_cache_mode=save_cache_mode
    )

# # ticker_names = pd.read_parquet("../data/main_mink.parquet")["unique_instrument_id"].unique().tolist()
# parquet_files = glob.glob("../data/main_mink/*.parquet")
# ticker_names = [os.path.splitext(os.path.basename(f))[0] for f in parquet_files]
# for ticker_name in tqdm(ticker_names, desc="Processing data for tickers"):
#     # local_data_provider_adjusted(ticker_names=ticker_name, save_cache_mode=True)
#     local_data_provider(ticker_names=ticker_name, save_cache_mode=True)

ticker_names = ['BC.INE', 'A.DCE', 'CZCE|F|MA|2509']
print("Begin loading data ...")
local_data_provider_instance = local_data_provider_adjusted(ticker_names=ticker_names)
print("End loading data ...")

if True:
    session_builder = BacktestTradingSessionBuilder(settings, PDFExporter(settings), ExcelExporter(settings))
    session_builder.set_frequency(Frequency.MIN_1)
    session_builder.set_backtest_name('Simple MA Strategy Demo')
    session_builder.set_portfolio_currency("CNY")

    session_builder.set_data_provider(local_data_provider_instance)
    session_builder.set_market_open_and_close_time({"hour": 21, "minute": 0}, {"hour": 15, "minute": 0})
    trading_session = session_builder.build(datetime(2025, 1, 10, 9, 1), datetime(2025, 5, 30, 15, 0))

    strategy = SimpleMAStrategy(trading_session, ChinaFutures("BC.INE"), long_ma_len=20, short_ma_len=5)
    CalculateAndPlaceOrdersPeriodicEvent.set_frequency(Frequency.MIN_1)
    CalculateAndPlaceOrdersPeriodicEvent.set_start_and_end_time(
        {"hour": 21, "minute": 0}, {"hour": 15, "minute": 0})
    strategy.subscribe(CalculateAndPlaceOrdersPeriodicEvent)

    trading_session.start_trading()

else:
    l1_1 = local_data_provider_instance.historical_price(ChinaFutures("BC.INE"), PriceField.Close, 85, datetime(2025, 7, 25, 10, 37))
    l1_2 = local_data_provider_instance.get_history(ChinaFutures("BC.INE"), ["adjustment_mul", "adjustment_add", "close_price", "close_price_adjusted", "volume", "DATA_ISSUE", "DATA_ISSUE_SOLUTION"], datetime(2025, 1, 22, 14, 39), datetime(2025, 1, 22, 14, 45))
    with pd.option_context('display.max_columns', None):
        print(l1_1)
        print(l1_2)
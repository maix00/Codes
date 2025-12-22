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

session_builder = BacktestTradingSessionBuilder(settings, PDFExporter(settings), ExcelExporter(settings))
session_builder.set_frequency(Frequency.MIN_1)
session_builder.set_backtest_name('Simple MA Strategy Demo')
session_builder.set_portfolio_currency("CNY")

data_path = "../data/main_mink_adjusted.csv"
ticker_names = pd.read_csv(data_path, usecols=["unique_instrument_id"])["unique_instrument_id"].unique().tolist()

print("Start of data loading ...")
ChinaFuturesMink2025 = CSVDataProvider(
    path=data_path, 
    tickers=ChinaFutures.from_ticker_str(ticker_names),
    index_col='trade_time',
    field_to_price_field_dict={
        'open_price_adjusted': PriceField.Open,
        'highest_price_adjusted': PriceField.High,
        'lowest_price_adjusted': PriceField.Low,
        'close_price_adjusted': PriceField.Close,
        'volume': PriceField.Volume,
    },
    start_date=datetime(2025, 1, 2, 9, 1),
    end_date=datetime(2025, 11, 30),
    frequency=Frequency.MIN_1,
    dateformat="%Y-%m-%d %H:%M:%S",
    ticker_col="unique_instrument_id"
)
print("End of data loading.")

session_builder.set_data_provider(ChinaFuturesMink2025)
session_builder.set_market_open_and_close_time({"hour": 21, "minute": 0}, {"hour": 15, "minute": 0})
trading_session = session_builder.build(datetime(2025, 1, 10, 9, 1), datetime(2025, 5, 30, 15, 0))

# l1_1 = ChinaFuturesMink2025.historical_price(ChinaFutures("BC.INE"), PriceField.Close, 85, datetime(2025, 7, 25, 10, 37))
# l1_2 = ChinaFuturesMink2025.get_history(ChinaFutures("BC.INE"), ["close_price", "close_price_adjusted", "volume", "DATA_ISSUE", "DATA_ISSUE_SOLUTION"], datetime(2025, 1, 22, 14, 39), datetime(2025, 1, 22, 14, 45))
# with pd.option_context('display.max_columns', None):
#     print(l1_1)
#     print(l1_2)

strategy = SimpleMAStrategy(trading_session, ChinaFutures("BC.INE"), long_ma_len=20, short_ma_len=5)
CalculateAndPlaceOrdersPeriodicEvent.set_frequency(Frequency.MIN_1)
CalculateAndPlaceOrdersPeriodicEvent.set_start_and_end_time(
    {"hour": 21, "minute": 0}, {"hour": 15, "minute": 0})
strategy.subscribe(CalculateAndPlaceOrdersPeriodicEvent)

trading_session.start_trading()
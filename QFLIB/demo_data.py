import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Sequence, Union, List, Dict, Optional

from qf_lib.common.enums.security_type import SecurityType
from qf_lib.common.enums.frequency import Frequency
from qf_lib.common.enums.price_field import PriceField
from qf_lib.common.tickers.tickers import Ticker
from qf_lib.data_providers.csv.csv_data_provider import CSVDataProvider
from qf_lib.data_providers.abstract_price_data_provider import AbstractPriceDataProvider
from qf_lib.backtesting.strategies.abstract_strategy import AbstractStrategy
from qf_lib.backtesting.order.execution_style import MarketOrder
from qf_lib.backtesting.order.time_in_force import TimeInForce
from qf_lib.backtesting.trading_session.backtest_trading_session import BacktestTradingSession
from qf_lib.backtesting.events.time_event.regular_time_event.market_open_event import MarketOpenEvent
from qf_lib.backtesting.events.time_event.regular_time_event.market_close_event import MarketCloseEvent

from os.path import join, dirname, abspath
from qf_lib.settings import Settings
from qf_lib.starting_dir import set_starting_dir_abs_path

settings_path = "./QFLIB/settings.json"
secret_settings_path = "../data/QFLIB/secret_settings.json"
settings = Settings(settings_path, secret_settings_path)

set_starting_dir_abs_path('/Users/maxdeux/Documents/GTHT/Codes')

from qf_lib.documents_utils.document_exporting.pdf_exporter import PDFExporter
from qf_lib.documents_utils.excel.excel_exporter import ExcelExporter
from qf_lib.backtesting.trading_session.backtest_trading_session_builder import BacktestTradingSessionBuilder

from qf_lib.backtesting.events.time_event.periodic_event.calculate_and_place_orders_event import \
    CalculateAndPlaceOrdersPeriodicEvent

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

backtest_name = 'Simple MA Strategy Demo'
ticker = MinkTicker("A")

pdf_exporter = PDFExporter(settings)
excel_exporter = ExcelExporter(settings)

session_builder = BacktestTradingSessionBuilder(settings, pdf_exporter, excel_exporter)
session_builder.set_frequency(Frequency.MIN_1)
session_builder.set_backtest_name(backtest_name)
session_builder.set_portfolio_currency("CNY")  # Set the portfolio currency explicitly

# # 检查两个CSV文件的行数是否相同，并找出缺失的行
# def compare_csv_rows(file1: str, file2: str, key_cols: List[str] = ["trade_time", "unique_instrument_id"]):
#     df1 = pd.read_csv(file1, usecols=key_cols)
#     df2 = pd.read_csv(file2, usecols=key_cols)

#     print(f"{file1} 行数: {len(df1)}")
#     print(f"{file2} 行数: {len(df2)}")

#     # 合并为字符串用于集合比较
#     df1_keys = set(df1.astype(str).agg("_".join, axis=1))
#     df2_keys = set(df2.astype(str).agg("_".join, axis=1))

#     missing_in_1 = df2_keys - df1_keys
#     missing_in_2 = df1_keys - df2_keys

#     if missing_in_1:
#         print(f"{file1} 缺少 {len(missing_in_1)} 行（相对于 {file2}）")
#     if missing_in_2:
#         print(f"{file2} 缺少 {len(missing_in_2)} 行（相对于 {file1}）")

#     # 打印缺失的行
#     if missing_in_1:
#         print(f"{file1} 缺少的行:")
#         print(df2[df2.astype(str).agg("_".join, axis=1).isin(missing_in_1)])
#     if missing_in_2:
#         print(f"{file2} 缺少的行:")
#         print(df1[df1.astype(str).agg("_".join, axis=1).isin(missing_in_2)])

# # 用法示例
# compare_csv_rows(
#     "../data/data_mink_main_ticks.csv",
#     "../data/data_mink_main_ticks_adjusted.csv"
# )

data_folder = Path("../data/data_mink_product_2025_main")
file_names: Sequence[str] = [f.stem for f in data_folder.iterdir() if f.is_file() and f.suffix == ".csv"]
file_names = "A"

print("Start of data loading ...")
MinkDataProvider = CSVDataProvider(
    path="../data/data_mink_main_ticks_adjusted.csv", 
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
    end_date=datetime(2025, 11, 30),
    frequency=Frequency.MIN_1,
    dateformat="%Y-%m-%d %H:%M:%S",
    ticker_col="product_id"
)
print("End of data loading.")

session_builder.set_data_provider(MinkDataProvider)
session_builder.set_market_open_and_close_time({"hour": 21, "minute": 0}, {"hour": 15, "minute": 0})
ts = session_builder.build(datetime(2025, 1, 10, 9, 1), datetime(2025, 5, 30, 15, 0))

# # l1 = MinkDataProvider.historical_price(MinkTicker("RS"), PriceField.Close, 85, datetime(2025, 7, 25, 10, 37))
# l1 = MinkDataProvider.get_history(MinkTicker("RS"), ["close_price", "DATA_ISSUE", "DATA_ISSUE_SOLUTION"], datetime(2025, 7, 24, 14, 59), datetime(2025, 7, 25, 10, 37))

# # l2 = MinkDataProvider.historical_price(MinkTicker("WR"), PriceField.Close, 5, datetime(2025, 7, 25, 9, 2))
# l2 = MinkDataProvider.get_history(MinkTicker("WR"), ["close_price", "DATA_ISSUE", "DATA_ISSUE_SOLUTION"], datetime(2025, 7, 24, 14, 59), datetime(2025, 7, 25, 9, 2))

# print(l1, l1.dtypes)
# print(l2, l2.dtypes)
        
class SimpleMAStrategy(AbstractStrategy):
    """
    Strategy which computes two simple moving averages (long - 20 minutes, short - 5 minutes)
    between 10:00 and 14:00, and creates a buy order in case if the short moving average is
    greater or equal to the long moving average.
    """
    def __init__(self, ts: BacktestTradingSession, ticker: Ticker):
        super().__init__(ts)
        self.broker = ts.broker
        self.order_factory = ts.order_factory
        self.data_provider = ts.data_provider
        self.ticker = ticker

    def calculate_and_place_orders(self):
        # Compute the moving averages
        long_ma_len = 20
        short_ma_len = 5
        
        # Use data handler to download last 20 minutes close prices and use them to compute the moving averages
        try:
            if hasattr(self.data_provider, 'historical_price'):
                long_ma_series = getattr(self.data_provider, 
                                         'historical_price')(self.ticker, 
                                                             PriceField.Close, 
                                                             long_ma_len, 
                                                             frequency=Frequency.MIN_1)
            else:
                raise AttributeError("Data provider does not support 'historical_price' method.")
        except ValueError:
            return  # Not enough data to compute the long MA yet
        
        long_ma_price = long_ma_series.mean()
        short_ma_series = long_ma_series.tail(short_ma_len)
        short_ma_price = short_ma_series.mean()

        if short_ma_price >= long_ma_price:
            # Place a buy Market Order, adjusting the position to a value equal to 100% of the portfolio
            orders = self.order_factory.target_percent_orders({self.ticker: 1.0},
                MarketOrder(), TimeInForce.DAY)
        else:
            orders = self.order_factory.target_percent_orders({self.ticker: 0.0},
                MarketOrder(), TimeInForce.DAY)

        # Cancel any open orders and place the newly created ones
        self.broker.cancel_all_open_orders()
        self.broker.place_orders(orders)

strategy = SimpleMAStrategy(ts, ticker)
CalculateAndPlaceOrdersPeriodicEvent.set_frequency(Frequency.MIN_1)
CalculateAndPlaceOrdersPeriodicEvent.set_start_and_end_time(
    {"hour": 10, "minute": 0},
    {"hour": 14, "minute": 0})
strategy.subscribe(CalculateAndPlaceOrdersPeriodicEvent)

ts.start_trading()
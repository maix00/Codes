from qf_lib.common.enums.frequency import Frequency
from qf_lib.common.enums.price_field import PriceField
from qf_lib.common.tickers.tickers import Ticker
from qf_lib.backtesting.strategies.abstract_strategy import AbstractStrategy
from qf_lib.backtesting.order.execution_style import MarketOrder
from qf_lib.backtesting.order.time_in_force import TimeInForce
from qf_lib.backtesting.trading_session.backtest_trading_session import BacktestTradingSession

def get_price_series(data_provider, ticker: Ticker, price_field: PriceField, length: int, frequency: Frequency):
    """
    Helper function to get historical price series from the data provider.
    """
    try:
        if hasattr(data_provider, 'historical_price'):
            return getattr(data_provider, 'historical_price')(ticker, price_field, length, frequency=frequency)
        else:
            raise AttributeError("Data provider does not support 'historical_price' method.")
    except ValueError:
        return None  # Not enough data to compute the series yet
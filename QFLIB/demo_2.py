import os
import warnings
import pandas as pd
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=pd.errors.DtypeWarning)

from datetime import datetime
from pathlib import Path
from typing import Sequence, Union, List, Dict, Optional

from qf_lib.common.enums.frequency import Frequency
from qf_lib.common.enums.price_field import PriceField
from qf_lib.common.tickers.tickers import Ticker
from qf_lib.containers.qf_data_array import QFDataArray
from qf_lib.common.utils.logging.qf_parent_logger import qf_logger
from qf_lib.common.utils.miscellaneous.to_list_conversion import convert_to_list
from qf_lib.containers.dataframe.qf_dataframe import QFDataFrame
from qf_lib.containers.dimension_names import DATES, TICKERS, FIELDS
from qf_lib.containers.futures.future_tickers.future_ticker import FutureTicker
from qf_lib.data_providers.helpers import normalize_data_array, tickers_dict_to_data_array
from qf_lib.data_providers.preset_data_provider import PresetDataProvider

import pickle
import gzip
import lzma

class MultipleParquetFuturesDataProvider(PresetDataProvider):
    
    def __init__(self, path: Union[str, Dict[str, str]], tickers: Union[Ticker, Sequence[Ticker]], index_col: str,
                 field_to_price_field_dict: Optional[Dict[str, PriceField]] = None,
                 fields: Optional[Union[str, List[str]]] = None, start_date: Optional[datetime] = None,
                 end_date: Optional[datetime] = None, frequency: Optional[Frequency] = Frequency.DAILY,
                 dateformat: Optional[str] = None, ticker_col: Optional[str] = None, 
                 qf_cache_path: Optional[str] = None, save_cache_mode: bool = False):

        self.logger = qf_logger.getChild(self.__class__.__name__)

        if fields and isinstance(fields, str):
            fields = [fields]

        # Convert to list and remove duplicates
        tickers, _ = convert_to_list(tickers, Ticker)
        tickers = list(dict.fromkeys(tickers))
        assert len([t for t in tickers if isinstance(t, FutureTicker)]) == 0, "FutureTickers are not supported by " \
                                                                              "this data provider"
        
        assert start_date is not None and end_date is not None and frequency is not None, \
            "When loading from cache, start_date, end_date and frequency must be provided."

        normalize_data_array_dict = {}
        for ticker in tickers:
            normalized_data_array = None
            if qf_cache_path is not None:
                normalized_data_array = self.load_compressed(qf_cache_path, ticker_str=ticker.as_string())
            if normalized_data_array is not None and normalized_data_array.size != 0:
                self.logger.info(f"Cache for ticker {ticker.as_string()} found.")
                normalize_data_array_dict[ticker.as_string()] = normalized_data_array
            else:
                self.logger.info(f"Cache for ticker {ticker.as_string()} not found.")
                _path = path[ticker.as_string()] if isinstance(path, dict) else path
                data_array, start_date, end_date, available_fields = self._get_data(_path, tickers, fields, start_date, end_date,
                                                                                frequency, field_to_price_field_dict,
                                                                                index_col, dateformat, ticker_col)
                if data_array is None:
                    self.logger.warning(f"No data found for ticker {ticker.as_string()}. Skipping.")
                    continue
                normalized_data_array = normalize_data_array(data_array, tickers, available_fields, False, False, False)
                normalize_data_array_dict[ticker.as_string()] = normalized_data_array

                if save_cache_mode and qf_cache_path is not None:
                    self.save_compressed(normalized_data_array, qf_cache_path, ticker_str=ticker.as_string())
                    self.logger.info(f"Cache for ticker {ticker.as_string()} saved.")

        if not normalize_data_array_dict:
            if save_cache_mode:
                self.logger.warning("No data was loaded for any ticker. No cache files were created.")
                return
            else:
                self.logger.error("No data was loaded for any ticker. Check the correctness of all data paths.")
        normalized_data_array = QFDataArray.concat(list(normalize_data_array_dict.values()), dim=TICKERS)
        assert start_date is not None and end_date is not None and frequency is not None, \
            "Data loading failed to provide start_date, end_date or frequency."
        super().__init__(data=normalized_data_array,
                         start_date=start_date,
                         end_date=end_date,
                         frequency=frequency)

    @staticmethod
    def save_compressed(obj, filepath, ticker_str: Optional[str] = None, use_lzma: bool = False):
        """
        Saves and compresses any Python object (e.g., QFDataFrame) to a file.
        Args:
            obj: The object to save.
            filepath (str): Path for the output file (.pkl.gz or .xz).
            use_lzma (bool): If True, uses LZMA for higher compression.
        """
        if ticker_str is not None:
            filepath = os.path.join(filepath, ticker_str + ('.xz' if use_lzma else '.gz'))
        else:
            assert filepath.endswith('.xz') if use_lzma else filepath.endswith('.gz')
        open_func = lzma.open if use_lzma else gzip.open
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        with open_func(filepath, 'wb') as f:
            assert isinstance(f, gzip.GzipFile) or isinstance(f, lzma.LZMAFile)
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        size = os.path.getsize(filepath)
        print(f"{filepath} Size: {size}")

    @staticmethod
    def load_compressed(filepath, ticker_str: Optional[str] = None, use_lzma: bool = False):
        """
        Loads an object from a compressed file.
        """
        # Detect compression from file extension
        if ticker_str is not None:
            filepath = os.path.join(filepath, ticker_str + ('.xz' if use_lzma else '.gz'))
        else:
            assert filepath.endswith('.xz') if use_lzma else filepath.endswith('.gz')
        if not Path(filepath).exists():
            return None
        open_func = lzma.open if use_lzma else gzip.open
        with open_func(filepath, 'rb') as f:
            assert isinstance(f, gzip.GzipFile) or isinstance(f, lzma.LZMAFile)
            return pickle.load(f)

    def _get_data(self, path: str, tickers: Sequence[Ticker], fields: Optional[Sequence[str]], start_date: Optional[datetime],
                  end_date: Optional[datetime], frequency: Frequency, field_to_price_field_dict: Optional[Dict[str, PriceField]],
                  index_col: str, dateformat: Optional[str], ticker_col: Optional[str]):

        tickers_str_mapping = {ticker.as_string(): ticker for ticker in tickers}
        tickers_prices_dict = {}
        available_fields = set()

        def _process_df(df, ticker_str):
            df.index = pd.to_datetime(df[index_col], format=dateformat)
            df = df[~df.index.duplicated(keep='first')]
            df = df.drop(index_col, axis=1)
            if Frequency.infer_freq(df.index) != frequency:
                self.logger.info(f"Inferred frequency for the file {path} is different than requested. "
                                 f"Skipping {path}.")
            else:

                start_time = start_date or df.index[0]
                end_time = end_date or df.index[-1]

                if fields:
                    df = df.loc[start_time:end_time, df.columns.isin(fields)]
                    fields_diff = set(fields).difference(df.columns)
                    if fields_diff:
                        self.logger.info(f"Not all fields are available for {path}. Difference: {fields_diff}")
                else:
                    df = df.loc[start_time:end_time, :]
                    available_fields.update(df.columns.tolist())

                if field_to_price_field_dict:
                    for key, value in field_to_price_field_dict.items():
                        df[value] = df[key]

                if ticker_str in tickers_str_mapping:
                    tickers_prices_dict[tickers_str_mapping[ticker_str]] = df
                else:
                    self.logger.info(f'Ticker {ticker_str} was not requested in the list of tickers. Skipping.')

        if ticker_col:
            df = QFDataFrame(pd.read_parquet(path))#, dtype={index_col: str}))
            if df.empty:
                return None, None, None, None
            available_tickers = df[ticker_col].dropna().unique().tolist()

            for ticker_str in available_tickers:
                sliced_df = df[df[ticker_col] == ticker_str]
                _process_df(sliced_df, ticker_str)

        else:
            tickers_paths = [list(Path(path).glob('**/{}.parquet'.format(ticker.as_string()))) for ticker in tickers]
            joined_tickers_paths = [item for sublist in tickers_paths for item in sublist]

            for _path in joined_tickers_paths:
                ticker_str = _path.resolve().name.replace('.parquet', '')
                df = QFDataFrame(pd.read_parquet(_path))#, dtype={index_col: str}))
                _process_df(df, ticker_str)

        if not tickers_prices_dict.values():
            raise ImportError("No data was found. Check the correctness of all data")

        if fields:
            available_fields = set(fields)
        else:
            available_fields = set(available_fields)

        if field_to_price_field_dict:
            available_fields = available_fields.union(set(field_to_price_field_dict.values()))

        if not start_date:
            start_date = min(list(df.index.min() for df in tickers_prices_dict.values()))

        if not end_date:
            end_date = max(list(df.index.max() for df in tickers_prices_dict.values()))

        result = tickers_dict_to_data_array(tickers_prices_dict, list(tickers_prices_dict.keys()), list(available_fields)), \
            start_date, end_date, list(available_fields)
        return result

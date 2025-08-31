from typing import Dict, Union, Optional
import pandas as pd

from core.loader import DataLoader
from core.models import Symbol, TimeFrame, TimeRange, DataType
from backtest.logger import get_logger


class MultiTimeframeData:
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._data: Dict[str, pd.DataFrame] = {}
        self._symbol_data: Dict[str, Dict[str, pd.DataFrame]] = {}
        self._logger = get_logger(__name__)
    
    def add(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        time_range: Optional[TimeRange] = None,
        force_download: bool = False
    ) -> 'MultiTimeframeData':
        try:
            self._logger.info(f"Loading {timeframe.value} data for {symbol}")
            data = self._loader.load(symbol, timeframe, DataType.OHLCV, time_range, force_download)
            
            self._data[timeframe.value] = data
            
            symbol_str = str(symbol)
            if symbol_str not in self._symbol_data:
                self._symbol_data[symbol_str] = {}
            self._symbol_data[symbol_str][timeframe.value] = data
            
            self._logger.info(f"Successfully loaded {len(data)} bars of {timeframe.value} data")
            return self
        except Exception as e:
            self._logger.warning(f"Failed to load {timeframe.value} data: {e}")
            return self
    
    def __contains__(self, timeframe: Union[str, TimeFrame]) -> bool:
        tf_key = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        return tf_key in self._data
    
    def __getitem__(self, timeframe: Union[str, TimeFrame]) -> pd.DataFrame:
        tf_key = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        if tf_key not in self._data:
            raise KeyError(f"Timeframe '{tf_key}' not found")
        return self._data[tf_key]
    
    def get_symbol_data(self, symbol: str, timeframe: Union[str, TimeFrame]) -> Optional[pd.DataFrame]:
        tf_key = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        if symbol in self._symbol_data and tf_key in self._symbol_data[symbol]:
            return self._symbol_data[symbol][tf_key]
        return None
    
    def get_all_symbols(self) -> list:
        return list(self._symbol_data.keys())
    
    def has_symbol(self, symbol: str, timeframe: Union[str, TimeFrame] = None) -> bool:
        if symbol not in self._symbol_data:
            return False
        if timeframe is None:
            return True
        tf_key = timeframe.value if isinstance(timeframe, TimeFrame) else timeframe
        return tf_key in self._symbol_data[symbol]
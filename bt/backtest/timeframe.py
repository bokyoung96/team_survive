from typing import Dict, Union, Optional
import pandas as pd

from core.loader import DataLoader
from core.models import Symbol, TimeFrame, TimeRange, DataType
from backtest.logger import get_logger


class MultiTimeframeData:
    def __init__(self, loader: DataLoader):
        self._loader = loader
        self._data: Dict[str, pd.DataFrame] = {}
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
import pandas as pd
from pathlib import Path
from typing import List, Dict

from loader import DataLoader
from factory import Factory
from models import Symbol, TimeFrame, TimeRange, DataType, Exchange
from protocols import TechnicalIndicator
from ti import IndicatorProcessor


class Analyzer:
    def __init__(self, exchange: Exchange, symbol: Symbol, timeframe: TimeFrame, time_range: TimeRange):
        self._exchange = exchange
        self._symbol = symbol
        self._timeframe = timeframe
        self._time_range = time_range

        factory = Factory(exchange=self._exchange, base_path=Path("./data"))
        self._loader = DataLoader(factory=factory)
        self._processor = IndicatorProcessor()

    def add_indicators(self, indicators: List[TechnicalIndicator]):
        for indicator in indicators:
            self._processor.add_indicator(indicator)

    def run(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        ohlcv = self._loader.load(
            symbol=self._symbol,
            timeframe=self._timeframe,
            data_type=DataType.OHLCV,
            time_range=self._time_range,
            force_download=force_download
        )

        if ohlcv.empty:
            print("No data loaded. Aborting analysis.")
            return {}

        res = self._processor.process(ohlcv)
        return res

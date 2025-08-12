from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import List, Dict

from core import DataLoader, Factory
from core.models import Symbol, TimeFrame, TimeRange, DataType, Exchange, MarketType
from core.protocols import TechnicalIndicator
from indicators import (
    IndicatorProcessor,
    IchimokuCloud,
    MovingAverage,
    RSI,
    MACD,
)


class Analyzer:
    def __init__(self, symbol: Symbol, timeframe: TimeFrame, time_range: TimeRange):
        self._symbol = symbol
        self._timeframe = timeframe
        self._time_range = time_range

        exchange = Exchange(id="binance", default_type=MarketType.SWAP)
        factory = Factory(exchange=exchange, base_path=Path("./fetch"))
        self._loader = DataLoader(factory=factory)
        self._processor = IndicatorProcessor()

    def add_indicators(self, indicators: List[TechnicalIndicator]) -> None:
        for indicator in indicators:
            self._processor.add_indicator(indicator)

    def run(self, force_download: bool = False) -> Dict[str, pd.DataFrame]:
        ohlcv = self._loader.load(
            symbol=self._symbol,
            timeframe=self._timeframe,
            data_type=DataType.OHLCV,
            time_range=self._time_range,
            force_download=force_download,
        )

        if ohlcv.empty:
            return {}

        result = self._processor.process(ohlcv)
        return result

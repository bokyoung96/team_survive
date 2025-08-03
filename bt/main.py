import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import List, Dict

from loader import DataLoader
from factory import Factory
from models import Symbol, TimeFrame, TimeRange, DataType, Exchange, MarketType
from tools import parse_symbol
from protocols import TechnicalIndicator
from ti import (
    IndicatorProcessor,
    IchimokuCloud,
    MovingAverage,
    RSI,
    MACD
)


class Analyzer:
    def __init__(self, symbol: Symbol, timeframe: TimeFrame, time_range: TimeRange):
        self._symbol = symbol
        self._timeframe = timeframe
        self._time_range = time_range

        exchange = Exchange(id="binance", default_type=MarketType.SWAP)
        factory = Factory(exchange=exchange, base_path=Path("./data"))
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

        result = self._processor.process(ohlcv)
        return result


def main():
    analyzer = Analyzer(
        symbol=parse_symbol("BTC/USDT:USDT"),
        timeframe=TimeFrame.M5,
        time_range=TimeRange(start=datetime(2025, 1, 1),
                             end=datetime(2025, 8, 3))
    )

    analyzer.add_indicators([
        IchimokuCloud(name='ichimoku'),
        MovingAverage(name='ma_50', length=50),
        RSI(name='rsi_14', length=14)
    ])

    res = analyzer.run(force_download=False)
    return res


if __name__ == "__main__":
    df = main()

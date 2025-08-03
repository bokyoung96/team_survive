from __future__ import annotations
import pandas as pd
from typing import Optional
from pathlib import Path
from datetime import datetime

from models import Symbol, TimeFrame, TimeRange, DataType, Exchange, MarketType
from factory import Factory
from tools import parse_symbol


class DataLoader:
    def __init__(self, factory: Factory):
        self._factory = factory
        self._repository = self._factory.create_repository()

    def load(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        data_type: DataType,
        time_range: Optional[TimeRange] = None,
    ) -> pd.DataFrame:
        data = self._repository.load(symbol, timeframe, data_type)
        if data is not None and self._is_data_sufficient(data, time_range):
            return data

        fetcher = self._factory.create_fetcher(data_type)
        fetched_data = fetcher.fetch(symbol, timeframe, time_range)
        self._repository.save(fetched_data, symbol, timeframe, data_type)
        return fetched_data

    def _is_data_sufficient(
        self, data: pd.DataFrame, time_range: Optional[TimeRange]
    ) -> bool:
        if time_range is None:
            return True
        return (
            not data.empty
            and data["timestamp"].min() <= time_range.start
            and data["timestamp"].max() >= time_range.end
        )


if __name__ == "__main__":
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    factory = Factory(exchange=exchange, base_path=Path("./data"))
    loader = DataLoader(factory=factory)

    symbol = parse_symbol("BTC/USDT:USDT")
    time_range = TimeRange(start=datetime(2023, 1, 1),
                           end=datetime(2025, 8, 3))

    ohlcv_data = loader.load(
        symbol=symbol,
        timeframe=TimeFrame.M5,
        data_type=DataType.OHLCV,
        time_range=time_range,
    )

    print(f"Loaded OHLCV data shape: {ohlcv_data.head()}")

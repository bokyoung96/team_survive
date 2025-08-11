from __future__ import annotations
import pandas as pd
from typing import Protocol, Optional

from models import Symbol, TimeFrame, TimeRange, DataType


class DataFetcher(Protocol):
    def fetch(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        time_range: Optional[TimeRange] = None,
    ) -> pd.DataFrame:
        ...


class Repository(Protocol):
    def load(self, symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> Optional[pd.DataFrame]:
        ...

    def save(self, data: pd.DataFrame, symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> None:
        ...


class TechnicalIndicator(Protocol):
    name: str

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        ...

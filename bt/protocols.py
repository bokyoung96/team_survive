import pandas as pd
from typing import Protocol, Optional

from models import Symbol, TimeFrame, TimeRange


class DataFetcher(Protocol):
    def fetch(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        time_range: Optional[TimeRange] = None,
    ) -> pd.DataFrame:
        ...


class Repository(Protocol):
    def load(self, symbol: Symbol, timeframe: TimeFrame) -> Optional[pd.DataFrame]:
        ...

    def save(self, data: pd.DataFrame, symbol: Symbol, timeframe: TimeFrame) -> None:
        ...

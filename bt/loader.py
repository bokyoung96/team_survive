from __future__ import annotations
import pandas as pd
from typing import Optional

from models import Symbol, TimeFrame, TimeRange, DataType
from factory import Factory


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
        force_download: bool = False,
    ) -> pd.DataFrame:
        if not force_download:
            data = self._repository.load(symbol, timeframe, data_type)
            if data is not None:
                saved_data = self._set_dt_idx(data)
                if self._is_data_sufficient(saved_data, time_range):
                    return saved_data

        fetcher = self._factory.create_fetcher(data_type)
        fetched_data = fetcher.fetch(symbol, timeframe, time_range)

        if fetched_data.empty:
            return fetched_data

        self._repository.save(fetched_data, symbol, timeframe, data_type)
        return self._set_dt_idx(fetched_data)

    def _set_dt_idx(self, data: pd.DataFrame) -> pd.DataFrame:
        if "timestamp" not in data.columns:
            return data

        df = data.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("timestamp")
        return df

    def _is_data_sufficient(
        self, data: pd.DataFrame, time_range: Optional[TimeRange]
    ) -> bool:
        if time_range is None:
            return True
        return (
            not data.empty
            and data.index.min() <= time_range.start
            and data.index.max() >= time_range.end
        )

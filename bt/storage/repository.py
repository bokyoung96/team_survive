from __future__ import annotations
import pandas as pd
from pathlib import Path
from typing import Optional

from core.models import Symbol, TimeFrame, Exchange, DataType
from core.protocols import Repository


class RepositoryManager(Repository):
    def __init__(self, base_path: Path, exchange: Exchange):
        self._base_path = base_path
        self._exchange = exchange

    def load(self, symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> Optional[pd.DataFrame]:
        file_path = self._get_path(symbol, timeframe, data_type)
        if file_path.exists():
            return pd.read_parquet(file_path)
        return None

    def save(self, data: pd.DataFrame, symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> None:
        file_path = self._get_path(symbol, timeframe, data_type)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        data.to_parquet(file_path)

    def _get_path(self, symbol: Symbol, timeframe: TimeFrame, data_type: DataType) -> Path:
        return (
            self._base_path
            / self._exchange.id
            / data_type.value
            / f"{symbol.base}_{symbol.quote}"
            / f"{timeframe.value}.parquet"
        )

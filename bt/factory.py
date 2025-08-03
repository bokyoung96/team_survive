from __future__ import annotations
from pathlib import Path

from models import Exchange, DataType
from protocols import DataFetcher, Repository
from fetchers import OHLCVFetcher
from repository import RegistryOrchestrator


class Factory:
    def __init__(self, exchange: Exchange, base_path: Path):
        self._exchange = exchange
        self._base_path = base_path

    def create_fetcher(self, data_type: DataType) -> DataFetcher:
        if data_type == DataType.OHLCV:
            return OHLCVFetcher(self._exchange)
        raise ValueError(f"Unsupported data type: {data_type}")

    def create_repository(self) -> Repository:
        return RegistryOrchestrator(self._base_path, self._exchange)

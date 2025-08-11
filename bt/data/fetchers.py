from __future__ import annotations
import logging
import pandas as pd
from tqdm import tqdm
from typing import Optional, List, Any

from core.models import Exchange, Symbol, TimeFrame, TimeRange, MarketType
from core.protocols import DataFetcher
from utils.tools import convert_ts_to_dt


class OHLCVFetcher(DataFetcher):
    def __init__(self, exchange: Exchange, market_type: MarketType = MarketType.SWAP):
        self._exchange = exchange
        self._market_type = market_type
        self._client = self._exchange.client()

    def fetch(
        self,
        symbol: Symbol,
        timeframe: TimeFrame,
        time_range: Optional[TimeRange] = None,
    ) -> pd.DataFrame:
        symbol_str = symbol.to_string(self._market_type)
        if time_range:
            return self._fetch_multi(symbol_str, timeframe, time_range)
        return self._fetch_once(symbol_str, timeframe)

    def _fetch_once(self, symbol_str: str, timeframe: TimeFrame) -> pd.DataFrame:
        ohlcv = self._client.fetch_ohlcv(symbol_str, timeframe.value)
        return self._to_df(ohlcv)

    def _fetch_multi(
        self, symbol_str: str, timeframe: TimeFrame, time_range: TimeRange
    ) -> pd.DataFrame:
        start_ts = int(time_range.start.timestamp() * 1000)
        end_ts = int(time_range.end.timestamp() * 1000)
        all_ohlcv: List[List[Any]] = []

        with tqdm(desc=f"Fetching {symbol_str}") as pbar:
            since = start_ts
            while since < end_ts:
                ohlcv = self._client.fetch_ohlcv(
                    symbol_str, timeframe.value, since=since
                )
                if not ohlcv:
                    break

                all_ohlcv.extend(ohlcv)
                last_ts = ohlcv[-1][0]
                if last_ts >= since:
                    since = last_ts + 1
                else:
                    break
                pbar.update(len(ohlcv))
                logging.info(
                    f"Fetched {len(ohlcv)} rows, last timestamp: {last_ts}")

        return self._to_df(all_ohlcv)

    def _to_df(self, ohlcv: List[List[Any]]) -> pd.DataFrame:
        if not ohlcv:
            return pd.DataFrame()
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df["timestamp"] = df["timestamp"].apply(convert_ts_to_dt)
        return df

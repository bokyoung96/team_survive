import pandas as pd
import pandas_ta as ta
from typing import List, Dict

from core.protocols import TechnicalIndicator


class IchimokuCloud:
    def __init__(self, name: str, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        self.name = name
        self._tenkan = tenkan
        self._kijun = kijun
        self._senkou = senkou

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        ichimoku_df = ta.ichimoku(
            high=ohlcv["high"],
            low=ohlcv["low"],
            close=ohlcv["close"],
            tenkan=self._tenkan,
            kijun=self._kijun,
            senkou=self._senkou,
        )[0]

        col_map = {
            f"ITS_{self._tenkan}": f"{self.name}_conversion_line",
            f"IKS_{self._kijun}": f"{self.name}_base_line",
            f"ISA_{self._tenkan}": f"{self.name}_leading_span_a",
            f"ISB_{self._kijun}": f"{self.name}_leading_span_b",
            f"ICS_{self._kijun}": f"{self.name}_lagging_span",
        }

        found_cols = {k: v for k, v in col_map.items()
                      if k in ichimoku_df.columns}
        return ichimoku_df[list(found_cols.keys())].rename(columns=found_cols)


class VolumeProfile:
    def __init__(self, name: str, bins: int = 20):
        self.name = name
        self._bins = bins

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        bins = pd.cut(ohlcv["close"], bins=self._bins,
                      labels=False, right=False)
        vp = ohlcv.groupby(bins)["volume"].sum()

        poc_level = vp.idxmax()
        poc_price_range = ohlcv["close"][bins == poc_level].agg(["min", "max"])
        poc = (poc_price_range["min"] + poc_price_range["max"]) / 2

        df = pd.DataFrame(index=ohlcv.index)
        df[f"{self.name}_poc"] = poc
        return df


class MovingAverage:
    def __init__(self, name: str, length: int = 20):
        self.name = name
        self._length = length

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        ma = ta.sma(ohlcv["close"], length=self._length)
        return pd.DataFrame({self.name: ma})


class RSI:
    def __init__(self, name: str, length: int = 14):
        self.name = name
        self._length = length

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        rsi = ta.rsi(ohlcv["close"], length=self._length)
        return pd.DataFrame({self.name: rsi})


class MACD:
    def __init__(self, name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        self.name = name
        self._fast = fast
        self._slow = slow
        self._signal = signal

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        macd_df = ta.macd(ohlcv["close"], fast=self._fast,
                          slow=self._slow, signal=self._signal)
        column_names = {
            f"MACD_{self._fast}_{self._slow}_{self._signal}": f"{self.name}",
            f"MACDh_{self._fast}_{self._slow}_{self._signal}": f"{self.name}_hist",
            f"MACDs_{self._fast}_{self._slow}_{self._signal}": f"{self.name}_signal",
        }
        return macd_df.rename(columns=column_names)


class IndicatorProcessor:
    def __init__(self):
        self._indicators: List[TechnicalIndicator] = []

    def add_indicator(self, indicator: TechnicalIndicator) -> None:
        self._indicators.append(indicator)

    def process(self, ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        indicator_dfs = [indicator.calculate(
            ohlcv) for indicator in self._indicators]
        ti_df = pd.concat(indicator_dfs, axis=1) if indicator_dfs else pd.DataFrame(
            index=ohlcv.index)
        return {"ohlcv": ohlcv, "indicators": ti_df}

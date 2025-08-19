import numpy as np
import pandas as pd
from typing import List, Dict

from core.protocols import TechnicalIndicator


class IchimokuCloud:
    def __init__(self, name: str, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        self.name = name
        self._tenkan = tenkan
        self._kijun = kijun
        self._senkou = senkou

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        
        # NOTE: Conversion Line (Tenkan-sen)
        tenkan_high = high.rolling(window=self._tenkan).max()
        tenkan_low = low.rolling(window=self._tenkan).min()
        conversion_line = (tenkan_high + tenkan_low) / 2
        
        # NOTE: Base Line (Kijun-sen)
        kijun_high = high.rolling(window=self._kijun).max()
        kijun_low = low.rolling(window=self._kijun).min()
        base_line = (kijun_high + kijun_low) / 2
        
        # NOTE: Leading Span A (Senkou Span A)
        leading_span_a = ((conversion_line + base_line) / 2).shift(self._kijun)
        
        # NOTE: Leading Span B (Senkou Span B)
        senkou_high = high.rolling(window=self._senkou).max()
        senkou_low = low.rolling(window=self._senkou).min()
        leading_span_b = ((senkou_high + senkou_low) / 2).shift(self._kijun)
        
        # NOTE: Lagging Span (Chikou Span)
        lagging_span = close.shift(-self._kijun)
        
        df = pd.DataFrame({
            f"{self.name}_conversion_line": conversion_line,
            f"{self.name}_base_line": base_line,
            f"{self.name}_leading_span_a": leading_span_a,
            f"{self.name}_leading_span_b": leading_span_b,
            f"{self.name}_lagging_span": lagging_span,
        })
        
        return df


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
        ma = ohlcv["close"].rolling(window=self._length).mean()
        return pd.DataFrame({self.name: ma})
    
    def reset(self) -> None:
        """Reset indicator state for new calculations"""
        pass


class RSI:
    def __init__(self, name: str, length: int = 14):
        self.name = name
        self._length = length

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        close = ohlcv["close"]
        delta = close.diff()
        
        gain = (delta.where(delta > 0, 0)).rolling(window=self._length).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self._length).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return pd.DataFrame({self.name: rsi})


class MACD:
    def __init__(self, name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        self.name = name
        self._fast = fast
        self._slow = slow
        self._signal = signal

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        close = ohlcv["close"]
        
        # NOTE: Calculate EMAs
        ema_fast = close.ewm(span=self._fast, adjust=False).mean()
        ema_slow = close.ewm(span=self._slow, adjust=False).mean()
        
        # NOTE: MACD line
        macd = ema_fast - ema_slow
        
        # NOTE: Signal line
        signal = macd.ewm(span=self._signal, adjust=False).mean()
        
        # NOTE: MACD histogram
        histogram = macd - signal
        
        df = pd.DataFrame({
            f"{self.name}": macd,
            f"{self.name}_signal": signal,
            f"{self.name}_hist": histogram,
        })
        
        return df


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
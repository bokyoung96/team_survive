import pandas as pd
from typing import List, Dict

from protocols import TechnicalIndicator


class IchimokuCloud:
    def __init__(self, name: str, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        self.name = name
        self._tenkan = tenkan
        self._kijun = kijun
        self._senkou = senkou

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        high = ohlcv['high']
        low = ohlcv['low']
        close = ohlcv['close']

        tenkan_sen = (high.rolling(window=self._tenkan).max() +
                      low.rolling(window=self._tenkan).min()) / 2
        kijun_sen = (high.rolling(window=self._kijun).max() +
                     low.rolling(window=self._kijun).min()) / 2
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(self._kijun)
        senkou_span_b = ((high.rolling(window=self._senkou).max(
        ) + low.rolling(window=self._senkou).min()) / 2).shift(self._kijun)
        chikou_span = close.shift(-self._kijun)

        df = pd.DataFrame({
            f'{self.name}_conversion_line': tenkan_sen,
            f'{self.name}_base_line': kijun_sen,
            f'{self.name}_leading_span_a': senkou_span_a,
            f'{self.name}_leading_span_b': senkou_span_b,
            f'{self.name}_lagging_span': chikou_span
        })
        return df


class VolumeProfile:
    def __init__(self, name: str, bins: int = 20):
        self.name = name
        self._bins = bins

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        bins = pd.cut(ohlcv['close'], bins=self._bins,
                      labels=False, right=False)
        vp = ohlcv.groupby(bins)['volume'].sum()

        poc_level = vp.idxmax()
        poc_price_range = ohlcv['close'][bins == poc_level].agg(['min', 'max'])
        poc = (poc_price_range['min'] + poc_price_range['max']) / 2

        df = pd.DataFrame(index=ohlcv.index)
        df[f'{self.name}_poc'] = poc
        return df


class MovingAverage:
    def __init__(self, name: str, length: int = 20):
        self.name = name
        self._length = length

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        ma = ohlcv['close'].rolling(window=self._length).mean()
        return pd.DataFrame({self.name: ma})


class RSI:
    def __init__(self, name: str, length: int = 14):
        self.name = name
        self._length = length

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        close_delta = ohlcv['close'].diff()
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)

        ma_up = up.ewm(com=self._length - 1, adjust=True,
                       min_periods=self._length).mean()
        ma_down = down.ewm(com=self._length - 1, adjust=True,
                           min_periods=self._length).mean()

        rsi = ma_up / ma_down
        rsi = 100 - (100 / (1 + rsi))
        return pd.DataFrame({self.name: rsi})


class MACD:
    def __init__(self, name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        self.name = name
        self._fast = fast
        self._slow = slow
        self._signal = signal

    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        ema_fast = ohlcv['close'].ewm(span=self._fast, adjust=False).mean()
        ema_slow = ohlcv['close'].ewm(span=self._slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self._signal, adjust=False).mean()
        histogram = macd_line - signal_line

        df = pd.DataFrame({
            f'{self.name}': macd_line,
            f'{self.name}_hist': histogram,
            f'{self.name}_signal': signal_line
        })
        return df


class IndicatorProcessor:
    def __init__(self):
        self._indicators: List[TechnicalIndicator] = []

    def add_indicator(self, indicator: TechnicalIndicator):
        self._indicators.append(indicator)

    def process(self, ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        indicator_dfs = [indicator.calculate(
            ohlcv) for indicator in self._indicators]

        ti_df = pd.concat(indicator_dfs, axis=1)

        return {
            "ohlcv": ohlcv,
            "indicators": ti_df
        }

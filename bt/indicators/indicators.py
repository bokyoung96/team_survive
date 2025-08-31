import numpy as np
import pandas as pd
import numba as nb
import polars as pl
from typing import List, Dict, Optional, Union
from abc import ABC, abstractmethod
from collections import deque
import numpy as np
from core.protocols import TechnicalIndicator


class BaseIndicator(ABC):
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        pass
    
    def calculate(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        return self._compute(ohlcv)
    
    def reset(self) -> None:
        pass


class RollingWindowIndicator(BaseIndicator):    
    def __init__(self, name: str, window: int):
        super().__init__(name)
        self.window = window


class IchimokuCloud(BaseIndicator):
    def __init__(self, name: str, tenkan: int = 9, kijun: int = 26, senkou: int = 52):
        super().__init__(name)
        self._tenkan = tenkan
        self._kijun = kijun
        self._senkou = senkou

    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        high = ohlcv["high"]
        low = ohlcv["low"]
        close = ohlcv["close"]
        
        tenkan_high = high.rolling(window=self._tenkan).max()
        tenkan_low = low.rolling(window=self._tenkan).min()
        conversion_line = (tenkan_high + tenkan_low) / 2
        
        kijun_high = high.rolling(window=self._kijun).max()
        kijun_low = low.rolling(window=self._kijun).min()
        base_line = (kijun_high + kijun_low) / 2
        
        leading_span_a = ((conversion_line + base_line) / 2).shift(self._kijun)
        
        senkou_high = high.rolling(window=self._senkou).max()
        senkou_low = low.rolling(window=self._senkou).min()
        leading_span_b = ((senkou_high + senkou_low) / 2).shift(self._kijun)
        
        lagging_span = close.shift(-self._kijun)
        
        return pd.DataFrame({
            f"{self.name}_conversion_line": conversion_line,
            f"{self.name}_base_line": base_line,
            f"{self.name}_leading_span_a": leading_span_a,
            f"{self.name}_leading_span_b": leading_span_b,
            f"{self.name}_lagging_span": lagging_span,
        })


class VolumeProfile(BaseIndicator):
    def __init__(self, name: str, bins: int = 20):
        super().__init__(name)
        self._bins = bins

    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        temp = pl.from_pandas(ohlcv[["close", "volume"]])
        
        close_min = temp["close"].min()
        close_max = temp["close"].max()
        close_range = close_max - close_min
        
        if close_range == 0:
            poc = close_min
        else:
            df_with_bins = temp.with_columns(
                ((pl.col("close") - close_min) / close_range * self._bins).floor().alias("bin")
            )
            
            volume_by_bin = df_with_bins.group_by("bin").agg(
                pl.col("volume").sum().alias("total_volume"),
                pl.col("close").mean().alias("avg_price")
            )
            
            poc_row = volume_by_bin.sort("total_volume", descending=True).head(1)
            poc = poc_row["avg_price"][0] if len(poc_row) > 0 else ohlcv["close"].median()
        
        df = pd.DataFrame(index=ohlcv.index)
        df[f"{self.name}_poc"] = poc
        return df


class SMA(RollingWindowIndicator):
    def __init__(self, name: str, length: int = 20):
        super().__init__(name, length)
        self._length = length
        self._buffer = deque(maxlen=length)
        self._sum = 0.0
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        temp = pl.from_pandas(ohlcv[["close"]])
        ma = temp.select(
            pl.col("close").rolling_mean(window_size=self._length).alias(self.name)
        ).to_pandas()
        ma.index = ohlcv.index
        return ma
    
    def update(self, price: float) -> Optional[float]:
        old_len = len(self._buffer)
        
        if old_len == self._length:
            self._sum -= self._buffer[0]
        
        self._buffer.append(price)
        self._sum += price
        
        if len(self._buffer) >= self._length:
            return self._sum / self._length
        elif len(self._buffer) > 0:
            return self._sum / len(self._buffer)
        return None
    
    def get_current(self) -> Optional[float]:
        if len(self._buffer) >= self._length:
            return self._sum / self._length
        elif len(self._buffer) > 0:
            return self._sum / len(self._buffer)
        return None
    
    def reset(self) -> None:
        super().reset()
        self._buffer.clear()
        self._sum = 0.0


class EMA(RollingWindowIndicator):
    def __init__(self, name: str, length: int = 20):
        super().__init__(name, length)
        self._length = length
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        @nb.jit(nopython=True, cache=True)
        def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
            n = len(prices)
            ema = np.empty(n)
            ema[:period-1] = np.nan
            
            if n < period:
                return ema
            
            ema[period-1] = np.mean(prices[:period])
            multiplier = 2.0 / (period + 1)
            
            for i in range(period, n):
                ema[i] = prices[i] * multiplier + ema[i-1] * (1 - multiplier)
            
            return ema
        
        prices = ohlcv["close"].values.astype(np.float64)
        ema_values = calculate_ema(prices, self._length)
        return pd.DataFrame({self.name: ema_values}, index=ohlcv.index)


class RSI(RollingWindowIndicator):
    def __init__(self, name: str, length: int = 14):
        super().__init__(name, length)
        self._length = length
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        @nb.jit(nopython=True, cache=True)
        def calculate_rsi(prices: np.ndarray, period: int) -> np.ndarray:
            n = len(prices)
            rsi = np.empty(n)
            rsi[:period] = np.nan
            
            if n < period + 1:
                return rsi
            
            deltas = np.diff(prices[:period+1])
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            
            avg_gain = np.mean(gains)
            avg_loss = np.mean(losses)
            
            if avg_loss == 0:
                rsi[period] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi[period] = 100 - (100 / (1 + rs))
            
            for i in range(period + 1, n):
                delta = prices[i] - prices[i-1]
                gain = delta if delta > 0 else 0.0
                loss = -delta if delta < 0 else 0.0
                
                avg_gain = (avg_gain * (period - 1) + gain) / period
                avg_loss = (avg_loss * (period - 1) + loss) / period
                
                if avg_loss == 0:
                    rsi[i] = 100.0
                else:
                    rs = avg_gain / avg_loss
                    rsi[i] = 100 - (100 / (1 + rs))
            
            return rsi
        
        prices = ohlcv["close"].values.astype(np.float64)
        rsi_values = calculate_rsi(prices, self._length)
        return pd.DataFrame({self.name: rsi_values}, index=ohlcv.index)


class MACD(BaseIndicator):
    def __init__(self, name: str, fast: int = 12, slow: int = 26, signal: int = 9):
        super().__init__(name)
        self._fast = fast
        self._slow = slow
        self._signal = signal

    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        @nb.jit(nopython=True, cache=True)
        def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
            n = len(prices)
            ema = np.empty(n)
            ema[:period-1] = np.nan
            
            if n < period:
                return ema
            
            ema[period-1] = np.mean(prices[:period])
            multiplier = 2.0 / (period + 1)
            
            for i in range(period, n):
                ema[i] = prices[i] * multiplier + ema[i-1] * (1 - multiplier)
            
            return ema
        
        prices = ohlcv["close"].values.astype(np.float64)
        
        ema_fast = calculate_ema(prices, self._fast)
        ema_slow = calculate_ema(prices, self._slow)
        
        macd = ema_fast - ema_slow
        
        macd_clean = macd[~np.isnan(macd)]
        signal_line = calculate_ema(macd_clean, self._signal)
        
        signal_aligned = np.full(len(prices), np.nan)
        valid_start = self._slow - 1 + self._signal - 1
        if valid_start < len(prices):
            signal_aligned[valid_start:] = signal_line[:len(prices)-valid_start]
        
        histogram = np.where(
            np.isnan(macd) | np.isnan(signal_aligned),
            np.nan,
            macd - signal_aligned
        )
        
        return pd.DataFrame({
            f"{self.name}": macd,
            f"{self.name}_signal": signal_aligned,
            f"{self.name}_hist": histogram,
        }, index=ohlcv.index)


class BollingerBands(RollingWindowIndicator):
    def __init__(self, name: str, length: int = 20, std: float = 2.0):
        super().__init__(name, length)
        self._length = length
        self._std = std
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        @nb.jit(nopython=True, cache=True)
        def calculate_bb(prices: np.ndarray, period: int, num_std: float) -> tuple:
            n = len(prices)
            middle = np.empty(n)
            upper = np.empty(n)
            lower = np.empty(n)
            
            for i in range(n):
                if i < period - 1:
                    middle[i] = np.nan
                    upper[i] = np.nan
                    lower[i] = np.nan
                else:
                    window = prices[i-period+1:i+1]
                    mean = np.mean(window)
                    std = np.std(window)
                    
                    middle[i] = mean
                    upper[i] = mean + std * num_std
                    lower[i] = mean - std * num_std
            
            return middle, upper, lower
        
        prices = ohlcv["close"].values.astype(np.float64)
        middle, upper, lower = calculate_bb(prices, self._length, self._std)
        
        return pd.DataFrame({
            f"{self.name}_middle": middle,
            f"{self.name}_upper": upper,
            f"{self.name}_lower": lower,
            f"{self.name}_width": upper - lower
        }, index=ohlcv.index)


class SupportResistance(BaseIndicator):
    def __init__(self, name: str, lookback: int = 20, min_touches: int = 2, tolerance: float = 0.02):
        super().__init__(name)
        self._lookback = lookback
        self._min_touches = min_touches
        self._tolerance = tolerance
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        temp = pl.from_pandas(ohlcv[["high", "low"]])
        
        resistance = temp.select(
            pl.col("high").rolling_max(window_size=self._lookback).alias("resistance")
        ).to_pandas()["resistance"]
        
        support = temp.select(
            pl.col("low").rolling_min(window_size=self._lookback).alias("support")
        ).to_pandas()["support"]
        
        high = ohlcv["high"]
        low = ohlcv["low"]
        
        close_to_resistance = (high >= resistance * (1 - self._tolerance)).rolling(window=self._lookback).sum()
        close_to_support = (low <= support * (1 + self._tolerance)).rolling(window=self._lookback).sum()
        
        return pd.DataFrame({
            f"{self.name}_resistance": resistance.values,
            f"{self.name}_support": support.values,
            f"{self.name}_resistance_touches": close_to_resistance,
            f"{self.name}_support_touches": close_to_support
        }, index=ohlcv.index)


class FibonacciLevels(BaseIndicator):
    def __init__(self, name: str, lookback: int = 50):
        super().__init__(name)
        self._lookback = lookback
        self._levels = [0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0]
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        temp = pl.from_pandas(ohlcv[["high", "low"]])
        
        high_max = temp.select(
            pl.col("high").rolling_max(window_size=self._lookback).alias("high_max")
        ).to_pandas()["high_max"]
        
        low_min = temp.select(
            pl.col("low").rolling_min(window_size=self._lookback).alias("low_min")
        ).to_pandas()["low_min"]
        
        diff = high_max - low_min
        
        result = pd.DataFrame(index=ohlcv.index)
        for level in self._levels:
            result[f"{self.name}_fib_{int(level*1000)}"] = low_min + (diff * level)

        return result


class MAAnalyzer(BaseIndicator):
    def __init__(self, name: str, periods: List[int]):
        super().__init__(name)
        self._periods = sorted(periods)
    
    def _compute(self, ohlcv: pd.DataFrame) -> pd.DataFrame:
        temp = pl.from_pandas(ohlcv[["close"]])
        
        mas = {}
        for period in self._periods:
            ma = temp.select(
                pl.col("close").rolling_mean(window_size=period).alias(f"ma_{period}")
            ).to_pandas()[f"ma_{period}"]
            mas[period] = ma
        
        result = pd.DataFrame(index=ohlcv.index)
        
        for period, ma in mas.items():
            result[f"{self.name}_ma{period}"] = ma
        
        if len(self._periods) >= 2:
            for i in range(len(self._periods) - 1):
                fast_period = self._periods[i]
                slow_period = self._periods[i + 1]
                result[f"{self.name}_cross_{fast_period}_{slow_period}"] = (
                    mas[fast_period] > mas[slow_period]
                ).astype(int)
        
        if len(self._periods) >= 2:
            ma_df = pd.DataFrame(mas, index=ohlcv.index)
            min_ma = ma_df.min(axis=1)
            max_ma = ma_df.max(axis=1)
            result[f"{self.name}_spread"] = (max_ma - min_ma) / ohlcv["close"] * 100
        
        return result


class IndicatorProcessor:    
    def __init__(self):
        self._indicators: List[TechnicalIndicator] = []

    def add_indicator(self, indicator: TechnicalIndicator) -> None:
        self._indicators.append(indicator)

    def process(self, ohlcv: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        indicator_dfs = []
        
        for indicator in self._indicators:
            try:
                result = indicator.calculate(ohlcv)
                indicator_dfs.append(result)
            except Exception as e:
                print(f"Error calculating {indicator.name}: {e}")
                continue
        
        ti_df = pd.concat(indicator_dfs, axis=1) if indicator_dfs else pd.DataFrame(
            index=ohlcv.index)
        return {"ohlcv": ohlcv, "indicators": ti_df}
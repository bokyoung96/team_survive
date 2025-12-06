from decimal import Decimal
from typing import Any, Dict, Optional
import pandas as pd
from datetime import datetime, date

from backtest.strategies import StreamingStrategy, TradingContext
from backtest.types import ActionType, Signal
from backtest.logger import get_logger


class Dolpha3Strategy(StreamingStrategy):
    def __init__(
        self,
        data: Any,
        lookback_periods: int = 500,
        position_size_pct: float = 0.01,
        scale_multiplier: float = 1.4,
        max_entries: int = 10,
        add_threshold: float = -0.05,
        stop_loss_pct: float = -0.15,
        daily_stop_limit: int = 3,
        use_aggressive_exits: bool = False,
    ):
        super().__init__("TrendPullback", lookback_periods=lookback_periods)
        self.data = data
        self.position_size_pct = Decimal(str(position_size_pct))
        self.scale_multiplier = Decimal(str(scale_multiplier))
        self.max_entries = max_entries
        self.add_threshold = Decimal(str(add_threshold))
        self.stop_loss_pct = Decimal(str(stop_loss_pct))
        self.daily_stop_limit = daily_stop_limit
        self.use_aggressive_exits = use_aggressive_exits
        self.logger = get_logger(__name__)

        self.entry_count = 0
        self.lowest_entry_price = Decimal("0")
        self.highest_price = Decimal("0")
        self.next_add_price: Optional[Decimal] = None
        self.tp_hits: Dict[float, bool] = {}
        self.daily_stops = 0
        self.last_stop_date: Optional[date] = None
        self.trading_enabled = True
        self.indicators: Dict[str, pd.DataFrame] = {}

    def reset_state(self) -> None:
        super().reset_state()
        self.entry_count = 0
        self.lowest_entry_price = Decimal("0")
        self.highest_price = Decimal("0")
        self.next_add_price = None
        self.tp_hits.clear()
        self.daily_stops = 0
        self.last_stop_date = None
        self.trading_enabled = True
        self.indicators.clear()

    def update_indicators(self, context: TradingContext) -> None:
        if self.indicators:
            return
        frames = {
            "3m": self.data["3m"],
            "15m": self.data["15m"],
            "30m": self.data["30m"],
            "1h": self.data["1h"],
            "4h": self.data["4h"],
        }
        self.indicators = {k: v.copy() for k, v in frames.items()}

        def ma(df: pd.DataFrame, length: int) -> pd.Series:
            return df["close"].rolling(length, min_periods=1).mean()

        def ema(series: pd.Series, length: int) -> pd.Series:
            return series.ewm(span=length, adjust=False).mean()

        def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
            close = df["close"]
            delta = close.diff()
            gain = delta.clip(lower=0).rolling(length).mean()
            loss = (-delta.clip(upper=0)).rolling(length).mean()
            rs = gain / loss.replace(0, pd.NA)
            return 100 - (100 / (1 + rs))

        def macd(df: pd.DataFrame) -> pd.DataFrame:
            fast = ema(df["close"], 12)
            slow = ema(df["close"], 26)
            line = fast - slow
            signal = line.ewm(span=9, adjust=False).mean()
            hist = line - signal
            return pd.DataFrame({"macd": line, "macd_signal": signal, "macd_hist": hist}, index=df.index)

        one_h = self.indicators["1h"]
        one_h["ma112"] = ma(one_h, 112)
        one_h["ma224"] = ma(one_h, 224)
        one_h["ma448"] = ma(one_h, 448)
        one_h["ma60"] = ma(one_h, 60)
        one_h["rsi"] = rsi(one_h)
        one_h[["macd", "macd_signal", "macd_hist"]] = macd(one_h)

        four_h = self.indicators["4h"]
        four_h[["macd", "macd_signal", "macd_hist"]] = macd(four_h)

        for tf, length in [("3m", 360), ("30m", 60)]:
            df = self.indicators[tf]
            df["ma"] = ma(df, length)
        fifteen = self.indicators["15m"]
        fifteen["rsi"] = rsi(fifteen)
        thirty = self.indicators["30m"]
        thirty["rsi"] = rsi(thirty)
        three = self.indicators["3m"]
        three["rsi"] = rsi(three)

    def process_bar(self, context: TradingContext) -> Optional[Signal]:
        ts = context.timestamp
        price = Decimal(str(context.current_bar["close"]))
        self.update_indicators(context)
        self._reset_daily(ts)

        pos = context.position
        if pos and pos.is_open:
            return self._manage_position(ts, price, pos)
        return self._try_entry(ts, price)

    def _reset_daily(self, ts: datetime) -> None:
        if self.last_stop_date and ts.date() != self.last_stop_date:
            self.daily_stops = 0
            self.trading_enabled = True

    def _latest(self, tf: str, col: str, ts: datetime) -> Optional[float]:
        df = self.indicators.get(tf)
        if df is None:
            return None
        sub = df.loc[:ts]
        if sub.empty or col not in sub.columns:
            return None
        return float(sub[col].ffill().iloc[-1])

    def _touches(self, tf: str, col: str, ts: datetime) -> int:
        df = self.indicators.get(tf)
        if df is None or col not in df.columns:
            return 0
        sub = df.loc[:ts]
        if sub.empty:
            return 0
        cond = (sub["low"] <= sub[col]) & (sub["high"] >= sub[col]) & sub[col].notna()
        hits = cond & ~cond.shift(1, fill_value=False)
        return int(hits.sum())

    def _trend_ok(self, ts: datetime) -> bool:
        m1 = self._latest("1h", "macd", ts)
        s1 = self._latest("1h", "macd_signal", ts)
        h1 = self._latest("1h", "macd_hist", ts)
        m4 = self._latest("4h", "macd", ts)
        s4 = self._latest("4h", "macd_signal", ts)
        h4 = self._latest("4h", "macd_hist", ts)
        if None in (m1, s1, h1, m4, s4, h4):
            return False

        ma112 = self._latest("1h", "ma112", ts)
        ma224 = self._latest("1h", "ma224", ts)
        ma448 = self._latest("1h", "ma448", ts)
        if None in (ma112, ma224, ma448):
            return False

        order_ok = ma112 > ma224 > ma448
        if not order_ok:
            return False
        spread_ok = (ma112 - ma224) / ma224 <= 0.2 and (ma224 - ma448) / ma448 <= 0.2
        macd_ok = m1 > s1 and h1 > 0 and m4 > s4 and h4 > 0
        return macd_ok and spread_ok

    def _rsi_ok(self, ts: datetime) -> bool:
        levels = [
            self._latest("3m", "rsi", ts),
            self._latest("15m", "rsi", ts),
            self._latest("30m", "rsi", ts),
            self._latest("1h", "rsi", ts),
        ]
        return all(v is not None and v <= 30 for v in levels)

    def _support_ok(self, ts: datetime) -> bool:
        checks = [
            ("3m", "ma"),
            ("30m", "ma"),
            ("1h", "ma60"),
        ]
        for tf, col in checks:
            ma_val = self._latest(tf, col, ts)
            close = self._latest(tf, "close", ts)
            if ma_val is None or close is None:
                return False
            if self._touches(tf, col, ts) < 2 or close < ma_val:
                return False
        return True

    def _entry_ready(self, ts: datetime) -> bool:
        return self.trading_enabled and self._trend_ok(ts) and self._rsi_ok(ts) and self._support_ok(ts)

    def _try_entry(self, ts: datetime, price: Decimal) -> Optional[Signal]:
        if not self._entry_ready(ts):
            return None
        self.entry_count = 1
        self.lowest_entry_price = price
        self.highest_price = price
        self.next_add_price = price * (Decimal("1") + self.add_threshold)
        self.tp_hits = {0.382: False, 0.5: False, 0.786: False}
        meta = {
            "entry_count": 1,
            "entry_timestamp": ts.isoformat(),
            "signal_period_id": ts.strftime("%Y-%m-%d"),
            "position_sizing": {
                "initial_size": float(self.position_size_pct),
                "scale_factor": float(self.scale_multiplier),
                "max_entries": self.max_entries,
            },
        }
        return Signal(type=ActionType.BUY, strength=1.0, metadata=meta)

    def _manage_position(self, ts: datetime, price: Decimal, pos: Any) -> Optional[Signal]:
        self.entry_count = int(pos.metadata.get("entry_count", 1))
        self.lowest_entry_price = min(self.lowest_entry_price or price, price)
        self.highest_price = max(self.highest_price or price, price)

        stop_sig = self._stop_signal(ts, price)
        if stop_sig:
            return stop_sig

        be_sig = self._breakeven_signal(price, pos)
        if be_sig:
            return be_sig

        tp_sig = self._take_profit_signal(price, pos)
        if tp_sig:
            return tp_sig

        add_sig = self._add_signal(price, pos)
        if add_sig:
            return add_sig
        return None

    def _stop_signal(self, ts: datetime, price: Decimal) -> Optional[Signal]:
        if self.entry_count < 3:
            return None
        trigger = self.lowest_entry_price * (Decimal("1") + self.stop_loss_pct)
        if price > trigger:
            return None
        self.daily_stops += 1
        self.last_stop_date = ts.date()
        if self.daily_stops >= self.daily_stop_limit:
            self.trading_enabled = False
        meta = {"reason": "stop", "entry_count": self.entry_count}
        return Signal(type=ActionType.SELL, strength=1.0, metadata=meta)

    def _breakeven_signal(self, price: Decimal, pos: Any) -> Optional[Signal]:
        if self.entry_count <= 1:
            return None
        if price < pos.entry_price:
            return None
        meta = {"reason": "breakeven", "entry_count": self.entry_count}
        return Signal(type=ActionType.SELL, strength=1.0, metadata=meta)

    def _take_profit_signal(self, price: Decimal, pos: Any) -> Optional[Signal]:
        if self.highest_price <= self.lowest_entry_price:
            return None
        low = self.lowest_entry_price
        high = max(self.highest_price, price)
        diff = high - low
        targets = [0.382, 0.5]
        weights = [Decimal("0.5"), Decimal("0.5")]
        if self.use_aggressive_exits:
            targets = [0.382, 0.5, 0.786]
            weights = [Decimal("0.5"), Decimal("0.25"), Decimal("0.25")]

        for level, weight in zip(targets, weights):
            if self.tp_hits.get(level):
                continue
            target_price = low + diff * Decimal(str(level))
            if price >= target_price:
                self.tp_hits[level] = True
                qty = (pos.open_quantity or pos.quantity) * weight
                meta = {"reason": f"tp_{level}", "entry_count": self.entry_count}
                return Signal(
                    type=ActionType.SELL,
                    strength=1.0,
                    quantity=qty,
                    metadata=meta,
                )
        return None

    def _add_signal(self, price: Decimal, pos: Any) -> Optional[Signal]:
        if self.entry_count >= self.max_entries:
            return None
        if self.next_add_price is None:
            self.next_add_price = pos.entry_price * (Decimal("1") + self.add_threshold)
        if price > self.next_add_price:
            return None
        next_count = self.entry_count + 1
        self.entry_count = next_count
        self.lowest_entry_price = min(self.lowest_entry_price, price)
        self.next_add_price = self.next_add_price * (Decimal("1") + self.add_threshold)
        meta = {
            "entry_count": next_count,
            "entry_timestamp": datetime.utcnow().isoformat(),
            "signal_period_id": pos.metadata.get("signal_period_id", "mtf"),
            "position_sizing": {
                "initial_size": float(self.position_size_pct),
                "scale_factor": float(self.scale_multiplier),
                "max_entries": self.max_entries,
            },
        }
        return Signal(type=ActionType.BUY, strength=1.0, metadata=meta)

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from typing import Sequence

if True:
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))

from execution.exchange import ExchangeGateway
from execution.strategy.base import SignalContext, TradeSignal, TradingStrategy
from execution.types import OrderSide, OrderType

__all__ = ["CpoStrategy"]


@dataclass
class CpoStrategy(TradingStrategy):
    exchange: ExchangeGateway
    symbol: str
    window: int = 20
    timeframe: str = "1m"
    min_history: int = 25
    confidence_scale: float = 1.0
    _last_signal_ts: int | None = field(default=None, init=False, repr=False)

    def generate(self, context: SignalContext) -> TradeSignal | None:
        limit = max(self.window + 2, self.min_history)
        candles = self.exchange.fetch_ohlcv(
            self.symbol, self.timeframe, limit=limit)
        closes = self._extract_closes(candles)
        if len(closes) < self.window + 1:
            return None

        last_ts = int(candles[-1][0])
        if self._last_signal_ts == last_ts:
            return None

        prev_close = closes[-2]
        current_close = closes[-1]
        prev_ma = self._sma(closes[-(self.window + 1): -1])
        current_ma = self._sma(closes[-self.window:])
        crossed_under = prev_close >= prev_ma and current_close < current_ma
        if not crossed_under:
            return None
        if context.position_size > 0:
            return None

        confidence = self._confidence(
            prev_close, prev_ma, current_close, current_ma)
        self._last_signal_ts = last_ts
        return TradeSignal(
            side=OrderSide.BUY,
            confidence=confidence,
            order_type=OrderType.MARKET,
        )

    def _extract_closes(self, candles: Sequence[Sequence[float | int]]) -> list[float]:
        closes: list[float] = []
        for entry in candles:
            if len(entry) < 5:
                continue
            close = entry[4]
            try:
                closes.append(float(close))
            except (TypeError, ValueError):
                continue
        return closes

    def _sma(self, values: Sequence[float]) -> float:
        return sum(values) / len(values)

    def _confidence(
        self,
        prev_close: float,
        prev_ma: float,
        current_close: float,
        current_ma: float,
    ) -> float:
        if current_ma <= 0:
            return 1.0
        penetration = max(0.0, (prev_close - prev_ma) +
                          (current_ma - current_close))
        scaled = penetration / current_ma
        return max(0.1, min(self.confidence_scale, scaled))

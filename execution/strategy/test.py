from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

if True:
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))

from execution.strategy.base import SignalContext, TradeSignal, TradingStrategy
from execution.types import OrderSide, OrderType

__all__ = ["TestStrategy"]


@dataclass
class TestStrategy(TradingStrategy):
    side: OrderSide = OrderSide.SELL
    once: bool = True
    confidence: float = 1.0
    _fired: bool = field(default=False, init=False, repr=False)

    def generate(self, context: SignalContext) -> TradeSignal | None:
        del context
        if self.once and self._fired:
            return None
        self._fired = True
        return TradeSignal(
            side=self.side,
            confidence=max(0.0, self.confidence),
            order_type=OrderType.MARKET,
        )

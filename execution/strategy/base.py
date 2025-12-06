from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Optional, Protocol

if True:
    sys.path.append(os.path.dirname(
        os.path.dirname(os.path.dirname(__file__))))

from execution.types import OrderSide, OrderType, ProtectiveTargets

__all__ = [
    "TradeSignal",
    "SignalContext",
    "TradingStrategy",
]


@dataclass(frozen=True)
class TradeSignal:
    side: OrderSide
    confidence: float
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    use_bbo_counterparty: bool = False
    protective_targets: Optional[ProtectiveTargets] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "side", OrderSide(self.side))
        object.__setattr__(self, "order_type", OrderType(self.order_type))


@dataclass(frozen=True)
class SignalContext:
    price: float
    avg_entry_price: float
    position_size: float


class TradingStrategy(Protocol):
    def generate(self, context: SignalContext) -> TradeSignal | None:
        ...

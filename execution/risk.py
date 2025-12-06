from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Protocol, Sequence

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.types import OrderRequest, PositionSnapshot


class RiskManager(Protocol):
    def approve(self, order: OrderRequest, position: PositionSnapshot) -> bool:
        ...


@dataclass(slots=True)
class MaxLeverageGuard:
    max_leverage: int

    def approve(self, order: OrderRequest, position: PositionSnapshot) -> bool:
        del position
        return order.leverage <= self.max_leverage


@dataclass(slots=True)
class MaxPositionGuard:
    max_position: float

    def approve(self, order: OrderRequest, position: PositionSnapshot) -> bool:
        projected = abs(position.position_amt + order.quantity)
        return projected <= self.max_position


@dataclass(slots=True)
class CompositeRiskManager:
    checks: Sequence[RiskManager]

    def approve(self, order: OrderRequest, position: PositionSnapshot) -> bool:
        for check in self.checks:
            if not check.approve(order, position):
                return False
        return True

from __future__ import annotations

import abc
import logging
import os
import sys
from dataclasses import dataclass, field

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.config import TradingConfig
from execution.exchange import ExchangeGateway
from execution.strategy import TradeSignal


class PositionSizer(abc.ABC):
    @abc.abstractmethod
    def size(self, signal: TradeSignal) -> float:
        raise NotImplementedError


@dataclass(slots=True)
class DynamicFractionSizer(PositionSizer):
    config: TradingConfig
    exchange: ExchangeGateway
    fraction: float = 1.0
    quote_currency: str = "USDT"
    include_leverage: bool = True
    balance_bucket: str = "total"
    _currency: str = field(init=False, repr=False)
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__), repr=False)

    def __post_init__(self) -> None:
        if self.fraction < 0:
            raise ValueError("fraction must be non-negative")
        self._currency = self.quote_currency.upper()

    def size(self, signal: TradeSignal) -> float:
        del signal
        balance_raw = self.exchange.account_balance(
            self._currency, bucket=self.balance_bucket)
        balance = max(0.0, balance_raw)
        if balance <= 0:
            self.logger.info(
                "Dynamic sizer | balance depleted | currency=%s bucket=%s raw_balance=%.6f fraction=%.4f",
                self._currency,
                self.balance_bucket,
                balance_raw,
                self.fraction,
            )
            return 0.0

        allocation = balance * self.fraction
        price = max(0.0, self.exchange.current_price(self.config.symbol))
        if price <= 0:
            self.logger.warning(
                "Dynamic sizer | invalid price | symbol=%s price=%.6f allocation=%.6f",
                self.config.symbol,
                price,
                allocation,
            )
            return 0.0

        quantity = allocation / price
        leverage_used = max(self.config.leverage,
                            1) if self.include_leverage else 1
        quantity *= leverage_used
        quantity = max(0.0, quantity)

        min_required = 0.0
        get_min_qty = getattr(self.exchange, "minimum_order_quantity", None)
        if callable(get_min_qty):
            try:
                min_candidate = get_min_qty(self.config.symbol)
                min_required = float(min_candidate) if min_candidate else 0.0
            except Exception as exc:  # pragma: no cover - defensive path
                self.logger.debug(
                    "Dynamic sizer | failed to fetch min quantity | error=%s", exc)
                min_required = 0.0

        if min_required > 0 and quantity < min_required:
            min_cost = min_required * price
            max_cost = allocation * leverage_used
            total_capacity = balance * leverage_used
            if min_cost > total_capacity:
                self.logger.warning(
                    "Dynamic sizer | minimum quantity %.6f exceeds available capacity %.6f | skipping order",
                    min_required,
                    total_capacity,
                )
                return 0.0

            if min_cost > max_cost:
                self.logger.info(
                    "Dynamic sizer | minimum quantity %.6f requires %.6f (above allocation %.6f) | using minimum",
                    min_required,
                    min_cost,
                    max_cost,
                )
            else:
                self.logger.info(
                    "Dynamic sizer | quantity %.6f below exchange minimum %.6f | bumping to minimum",
                    quantity,
                    min_required,
                )
            quantity = min_required

        self.logger.info(
            (
                "Dynamic sizer | currency=%s bucket=%s balance=%.6f allocation=%.6f "
                "price=%.6f quantity=%.6f leverage_applied=%s"
            ),
            self._currency,
            self.balance_bucket,
            balance,
            allocation,
            price,
            quantity,
            leverage_used,
        )
        return quantity

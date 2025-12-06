from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.config import TradingConfig
from execution.exchange import ExchangeGateway
from execution.events import ErrorEvent, EventBus, OrderEvent
from execution.orders import OrderPlan, OrderPlanner
from execution.risk import RiskManager
from execution.sizing import PositionSizer
from execution.strategy import SignalContext, TradeSignal, TradingStrategy
from execution.types import BestBidAsk, OrderRequest, OrderResult, OrderType, PositionSnapshot


@dataclass(slots=True)
class TradingEngine:
    config: TradingConfig
    exchange: ExchangeGateway
    strategy: TradingStrategy
    sizer: PositionSizer
    risk: RiskManager
    planner: OrderPlanner | None = None
    bus: EventBus | None = None
    _logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__), init=False)

    def __post_init__(self) -> None:
        if self.planner is None:
            self.planner = OrderPlanner(self.config)

    def run_cycle(
        self,
        *,
        price_hint: float | None = None,
        position_snapshot: PositionSnapshot | None = None,
        best_bid_ask: Optional[BestBidAsk] = None,
    ) -> Optional[TradeSignal]:
        position = position_snapshot or self.exchange.position(self.config.symbol)
        price = price_hint if price_hint is not None else self.exchange.current_price(self.config.symbol)
        context = SignalContext(
            price=price,
            avg_entry_price=position.entry_price,
            position_size=position.position_amt,
        )
        signal = self.strategy.generate(context)
        if signal is None:
            self._logger.debug("Strategy %s produced no signal", type(self.strategy).__name__)
            return None

        quantity = self.sizer.size(signal)
        if quantity <= 0:
            self._logger.debug("Sizer returned non-positive quantity for signal %s", signal)
            return None

        book_snapshot = best_bid_ask
        if book_snapshot is None and signal.order_type == OrderType.LIMIT and signal.use_bbo_counterparty:
            book_snapshot = self.exchange.best_bid_ask(self.config.symbol)

        plan = self._create_plan(signal, quantity, price, book_snapshot)
        if not self.risk.approve(plan.entry, position):
            self._logger.info("Order rejected by risk manager | request=%s", plan.entry)
            return None

        if not plan.protective:
            self._logger.debug("No protective orders derived for signal %s", signal)

        try:
            entry_result = self.exchange.submit_order(plan.entry)
        except Exception as exc:
            message = f"Entry order submission failed | request={plan.entry}"
            self._logger.exception(message)
            self._emit_error(message, exc)
            return None

        self._logger.info(
            "Submitted entry order | symbol=%s side=%s qty=%s type=%s price=%s",
            plan.entry.symbol,
            plan.entry.side.value,
            plan.entry.quantity,
            plan.entry.order_type.value,
            plan.entry.price,
        )
        self._emit_order(plan.entry, entry_result)

        for protective in plan.protective:
            try:
                result = self.exchange.submit_order(protective)
            except Exception as exc:
                message = f"Protective order submission failed | request={protective}"
                self._logger.exception(message)
                self._emit_error(message, exc)
                continue

            self._logger.info(
                "Submitted protective order | symbol=%s side=%s qty=%s type=%s price=%s",
                protective.symbol,
                protective.side.value,
                protective.quantity,
                protective.order_type.value,
                protective.stop_price or protective.price,
            )
            self._emit_order(protective, result)
        return signal

    def _create_plan(
        self,
        signal: TradeSignal,
        quantity: float,
        reference_price: float,
        best_bid_ask: Optional[BestBidAsk],
    ) -> OrderPlan:
        assert self.planner is not None
        return self.planner.plan(signal, quantity, reference_price, best_bid_ask)

    def _emit_order(self, request: OrderRequest, result: OrderResult) -> None:
        if self.bus is None:
            return
        self.bus.publish(OrderEvent(request=request, result=result))

    def _emit_error(self, message: str, exc: Exception) -> None:
        if self.bus is None:
            return
        self.bus.publish(ErrorEvent(message=message, exception=exc))

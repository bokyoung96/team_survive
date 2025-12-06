from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, DefaultDict, List, Protocol

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.strategy import TradeSignal
from execution.types import (BestBidAsk, OrderRequest, OrderResult,
                             PositionSnapshot)

EventHandler = Callable[[Any], None]


class EventType(str, Enum):
    MARKET = "market"
    SIGNAL = "signal"
    ORDER_SUBMITTED = "order_submitted"
    ORDER_FILLED = "order_filled"
    ERROR = "error"


@dataclass(frozen=True)
class MarketEvent:
    symbol: str
    price: float
    timestamp: datetime | None = None
    best_bid_ask: BestBidAsk | None = None
    position: PositionSnapshot | None = None
    event_type: EventType = EventType.MARKET


@dataclass(frozen=True)
class SignalEvent:
    signal: TradeSignal
    timestamp: datetime | None = None
    event_type: EventType = EventType.SIGNAL


@dataclass(frozen=True)
class OrderEvent:
    request: OrderRequest
    result: OrderResult | None = None
    timestamp: datetime | None = None
    event_type: EventType = EventType.ORDER_SUBMITTED


@dataclass(frozen=True)
class OrderFillEvent:
    result: OrderResult
    timestamp: datetime | None = None
    event_type: EventType = EventType.ORDER_FILLED


@dataclass(frozen=True)
class ErrorEvent:
    message: str
    exception: Exception | None = None
    timestamp: datetime | None = None
    event_type: EventType = EventType.ERROR


class SupportsEventType(Protocol):
    event_type: EventType


class EventBus:
    def __init__(self) -> None:
        self._subscribers: DefaultDict[EventType |
                                       None, List[EventHandler]] = defaultdict(list)

    def subscribe(self, handler: EventHandler, event_type: EventType | None = None) -> None:
        self._subscribers[event_type].append(handler)

    def unsubscribe(self, handler: EventHandler, event_type: EventType | None = None) -> None:
        handlers = self._subscribers.get(event_type, [])
        try:
            handlers.remove(handler)
        except ValueError:
            return
        if not handlers:
            self._subscribers.pop(event_type, None)

    def publish(self, event: SupportsEventType) -> None:
        for handler in list[EventHandler](self._subscribers.get(event.event_type, ())):
            handler(event)
        for handler in list[EventHandler](self._subscribers.get(None, ())):
            handler(event)


@dataclass(slots=True)
class OrderLogger:
    track_submissions: bool = True
    track_fills: bool = True
    logger: logging.Logger = field(
        default_factory=lambda: logging.getLogger(__name__))

    def __call__(self, event: SupportsEventType) -> None:
        if isinstance(event, OrderEvent) and self.track_submissions:
            self.logger.info(
                "Order | submitted | id=%s symbol=%s side=%s type=%s qty=%.6f price=%s reduce_only=%s",
                event.result.order_id if event.result else "pending",
                event.request.symbol,
                event.request.side.value,
                event.request.order_type.value,
                event.request.quantity,
                event.request.price,
                event.request.reduce_only,
            )
        elif isinstance(event, OrderFillEvent) and self.track_fills:
            self.logger.info(
                "Order | filled   | id=%s status=%s symbol=%s side=%s",
                event.result.order_id,
                event.result.status,
                event.result.symbol,
                event.result.side.value,
            )
        elif isinstance(event, ErrorEvent):
            self.logger.error("Order | error    | %s",
                              event.message, exc_info=event.exception)

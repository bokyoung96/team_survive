from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_MARKET = "STOP_MARKET"
    TAKE_PROFIT_MARKET = "TAKE_PROFIT_MARKET"
    STOP = "STOP"
    TAKE_PROFIT = "TAKE_PROFIT"


class TargetKind(str, Enum):
    PRICE = "price"
    PERCENT = "percent"
    ROI = "roi"
    PNL = "pnl"


@dataclass(frozen=True)
class TargetSpec:
    kind: TargetKind
    value: float


@dataclass(frozen=True)
class ProtectiveTargets:
    stop_loss: Optional[TargetSpec] = None
    take_profit: Optional[TargetSpec] = None


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    leverage: int
    price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: Optional[str] = None
    reduce_only: bool = False


@dataclass(frozen=True)
class OrderResult:
    symbol: str
    side: OrderSide
    order_id: str
    status: str


@dataclass(frozen=True)
class PositionSnapshot:
    symbol: str
    position_amt: float
    entry_price: float
    leverage: int
    unrealized_pnl: float


@dataclass(frozen=True)
class BestBidAsk:
    bid_price: float
    bid_qty: float
    ask_price: float
    ask_qty: float

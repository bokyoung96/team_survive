from enum import Enum
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Optional, Dict, Any


class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"
    ENTRY = "entry"
    EXIT = "exit"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class PositionStatus(Enum):
    OPEN = "open"
    CLOSED = "closed"
    PARTIALLY_CLOSED = "partially_closed"


@dataclass(frozen=True)
class Signal:
    type: ActionType
    strength: float
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_entry(self) -> bool:
        return self.type in [ActionType.BUY, ActionType.SELL, ActionType.ENTRY]
    
    @property
    def is_exit(self) -> bool:
        return self.type in [ActionType.CLOSE, ActionType.CLOSE_LONG, ActionType.CLOSE_SHORT, ActionType.EXIT]


@dataclass(frozen=True)
class TransactionCost:
    maker_fee: Decimal = Decimal("0.0002")
    taker_fee: Decimal = Decimal("0.0004")
    slippage: Decimal = Decimal("0.0001")
    fixed_cost: Decimal = Decimal("0")
    
    def calculate_cost(
        self,
        price: Decimal,
        quantity: Decimal,
        is_maker: bool = False
    ) -> Dict[str, Decimal]:
        notional = price * quantity
        fee_rate = self.maker_fee if is_maker else self.taker_fee
        
        fee = notional * fee_rate
        slippage_cost = notional * self.slippage
        total_cost = fee + slippage_cost + self.fixed_cost
        
        return {
            "fee": fee,
            "slippage": slippage_cost,
            "fixed": self.fixed_cost,
            "total": total_cost
        }
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Optional, List, Dict, Any
import pandas as pd

from backtest.types import ActionType, OrderType, OrderStatus, PositionStatus, TransactionCost




@dataclass
class Order:
    id: str
    symbol: str
    side: ActionType
    order_type: OrderType
    quantity: Decimal
    price: Optional[Decimal] = None
    stop_price: Optional[Decimal] = None
    timestamp: Optional[datetime] = None
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: Decimal = Decimal("0")
    filled_price: Optional[Decimal] = None
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def remaining_quantity(self) -> Decimal:
        return self.quantity - self.filled_quantity
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.PARTIALLY_FILLED]
    
    def fill(
        self,
        price: Decimal,
        quantity: Optional[Decimal] = None,
        commission: Decimal = Decimal("0"),
        slippage: Decimal = Decimal("0")
    ) -> None:
        if price <= 0:
            raise ValueError(f"Fill price must be positive, got {price}")
        
        if commission < 0 or slippage < 0:
            raise ValueError(f"Commission and slippage must be non-negative, got commission={commission}, slippage={slippage}")
        
        fill_qty = quantity or self.remaining_quantity
        
        if fill_qty <= 0:
            raise ValueError(f"Fill quantity must be positive, got {fill_qty}")
        
        if fill_qty > self.remaining_quantity:
            raise ValueError(f"Fill quantity {fill_qty} exceeds remaining {self.remaining_quantity}")
        
        if self.filled_quantity == 0:
            self.filled_price = price
            self.filled_quantity = fill_qty
        else:
            total_value = (self.filled_price * self.filled_quantity) + (price * fill_qty)
            self.filled_quantity += fill_qty
            self.filled_price = total_value / self.filled_quantity
        self.commission += commission
        self.slippage += slippage
        
        if self.filled_quantity >= self.quantity:
            self.status = OrderStatus.FILLED
        else:
            self.status = OrderStatus.PARTIALLY_FILLED


@dataclass
class Position:
    id: str
    symbol: str
    side: ActionType
    entry_price: Decimal
    quantity: Decimal
    entry_time: datetime
    status: PositionStatus = PositionStatus.OPEN
    exit_price: Optional[Decimal] = None
    exit_time: Optional[datetime] = None
    closed_quantity: Decimal = Decimal("0")
    commission: Decimal = Decimal("0")
    slippage: Decimal = Decimal("0")
    realized_pnl: Decimal = Decimal("0")
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def open_quantity(self) -> Decimal:
        return self.quantity - self.closed_quantity
    
    @property
    def is_open(self) -> bool:
        return self.status == PositionStatus.OPEN
    
    @property
    def average_entry_price(self) -> Decimal:
        return self.entry_price
    
    def calculate_pnl(self, current_price: Decimal) -> Dict[str, Decimal]:
        if self.side == ActionType.BUY:
            price_diff = current_price - self.entry_price
        else:
            price_diff = self.entry_price - current_price
            
        gross_pnl = price_diff * self.open_quantity
        net_pnl = gross_pnl - self.commission - self.slippage
        
        return {
            "gross_pnl": gross_pnl,
            "commission": self.commission,
            "slippage": self.slippage,
            "net_pnl": net_pnl,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": net_pnl if self.is_open else Decimal("0")
        }
    
    def close(
        self,
        price: Decimal,
        quantity: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
        commission: Decimal = Decimal("0"),
        slippage: Decimal = Decimal("0")
    ) -> Decimal:
        close_qty = quantity or self.open_quantity
        
        if close_qty > self.open_quantity:
            raise ValueError(f"Close quantity {close_qty} exceeds open {self.open_quantity}")
        
        if self.side == ActionType.BUY:
            gross_pnl = (price - self.entry_price) * close_qty
        else:
            gross_pnl = (self.entry_price - price) * close_qty
            
        self.closed_quantity += close_qty
        self.commission += commission
        self.slippage += slippage
        self.realized_pnl += gross_pnl - commission - slippage
        
        if self.closed_quantity >= self.quantity:
            self.status = PositionStatus.CLOSED
            self.exit_price = price
            self.exit_time = timestamp
        else:
            self.status = PositionStatus.PARTIALLY_CLOSED
            
        return self.realized_pnl


@dataclass
class Portfolio:
    initial_capital: Decimal
    currency: str = "USDT"
    cash: Optional[Decimal] = None
    positions: Dict[str, List[Position]] = field(default_factory=dict)
    orders: List[Order] = field(default_factory=list)
    closed_positions: List[Position] = field(default_factory=list)
    transaction_history: List[Dict[str, Any]] = field(default_factory=list)
    
    _last_metrics: Optional[Dict[str, Any]] = field(default=None, init=False)
    _last_metrics_prices: Optional[Dict[str, Decimal]] = field(default=None, init=False)
    
    def __post_init__(self):
        if self.cash is None:
            self.cash = self.initial_capital
    
    def _get_open_positions_map(self) -> Dict[str, Position]:
        open_positions = {}
        for symbol, positions in self.positions.items():
            open_pos = [p for p in positions if p.is_open]
            if open_pos:
                open_positions[symbol] = open_pos[0]
        return open_positions
    
    def get_total_value(self, current_prices: Optional[Dict[str, Decimal]] = None) -> Decimal:
        open_positions = self._get_open_positions_map()
        
        if current_prices:
            return self.cash + sum(
                pos.open_quantity * current_prices.get(symbol, pos.entry_price)
                for symbol, pos in open_positions.items()
            )
        else:
            return self.cash + sum(
                pos.open_quantity * pos.entry_price
                for pos in open_positions.values()
            )
    
    @property
    def total_value(self) -> Decimal:
        return self.get_total_value()
    
    @property
    def open_positions(self) -> List[Position]:
        return [
            pos for positions in self.positions.values()
            for pos in positions if pos.is_open
        ]
    
    @property
    def position_count(self) -> int:
        return len(self.open_positions)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        if symbol in self.positions:
            open_positions = [p for p in self.positions[symbol] if p.is_open]
            return open_positions[0] if open_positions else None
        return None
    
    def add_position(self, position: Position) -> None:
        if position.symbol not in self.positions:
            self.positions[position.symbol] = []
        self.positions[position.symbol].append(position)
        
        if position.side == ActionType.BUY:
            self.cash -= (position.quantity * position.entry_price + position.commission + position.slippage)
        elif position.side == ActionType.SELL:
            self.cash += (position.quantity * position.entry_price - position.commission - position.slippage)
        else:
            raise ValueError(f"Invalid position side: {position.side}")
        
        self._last_metrics = None
        self._last_metrics_prices = None
    
    def close_position(
        self,
        position: Position,
        price: Decimal,
        quantity: Optional[Decimal] = None,
        timestamp: Optional[datetime] = None,
        commission: Decimal = Decimal("0"),
        slippage: Decimal = Decimal("0")
    ) -> Decimal:
        pnl = position.close(price, quantity, timestamp, commission, slippage)
        
        close_qty = quantity or position.open_quantity
        
        if position.side == ActionType.BUY:
            self.cash += (close_qty * price - commission - slippage)
        elif position.side == ActionType.SELL:
            self.cash -= (close_qty * price + commission + slippage)
        else:
            raise ValueError(f"Invalid position side: {position.side}")
            
        if not position.is_open:
            self.closed_positions.append(position)
        
        self._last_metrics = None
        self._last_metrics_prices = None
            
        return pnl
    
    def calculate_metrics(self, current_prices: Dict[str, Decimal]) -> Dict[str, Any]:
        if (self._last_metrics is not None and 
            self._last_metrics_prices is not None and
            self._last_metrics_prices == current_prices and
            self._last_metrics.get("closed_trades") == len(self.closed_positions)):
            cached_metrics = self._last_metrics.copy()
            cached_metrics["cash"] = self.cash
            cached_metrics["position_count"] = self.position_count
            return cached_metrics
        
        total_value = self.cash
        unrealized_pnl = Decimal("0")
        
        open_positions = self._get_open_positions_map()
        for symbol, pos in open_positions.items():
            if symbol in current_prices:
                pnl = pos.calculate_pnl(current_prices[symbol])
                unrealized_pnl += pnl["unrealized_pnl"]
                total_value += pos.open_quantity * current_prices.get(symbol, pos.entry_price)
        
        realized_pnl = sum(pos.realized_pnl for pos in self.closed_positions)
        total_pnl = realized_pnl + unrealized_pnl
        
        metrics = {
            "cash": self.cash,
            "total_value": total_value,
            "realized_pnl": realized_pnl,
            "unrealized_pnl": unrealized_pnl,
            "total_pnl": total_pnl,
            "return": (total_value - self.initial_capital) / self.initial_capital,
            "position_count": self.position_count,
            "closed_trades": len(self.closed_positions)
        }
        
        self._last_metrics = metrics.copy()
        self._last_metrics_prices = current_prices.copy()
        
        return metrics


@dataclass
class BacktestResult:
    portfolio: Portfolio
    equity_curve: pd.DataFrame
    trades: pd.DataFrame
    signals: pd.DataFrame
    metrics: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> Dict[str, Any]:
        return {
            **self.metrics,
            "total_trades": len(self.trades),
            "winning_trades": len(self.trades[self.trades["pnl"] > 0]),
            "losing_trades": len(self.trades[self.trades["pnl"] < 0]),
            "metadata": self.metadata
        }
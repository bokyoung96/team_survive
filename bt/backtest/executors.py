from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple
import pandas as pd

from backtest.models import (
    Order,
    Position,
    Portfolio
)
from backtest.types import ActionType, Signal, OrderType, OrderStatus, TransactionCost


class OrderExecutor:
    def __init__(self, transaction_cost: Optional[TransactionCost] = None):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.order_id_counter = 0
    
    def create_order_from_signal(
        self,
        signal: Signal,
        symbol: str,
        timestamp: datetime,
        portfolio: Portfolio
    ) -> Optional[Order]:
        if signal.is_exit:
            position = portfolio.get_position(symbol)
            if not position or not position.is_open:
                return None
            
            quantity = signal.quantity or position.open_quantity
            side = ActionType.SELL if position.side == ActionType.BUY else ActionType.BUY
            
        elif signal.is_entry:
            side = ActionType.BUY if signal.type == ActionType.BUY else ActionType.SELL
            
            if signal.quantity:
                quantity = signal.quantity
            else:
                available_capital = portfolio.cash * Decimal("0.95")
                if signal.price:
                    quantity = available_capital / signal.price
                else:
                    return None
        else:
            return None
        
        self.order_id_counter += 1
        order = Order(
            id=f"ORDER_{self.order_id_counter:06d}",
            symbol=symbol,
            side=side,
            order_type=OrderType.MARKET,
            quantity=quantity,
            price=signal.price,
            timestamp=timestamp,
            metadata=signal.metadata
        )
        
        return order
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.Series,
        portfolio: Portfolio,
        timestamp: datetime
    ) -> Tuple[bool, Optional[Position]]:
        fill_price = Decimal(str(market_data["close"]))
        
        if order.quantity == Decimal("-1"):
            if order.side == ActionType.BUY:
                available_capital = portfolio.cash * Decimal("0.95")
                order.quantity = available_capital / fill_price
            else:
                existing_position = portfolio.get_position(order.symbol)
                if existing_position and existing_position.is_open:
                    order.quantity = existing_position.open_quantity
                else:
                    return False, None
        
        costs = self.transaction_cost.calculate_cost(
            fill_price,
            order.quantity,
            is_maker=False
        )
        
        if order.side == ActionType.BUY:
            total_cost = (fill_price * order.quantity) + costs["total"]
            if portfolio.cash < total_cost:
                order.status = OrderStatus.REJECTED
                return False, None
        
        order.fill(
            price=fill_price,
            quantity=order.quantity,
            commission=costs["fee"],
            slippage=costs["slippage"]
        )
        
        existing_position = portfolio.get_position(order.symbol)
        
        if existing_position and existing_position.is_open:
            if order.side != existing_position.side:
                close_quantity = min(order.quantity, existing_position.open_quantity)
                portfolio.close_position(
                    existing_position,
                    fill_price,
                    close_quantity,
                    timestamp,
                    costs["fee"],
                    costs["slippage"]
                )
                
                if order.quantity > close_quantity:
                    remaining = order.quantity - close_quantity
                    new_position = Position(
                        id=f"POS_{self.order_id_counter:06d}",
                        symbol=order.symbol,
                        side=order.side,
                        entry_price=fill_price,
                        quantity=remaining,
                        entry_time=timestamp,
                        commission=costs["fee"],
                        slippage=costs["slippage"]
                    )
                    portfolio.add_position(new_position)
                    return True, new_position
                
                return True, None
            else:
                new_position = Position(
                    id=f"POS_{self.order_id_counter:06d}",
                    symbol=order.symbol,
                    side=order.side,
                    entry_price=fill_price,
                    quantity=order.quantity,
                    entry_time=timestamp,
                    commission=costs["fee"],
                    slippage=costs["slippage"]
                )
                portfolio.add_position(new_position)
                return True, new_position
        else:
            position = Position(
                id=f"POS_{self.order_id_counter:06d}",
                symbol=order.symbol,
                side=order.side,
                entry_price=fill_price,
                quantity=order.quantity,
                entry_time=timestamp,
                commission=costs["fee"],
                slippage=costs["slippage"]
            )
            portfolio.add_position(position)
            return True, position
    
    def create_order_from_signal_data(
        self,
        signal_type: ActionType,
        strength: float,
        symbol: str,
        timestamp: datetime,
        portfolio: Portfolio,
        metadata: dict = None
    ) -> Optional[Order]:
        if signal_type in [ActionType.SELL, ActionType.EXIT]:
            position = portfolio.get_position(symbol)
            if not position or not position.is_open:
                return None
            
            quantity = position.open_quantity
            side = ActionType.SELL if position.side == ActionType.BUY else ActionType.BUY
            
        elif signal_type in [ActionType.BUY, ActionType.ENTRY]:
            side = ActionType.BUY if signal_type == ActionType.BUY else ActionType.SELL
            
            quantity = Decimal("-1")
        else:
            return None
        
        self.order_id_counter += 1
        return Order(
            id=f"ORD_{self.order_id_counter:06d}",
            symbol=symbol,
            order_type=OrderType.MARKET,
            side=side,
            quantity=quantity,
            timestamp=timestamp,
            status=OrderStatus.PENDING,
            metadata=metadata or {}
        )
    
    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        market_data: pd.Series
    ) -> bool:
        if order.quantity <= 0:
            return False
        
        if order.side == ActionType.BUY:
            required_capital = order.quantity * Decimal(str(market_data["close"]))
            required_capital *= Decimal("1.05")
            if portfolio.cash < required_capital:
                return False
        return True
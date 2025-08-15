from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple
import pandas as pd

from backtest.models import (
    Order,
    Position,
    Portfolio
)
from backtest.types import ActionType, OrderType, OrderStatus, TransactionCost


class OrderCreator:
    def __init__(self):
        self.order_id_counter = 0

    def create_order(
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


class OrderValidator:
    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        market_data: pd.Series
    ) -> bool:
        if order.quantity == Decimal("-1"):
            return True

        if order.quantity <= 0:
            return False

        if order.side == ActionType.BUY:
            required_capital = order.quantity * \
                Decimal(str(market_data["close"]))
            required_capital *= Decimal("1.05")  # Add 5% buffer
            if portfolio.cash < required_capital:
                return False

        return True


class OrderExecutor:
    def __init__(self, transaction_cost: Optional[TransactionCost] = None):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.position_id_counter = 0

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
                if order.metadata and "position_sizing" in order.metadata:
                    position_sizing = order.metadata["position_sizing"]
                    entry_count = order.metadata.get("entry_count", 1) - 1
                    
                    if entry_count < position_sizing["max_entries"]:
                        initial_size = Decimal(str(position_sizing["initial_size"]))
                        scale_factor = Decimal(str(position_sizing["scale_factor"]))
                        size_percent = initial_size * (scale_factor ** entry_count)
                        position_value = portfolio.total_value * size_percent
                        order.quantity = position_value / fill_price
                    else:
                        return False, None
                else:
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
                close_quantity = min(
                    order.quantity, existing_position.open_quantity)
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
                    new_position = self._create_position(
                        order.symbol,
                        order.side,
                        fill_price,
                        remaining,
                        timestamp,
                        costs
                    )
                    portfolio.add_position(new_position)
                    return True, new_position

                return True, None
            else:
                new_position = self._create_position(
                    order.symbol,
                    order.side,
                    fill_price,
                    order.quantity,
                    timestamp,
                    costs
                )
                portfolio.add_position(new_position)
                return True, new_position
        else:
            position = self._create_position(
                order.symbol,
                order.side,
                fill_price,
                order.quantity,
                timestamp,
                costs
            )
            portfolio.add_position(position)
            return True, position

    def _create_position(
        self,
        symbol: str,
        side: ActionType,
        price: Decimal,
        quantity: Decimal,
        timestamp: datetime,
        costs: dict
    ) -> Position:
        self.position_id_counter += 1
        return Position(
            id=f"POS_{self.position_id_counter:06d}",
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            entry_time=timestamp,
            commission=costs["fee"],
            slippage=costs["slippage"]
        )

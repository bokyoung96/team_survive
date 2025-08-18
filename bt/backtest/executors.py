from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from backtest.models import Order, Position, Portfolio
from backtest.types import ActionType, OrderType, OrderStatus, TransactionCost
from backtest.validators import OrderValidator
from backtest.logger import get_logger


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
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        side, quantity = self._get_order_params(signal_type, portfolio, symbol)
        
        if side is None:
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
    
    def _get_order_params(
        self, 
        signal_type: ActionType, 
        portfolio: Portfolio, 
        symbol: str
    ) -> Tuple[Optional[ActionType], Decimal]:
        if signal_type in [ActionType.SELL, ActionType.EXIT, ActionType.CLOSE]:
            position = portfolio.get_position(symbol)
            if not position or not position.is_open:
                return None, Decimal("0")
            
            quantity = position.open_quantity
            side = ActionType.SELL if position.side == ActionType.BUY else ActionType.BUY
            return side, quantity
        
        elif signal_type in [ActionType.BUY, ActionType.ENTRY]:
            side = ActionType.BUY if signal_type == ActionType.BUY else ActionType.SELL
            quantity = Decimal("-1")
            return side, quantity
        
        return None, Decimal("0")


class QuantityCalculator:
    def __init__(self, logger=None):
        self.logger = logger or get_logger(__name__)
    
    def calculate_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        if order.quantity != Decimal("-1"):
            return True, order.quantity
        
        if order.side == ActionType.BUY:
            return self._calculate_buy_quantity(order, portfolio, fill_price)
        elif order.side == ActionType.SELL:
            return self._calculate_sell_quantity(order, portfolio)
        
        return False, Decimal("0")
    
    def _calculate_buy_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        max_cash_usage = Decimal("0.95")
        
        if "position_sizing" in order.metadata:
            position_sizing = order.metadata["position_sizing"]
            entry_count = order.metadata.get("entry_count", 1)
            
            if entry_count > position_sizing["max_entries"]:
                self.logger.warning(f"Entry count exceeds max: {entry_count}")
                return False, Decimal("0")
            
            initial_size = Decimal(str(position_sizing["initial_size"]))
            scale_factor = Decimal(str(position_sizing["scale_factor"]))
            size_percent = initial_size * (scale_factor ** (entry_count - 1))
            quantity = portfolio.get_total_value() * size_percent / fill_price
        else:
            quantity = portfolio.cash * max_cash_usage / fill_price
        
        # Ensure we don't exceed available cash
        max_quantity = portfolio.cash * max_cash_usage / fill_price
        quantity = min(quantity, max_quantity)
        
        return (True, quantity) if quantity > 0 else (False, Decimal("0"))
    
    def _calculate_sell_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio
    ) -> Tuple[bool, Decimal]:
        position = portfolio.get_position(order.symbol)
        if not position or not position.is_open:
            return False, Decimal("0")
        
        return True, position.open_quantity
    


class PositionCreator:
    def __init__(self):
        self.position_id_counter = 0
    
    def create_position(
        self,
        symbol: str,
        side: ActionType,
        price: Decimal,
        quantity: Decimal,
        timestamp: datetime,
        costs: Dict[str, Decimal],
        metadata: Optional[Dict[str, Any]] = None
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
            slippage=costs["slippage"],
            metadata=metadata or {}
        )


class OrderExecutor:
    def __init__(
        self, 
        transaction_cost: Optional[TransactionCost] = None, 
        validator: Optional[OrderValidator] = None
    ):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.validator = validator or OrderValidator()
        self.logger = get_logger(__name__)
        
        self.quantity_calculator = QuantityCalculator(self.logger)
        self.position_creator = PositionCreator()
    
    def execute_order(
        self,
        order: Order,
        market_data: pd.Series,
        portfolio: Portfolio,
        timestamp: datetime
    ) -> Tuple[bool, Optional[Position]]:
        fill_price = Decimal(str(market_data["close"]))
        
        is_valid, error_msg = self.validator.validate_order(
            order, portfolio, market_data, timestamp
        )
        if not is_valid:
            order.status = OrderStatus.REJECTED
            self.logger.warning(f"Order validation failed: {error_msg}")
            return False, None
        
        if order.quantity == Decimal("-1"):
            success, calculated_quantity = self.quantity_calculator.calculate_quantity(
                order, portfolio, fill_price
            )
            if not success:
                order.status = OrderStatus.REJECTED
                return False, None
            
            order.quantity = calculated_quantity
            
            is_valid, error_msg = self.validator.validate_order(
                order, portfolio, market_data, timestamp
            )
            if not is_valid:
                order.status = OrderStatus.REJECTED
                self.logger.warning(f"Validation failed after quantity calc: {error_msg}")
                return False, None
        
        costs = self.transaction_cost.calculate_cost(
            fill_price, order.quantity, is_maker=False
        )
        
        order.fill(
            price=fill_price,
            quantity=order.quantity,
            commission=costs["fee"],
            slippage=costs["slippage"]
        )
        
        success, position = self._handle_position(
            order, portfolio, fill_price, timestamp, costs
        )
        
        if success and order.side == ActionType.SELL:
            existing_position = portfolio.get_position(order.symbol)
            if not existing_position or not existing_position.is_open:
                self.validator.on_position_closed(order.symbol)
        
        return success, position
    
    def _handle_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal]
    ) -> Tuple[bool, Optional[Position]]:
        existing_position = portfolio.get_position(order.symbol)
        
        if not existing_position or not existing_position.is_open:
            return self._create_new_position(
                order, portfolio, fill_price, timestamp, costs
            )
        
        if order.side != existing_position.side:
            return self._handle_opposite_position(
                order, portfolio, existing_position, fill_price, timestamp, costs
            )
        
        return self._add_to_position(
            order, portfolio, existing_position, fill_price, timestamp, costs
        )
    
    def _create_new_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal]
    ) -> Tuple[bool, Position]:
        position = self.position_creator.create_position(
            order.symbol, order.side, fill_price, order.quantity,
            timestamp, costs, order.metadata
        )
        
        signal_period_id = order.metadata.get("signal_period_id", "unknown")
        entry_count = order.metadata.get("entry_count", 1)
        self.logger.info(
            f"ENTRY #{entry_count}: Opening new position @ ${fill_price} "
            f"[Period: {signal_period_id}]"
        )
        
        portfolio.add_position(position)
        return True, position
    
    def _handle_opposite_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        existing_position: Position, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal]
    ) -> Tuple[bool, Optional[Position]]:
        close_quantity = min(order.quantity, existing_position.open_quantity)
        
        signal_period_id = existing_position.metadata.get("signal_period_id", "unknown")
        entry_count = existing_position.metadata.get("entry_count", 0)
        self.logger.info(
            f"EXIT: Closing position with {entry_count} entries @ ${fill_price} "
            f"[Period: {signal_period_id}]"
        )
        
        portfolio.close_position(
            existing_position, fill_price, close_quantity,
            timestamp, costs["fee"], costs["slippage"]
        )
        
        if order.quantity > close_quantity:
            remaining = order.quantity - close_quantity
            new_position = self.position_creator.create_position(
                order.symbol, order.side, fill_price, remaining,
                timestamp, costs, order.metadata
            )
            portfolio.add_position(new_position)
            return True, new_position
        
        return True, None
    
    def _add_to_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        existing_position: Position, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal]
    ) -> Tuple[bool, Position]:
        entry_count = order.metadata.get("entry_count", 1)
        signal_period_id = order.metadata.get("signal_period_id", "unknown")
        
        self.logger.info(
            f"ENTRY #{entry_count}: Adding to position @ ${fill_price} "
            f"[Period: {signal_period_id}]"
        )
        
        # Calculate new average entry price
        open_qty = existing_position.open_quantity
        total_cost = existing_position.entry_price * open_qty + fill_price * order.quantity
        total_quantity = open_qty + order.quantity
        
        # Update position
        existing_position.entry_price = total_cost / total_quantity
        existing_position.quantity = existing_position.closed_quantity + total_quantity
        existing_position.commission += costs["fee"]
        existing_position.slippage += costs["slippage"]
        existing_position.entry_time = timestamp
        
        if order.metadata:
            existing_position.metadata.update(order.metadata)
        
        # Update portfolio cash
        total_cost = order.quantity * fill_price + costs["fee"] + costs["slippage"]
        if order.side == ActionType.BUY:
            if portfolio.cash < total_cost:
                raise ValueError(f"Insufficient cash: {portfolio.cash} < {total_cost}")
            portfolio.cash -= total_cost
        else:
            portfolio.cash += order.quantity * fill_price - costs["fee"] - costs["slippage"]
        
        return True, existing_position
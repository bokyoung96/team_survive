from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Optional, Tuple, Dict, Any
import pandas as pd

from backtest.models import (
    Order,
    Position,
    Portfolio
)
from backtest.types import ActionType, OrderType, OrderStatus, TransactionCost


class SignalTypeHandler(ABC):
    """Abstract handler for different signal types."""
    
    @abstractmethod
    def can_handle(self, signal_type: ActionType) -> bool:
        """Check if this handler can process the given signal type."""
        pass
    
    @abstractmethod
    def create_order_params(
        self, 
        signal_type: ActionType, 
        portfolio: Portfolio, 
        symbol: str
    ) -> Optional[Tuple[ActionType, Decimal]]:
        """Create order parameters (side, quantity) for the signal."""
        pass


class ExitSignalHandler(SignalTypeHandler):
    """Handles exit/close signal types."""
    
    def can_handle(self, signal_type: ActionType) -> bool:
        return signal_type in [ActionType.SELL, ActionType.EXIT, ActionType.CLOSE]
    
    def create_order_params(
        self, 
        signal_type: ActionType, 
        portfolio: Portfolio, 
        symbol: str
    ) -> Optional[Tuple[ActionType, Decimal]]:
        position = portfolio.get_position(symbol)
        if not position or not position.is_open:
            return None

        quantity = position.open_quantity
        side = ActionType.SELL if position.side == ActionType.BUY else ActionType.BUY
        return side, quantity


class EntrySignalHandler(SignalTypeHandler):
    """Handles entry/buy signal types."""
    
    def can_handle(self, signal_type: ActionType) -> bool:
        return signal_type in [ActionType.BUY, ActionType.ENTRY]
    
    def create_order_params(
        self, 
        signal_type: ActionType, 
        portfolio: Portfolio, 
        symbol: str
    ) -> Optional[Tuple[ActionType, Decimal]]:
        side = ActionType.BUY if signal_type == ActionType.BUY else ActionType.SELL
        quantity = Decimal("-1")  # Will be calculated later
        return side, quantity


class OrderCreator:
    """Creates orders using chain of responsibility pattern."""
    
    def __init__(self):
        self.order_id_counter = 0
        self.signal_handlers = [
            ExitSignalHandler(),
            EntrySignalHandler()
        ]

    def create_order(
        self,
        signal_type: ActionType,
        strength: float,
        symbol: str,
        timestamp: datetime,
        portfolio: Portfolio,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Optional[Order]:
        """Create order using appropriate signal handler."""
        order_params = self._get_order_parameters(signal_type, portfolio, symbol)
        if not order_params:
            return None
            
        side, quantity = order_params
        
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
    
    def _get_order_parameters(
        self, 
        signal_type: ActionType, 
        portfolio: Portfolio, 
        symbol: str
    ) -> Optional[Tuple[ActionType, Decimal]]:
        """Get order parameters using chain of handlers."""
        for handler in self.signal_handlers:
            if handler.can_handle(signal_type):
                return handler.create_order_params(signal_type, portfolio, symbol)
        return None


class ValidationRule(ABC):
    """Abstract validation rule."""
    
    @abstractmethod
    def validate(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        market_data: pd.Series
    ) -> bool:
        """Validate order against this rule."""
        pass


class DynamicQuantityRule(ValidationRule):
    """Validates orders with dynamic quantity calculation."""
    
    def validate(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        market_data: pd.Series
    ) -> bool:
        # Dynamic quantity orders (-1) are always valid at this stage
        return order.quantity == Decimal("-1")


class PositiveQuantityRule(ValidationRule):
    """Validates that quantity is positive."""
    
    def validate(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        market_data: pd.Series
    ) -> bool:
        return order.quantity > 0


class SufficientCapitalRule(ValidationRule):
    """Validates sufficient capital for buy orders."""
    
    def validate(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        market_data: pd.Series
    ) -> bool:
        if order.side != ActionType.BUY:
            return True
            
        required_capital = order.quantity * Decimal(str(market_data["close"]))
        required_capital *= Decimal("1.05")  # Add 5% buffer
        return portfolio.cash >= required_capital


class OrderValidator:
    """Validates orders using multiple validation rules."""
    
    def __init__(self):
        self.validation_rules = [
            DynamicQuantityRule(),
            PositiveQuantityRule(),
            SufficientCapitalRule()
        ]

    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        market_data: pd.Series
    ) -> bool:
        """Validate order using all validation rules."""
        # Special case: dynamic quantity orders bypass normal validation
        if order.quantity == Decimal("-1"):
            return True
        
        # For regular orders, all non-dynamic rules must pass
        regular_rules = [r for r in self.validation_rules 
                        if not isinstance(r, DynamicQuantityRule)]
        
        for rule in regular_rules:
            if not rule.validate(order, portfolio, market_data):
                return False
                
        return True


class QuantityCalculator(ABC):
    """Abstract quantity calculator."""
    
    @abstractmethod
    def calculate_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        """Calculate order quantity. Returns (success, quantity)."""
        pass


class BuyQuantityCalculator(QuantityCalculator):
    """Calculates quantity for buy orders."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def calculate_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        if order.side != ActionType.BUY:
            return True, order.quantity
            
        if "position_sizing" in order.metadata:
            return self._calculate_sized_quantity(order, portfolio, fill_price)
        else:
            return self._calculate_default_quantity(order, portfolio, fill_price)
    
    def _calculate_sized_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        """Calculate quantity based on position sizing rules."""
        position_sizing = order.metadata["position_sizing"]
        entry_count = order.metadata.get("entry_count", 1)
        
        if entry_count > position_sizing["max_entries"]:
            self.logger.warning(
                f"Entry count {entry_count} exceeds max entries {position_sizing['max_entries']}"
            )
            return False, Decimal("0")
        
        initial_size = Decimal(str(position_sizing["initial_size"]))
        scale_factor = Decimal(str(position_sizing["scale_factor"]))
        size_percent = initial_size * (scale_factor ** (entry_count - 1))
        position_value = portfolio.get_total_value() * size_percent
        quantity = position_value / fill_price
        
        if quantity <= 0:
            self.logger.warning(f"Invalid position size calculated: {quantity}")
            return False, Decimal("0")
        
        # Apply capital constraint
        max_quantity = portfolio.cash * Decimal("0.95") / fill_price
        if quantity > max_quantity:
            quantity = max_quantity
            self.logger.debug(f"Position size adjusted to available capital: {quantity}")
        
        self._log_entry_info(order, quantity, size_percent, fill_price, entry_count)
        return True, quantity
    
    def _calculate_default_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        """Calculate quantity using default logic."""
        available_capital = portfolio.cash * Decimal("0.95")
        quantity = available_capital / fill_price
        
        if quantity <= 0:
            self.logger.warning(f"Insufficient capital for position: {available_capital}")
            return False, Decimal("0")
            
        return True, quantity
    
    def _log_entry_info(
        self, 
        order: Order, 
        quantity: Decimal, 
        size_percent: Decimal, 
        fill_price: Decimal, 
        entry_count: int
    ) -> None:
        """Log entry information."""
        total_entries = order.metadata.get("total_entries", entry_count)
        signal_period_id = order.metadata.get("signal_period_id", "unknown")
        self.logger.info(
            f"Entry #{entry_count} (Total: #{total_entries}): "
            f"Size {float(size_percent)*100:.2f}% = {quantity:.6f} @ ${fill_price} "
            f"[Period: {signal_period_id}]"
        )


class SellQuantityCalculator(QuantityCalculator):
    """Calculates quantity for sell orders."""
    
    def calculate_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        if order.side != ActionType.SELL:
            return True, order.quantity
            
        existing_position = portfolio.get_position(order.symbol)
        if existing_position and existing_position.is_open:
            return True, existing_position.open_quantity
        else:
            return False, Decimal("0")


class PositionHandler(ABC):
    """Abstract position handler for different scenarios."""
    
    @abstractmethod
    def can_handle(
        self, 
        existing_position: Optional[Position], 
        order: Order
    ) -> bool:
        """Check if this handler can process the scenario."""
        pass
    
    @abstractmethod
    def handle_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal],
        position_creator
    ) -> Tuple[bool, Optional[Position]]:
        """Handle the position scenario."""
        pass


class NewPositionHandler(PositionHandler):
    """Handles creation of new positions."""
    
    def can_handle(
        self, 
        existing_position: Optional[Position], 
        order: Order
    ) -> bool:
        return existing_position is None or not existing_position.is_open
    
    def handle_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal],
        position_creator
    ) -> Tuple[bool, Optional[Position]]:
        position = position_creator.create_position(
            order.symbol, order.side, fill_price, order.quantity,
            timestamp, costs, order.metadata
        )
        portfolio.add_position(position)
        return True, position


class OppositePositionHandler(PositionHandler):
    """Handles orders opposite to existing position (closing/reversing)."""
    
    def __init__(self, logger=None):
        from backtest.logger import get_logger
        self.logger = logger or get_logger(__name__)
    
    def can_handle(
        self, 
        existing_position: Optional[Position], 
        order: Order
    ) -> bool:
        return (existing_position is not None and 
                existing_position.is_open and 
                order.side != existing_position.side)
    
    def handle_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal],
        position_creator
    ) -> Tuple[bool, Optional[Position]]:
        existing_position = portfolio.get_position(order.symbol)
        
        close_quantity = min(order.quantity, existing_position.open_quantity)
        
        # Log exit information
        signal_period_id = existing_position.metadata.get("signal_period_id", "unknown")
        total_entries = existing_position.metadata.get("total_entries", 0)
        self.logger.info(
            f"EXIT: Closing position with {total_entries} entries @ ${fill_price} "
            f"[Period: {signal_period_id}]"
        )
        
        portfolio.close_position(
            existing_position, fill_price, close_quantity,
            timestamp, costs["fee"], costs["slippage"]
        )

        # Create new position if there's remaining quantity
        if order.quantity > close_quantity:
            remaining = order.quantity - close_quantity
            new_position = position_creator.create_position(
                order.symbol, order.side, fill_price, remaining,
                timestamp, costs, order.metadata
            )
            portfolio.add_position(new_position)
            return True, new_position

        return True, None


class SameSidePositionHandler(PositionHandler):
    """Handles orders in same direction as existing position (adding)."""
    
    def can_handle(
        self, 
        existing_position: Optional[Position], 
        order: Order
    ) -> bool:
        return (existing_position is not None and 
                existing_position.is_open and 
                order.side == existing_position.side)
    
    def handle_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal],
        position_creator
    ) -> Tuple[bool, Optional[Position]]:
        existing_position = portfolio.get_position(order.symbol)
        
        # Update position with new entry
        self._update_existing_position(
            existing_position, order, fill_price, timestamp, costs
        )
        
        # Update portfolio cash
        self._update_portfolio_cash(portfolio, order, fill_price, costs)
        
        return True, existing_position
    
    def _update_existing_position(
        self, 
        position: Position, 
        order: Order, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal]
    ) -> None:
        """Update existing position with new entry."""
        open_qty = position.open_quantity
        total_cost = position.entry_price * open_qty
        new_cost = fill_price * order.quantity
        total_quantity = open_qty + order.quantity
        
        position.entry_price = (total_cost + new_cost) / total_quantity
        position.quantity = position.closed_quantity + total_quantity
        position.commission += costs["fee"]
        position.slippage += costs["slippage"]
        position.entry_time = timestamp
        
        if order.metadata:
            position.metadata.update(order.metadata)
    
    def _update_portfolio_cash(
        self, 
        portfolio: Portfolio, 
        order: Order, 
        fill_price: Decimal, 
        costs: Dict[str, Decimal]
    ) -> None:
        """Update portfolio cash based on order side."""
        if order.side == ActionType.BUY:
            total_cost = order.quantity * fill_price + costs["fee"] + costs["slippage"]
            if portfolio.cash < total_cost:
                raise ValueError(f"Insufficient cash: {portfolio.cash} < {total_cost}")
            portfolio.cash -= total_cost
        elif order.side == ActionType.SELL:
            portfolio.cash += (order.quantity * fill_price - costs["fee"] - costs["slippage"])
        else:
            raise ValueError(f"Invalid order side: {order.side}")


class PositionCreator:
    """Creates new position objects."""
    
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
        """Create a new position."""
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
    """Executes orders with improved structure and separation of concerns."""
    
    def __init__(self, transaction_cost: Optional[TransactionCost] = None):
        self.transaction_cost = transaction_cost or TransactionCost()
        from backtest.logger import get_logger
        self._logger = get_logger(__name__)
        
        # Initialize components
        self.position_creator = PositionCreator()
        
        # Initialize quantity calculators
        self.quantity_calculators = [
            BuyQuantityCalculator(self._logger),
            SellQuantityCalculator()
        ]
        
        # Initialize position handlers
        self.position_handlers = [
            OppositePositionHandler(self._logger),
            SameSidePositionHandler(),
            NewPositionHandler()  # This should be last (fallback)
        ]

    def execute_order(
        self,
        order: Order,
        market_data: pd.Series,
        portfolio: Portfolio,
        timestamp: datetime
    ) -> Tuple[bool, Optional[Position]]:
        """Execute order with improved structure."""
        fill_price = Decimal(str(market_data["close"]))
        
        # Calculate quantity if dynamic (-1)
        if order.quantity == Decimal("-1"):
            success, calculated_quantity = self._calculate_dynamic_quantity(
                order, portfolio, fill_price
            )
            if not success:
                return False, None
            order.quantity = calculated_quantity
        
        # Validate sufficient capital for buy orders
        if not self._validate_sufficient_capital(order, portfolio, fill_price):
            order.status = OrderStatus.REJECTED
            return False, None
        
        # Calculate transaction costs
        costs = self.transaction_cost.calculate_cost(
            fill_price, order.quantity, is_maker=False
        )
        
        # Fill the order
        order.fill(
            price=fill_price,
            quantity=order.quantity,
            commission=costs["fee"],
            slippage=costs["slippage"]
        )
        
        # Handle position based on scenario
        return self._handle_position(
            order, portfolio, fill_price, timestamp, costs
        )
    
    def _calculate_dynamic_quantity(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> Tuple[bool, Decimal]:
        """Calculate quantity for dynamic orders."""
        for calculator in self.quantity_calculators:
            success, quantity = calculator.calculate_quantity(order, portfolio, fill_price)
            if not success:
                return False, Decimal("0")
            if quantity != order.quantity:  # Calculator modified the quantity
                return True, quantity
                
        return True, order.quantity
    
    def _validate_sufficient_capital(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal
    ) -> bool:
        """Validate sufficient capital for buy orders."""
        if order.side != ActionType.BUY:
            return True
            
        costs = self.transaction_cost.calculate_cost(
            fill_price, order.quantity, is_maker=False
        )
        total_cost = (fill_price * order.quantity) + costs["total"]
        return portfolio.cash >= total_cost
    
    def _handle_position(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        fill_price: Decimal, 
        timestamp: datetime, 
        costs: Dict[str, Decimal]
    ) -> Tuple[bool, Optional[Position]]:
        """Handle position based on current state and order."""
        existing_position = portfolio.get_position(order.symbol)
        
        for handler in self.position_handlers:
            if handler.can_handle(existing_position, order):
                return handler.handle_position(
                    order, portfolio, fill_price, timestamp, 
                    costs, self.position_creator
                )
        
        # Should never reach here if handlers are comprehensive
        raise ValueError(f"No handler found for position scenario: {existing_position}, {order}")
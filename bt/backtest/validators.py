from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, Any, Optional, Set, Tuple
from datetime import datetime
import pandas as pd

from backtest.models import Order, Portfolio, Position
from backtest.types import ActionType, OrderStatus


@dataclass
class MarketRules:
    """Market-specific trading rules and constraints."""
    min_order_size: Decimal = Decimal("0.000001")  # More flexible for small orders
    max_order_size: Decimal = Decimal("1000000")
    min_notional: Decimal = Decimal("1")  # Reduced from 10 to 1
    max_leverage: Decimal = Decimal("10")
    tick_size: Decimal = Decimal("0.01")
    lot_size: Decimal = Decimal("0.000001")  # More flexible lot size
    max_orders_per_bar: int = 10
    cooldown_bars: int = 0
    trading_hours: Optional[Tuple[int, int]] = None  # (start_hour, end_hour) in 24h format
    allowed_symbols: Optional[Set[str]] = None
    

@dataclass
class ValidationState:
    """Maintains validation state across orders."""
    executed_signal_ids: Set[str] = None
    bar_order_count: Dict[str, int] = None
    last_entry_bar: Dict[str, pd.Timestamp] = None
    position_entries: Dict[str, int] = None  # Track entry count per position
    max_entries_per_position: int = 10  # Allow up to 10 martingale entries
    
    def __post_init__(self):
        if self.executed_signal_ids is None:
            self.executed_signal_ids = set()
        if self.bar_order_count is None:
            self.bar_order_count = {}
        if self.last_entry_bar is None:
            self.last_entry_bar = {}
        if self.position_entries is None:
            self.position_entries = {}
    
    def reset_position_entries(self, symbol: str):
        """Reset entry count when position is closed."""
        if symbol in self.position_entries:
            del self.position_entries[symbol]
    
    def increment_position_entries(self, symbol: str) -> int:
        """Increment and return entry count for position."""
        if symbol not in self.position_entries:
            self.position_entries[symbol] = 0
        self.position_entries[symbol] += 1
        return self.position_entries[symbol]
    
    def get_position_entries(self, symbol: str) -> int:
        """Get current entry count for position."""
        return self.position_entries.get(symbol, 0)


class OrderValidator:
    """Centralized order validation with all rules in one place."""
    
    def __init__(
        self, 
        market_rules: Optional[MarketRules] = None,
        validation_state: Optional[ValidationState] = None
    ):
        self.market_rules = market_rules or MarketRules()
        self.state = validation_state or ValidationState()
        self._validation_errors = []
    
    def validate_order(
        self,
        order: Order,
        portfolio: Portfolio,
        market_data: pd.Series,
        timestamp: Optional[pd.Timestamp] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate order against all rules.
        Returns (is_valid, error_message).
        """
        self._validation_errors = []
        
        # Skip validation for dynamic quantity orders (will be calculated later)
        if order.quantity == Decimal("-1"):
            return True, None
        
        # Run all validation checks
        checks = [
            self._check_symbol_allowed(order),
            self._check_trading_hours(order, timestamp),
            self._check_quantity_limits(order),
            self._check_notional_limits(order, market_data),
            self._check_position_consistency(order, portfolio),
            self._check_sufficient_capital(order, portfolio, market_data),
            self._check_max_entries(order, portfolio),
            self._check_duplicate_signal(order, timestamp),
            self._check_bar_order_limit(order, timestamp),
            self._check_cooldown_period(order, timestamp)
        ]
        
        # All checks must pass
        if all(checks):
            # Update state for successful validation
            self._update_validation_state(order, timestamp)
            return True, None
        else:
            error_msg = "; ".join(self._validation_errors)
            return False, error_msg
    
    def _check_symbol_allowed(self, order: Order) -> bool:
        """Check if symbol is in allowed list."""
        if self.market_rules.allowed_symbols:
            if order.symbol not in self.market_rules.allowed_symbols:
                self._validation_errors.append(f"Symbol {order.symbol} not allowed")
                return False
        return True
    
    def _check_trading_hours(self, order: Order, timestamp: Optional[pd.Timestamp]) -> bool:
        """Check if order is within trading hours."""
        if self.market_rules.trading_hours and timestamp:
            hour = timestamp.hour
            start_hour, end_hour = self.market_rules.trading_hours
            if not (start_hour <= hour < end_hour):
                self._validation_errors.append(
                    f"Outside trading hours ({start_hour}-{end_hour})"
                )
                return False
        return True
    
    def _check_quantity_limits(self, order: Order) -> bool:
        """Check order quantity against min/max limits."""
        if order.quantity < self.market_rules.min_order_size:
            self._validation_errors.append(
                f"Order size {order.quantity} below minimum {self.market_rules.min_order_size}"
            )
            return False
        
        if order.quantity > self.market_rules.max_order_size:
            self._validation_errors.append(
                f"Order size {order.quantity} exceeds maximum {self.market_rules.max_order_size}"
            )
            return False
        
        # Skip lot size validation - allow any quantity
        
        return True
    
    def _check_notional_limits(self, order: Order, market_data: pd.Series) -> bool:
        """Check order notional value against minimum."""
        price = Decimal(str(market_data['close']))
        notional = order.quantity * price
        
        if notional < self.market_rules.min_notional:
            self._validation_errors.append(
                f"Notional {notional} below minimum {self.market_rules.min_notional}"
            )
            return False
        
        return True
    
    def _check_position_consistency(self, order: Order, portfolio: Portfolio) -> bool:
        """Check order consistency with existing position."""
        position = portfolio.get_position(order.symbol)
        
        # No position - any order is valid
        if not position or not position.is_open:
            return True
        
        # Check for conflicting close orders
        if order.side == ActionType.SELL and position.side == ActionType.SELL:
            self._validation_errors.append(
                "Cannot sell when already in short position"
            )
            return False
        
        if order.side == ActionType.BUY and position.side == ActionType.BUY:
            # This is adding to position, which is allowed
            pass
        
        return True
    
    def _check_sufficient_capital(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        market_data: pd.Series
    ) -> bool:
        """Check if portfolio has sufficient capital for order."""
        if order.side != ActionType.BUY:
            return True
        
        price = Decimal(str(market_data['close']))
        required_capital = order.quantity * price * Decimal("1.05")  # 5% buffer
        
        if portfolio.cash < required_capital:
            self._validation_errors.append(
                f"Insufficient capital: {portfolio.cash} < {required_capital}"
            )
            return False
        
        return True
    
    def _check_max_entries(self, order: Order, portfolio: Portfolio) -> bool:
        """Check if max entries per position exceeded."""
        position = portfolio.get_position(order.symbol)
        
        # Only check for entry orders
        if order.side == ActionType.BUY:
            # Get current entries from position metadata (the source of truth)
            current_entries = 0
            if position and position.is_open:
                current_entries = position.metadata.get('entry_count', 1)
            
            # Get max_entries from order metadata (from strategy) or use default
            max_entries = self.state.max_entries_per_position  # default
            if order.metadata and 'position_sizing' in order.metadata:
                max_entries = order.metadata['position_sizing'].get('max_entries', max_entries)
            
            # Check if we've reached the limit
            if position and position.is_open and position.side == order.side:
                if current_entries >= max_entries:
                    self._validation_errors.append(
                        f"Max entries ({max_entries}) reached for position with {current_entries} entries"
                    )
                    return False
        
        return True
    
    def _check_duplicate_signal(self, order: Order, timestamp: Optional[pd.Timestamp]) -> bool:
        """Check for duplicate signal execution."""
        if timestamp and order.metadata:
            signal_id = order.metadata.get('signal_id')
            if signal_id:
                # Create idempotency key
                idempotency_key = f"{signal_id}|{timestamp}|{order.symbol}"
                
                if idempotency_key in self.state.executed_signal_ids:
                    self._validation_errors.append(
                        f"Duplicate signal {signal_id} at {timestamp}"
                    )
                    return False
        
        return True
    
    def _check_bar_order_limit(self, order: Order, timestamp: Optional[pd.Timestamp]) -> bool:
        """Check if bar order limit exceeded."""
        if timestamp:
            bar_key = f"{order.symbol}|{timestamp}"
            current_count = self.state.bar_order_count.get(bar_key, 0)
            
            if current_count >= self.market_rules.max_orders_per_bar:
                self._validation_errors.append(
                    f"Max orders per bar ({self.market_rules.max_orders_per_bar}) exceeded"
                )
                return False
        
        return True
    
    def _check_cooldown_period(self, order: Order, timestamp: Optional[pd.Timestamp]) -> bool:
        """Check if cooldown period has passed since last entry."""
        if self.market_rules.cooldown_bars > 0 and timestamp:
            if order.side == ActionType.BUY:  # Entry order
                last_entry = self.state.last_entry_bar.get(order.symbol)
                if last_entry:
                    # Simple bar count check (would need proper bar counting in production)
                    # For now, just check if it's the same timestamp
                    if last_entry == timestamp:
                        self._validation_errors.append(
                            f"Cooldown period ({self.market_rules.cooldown_bars} bars) not met"
                        )
                        return False
        
        return True
    
    def _update_validation_state(self, order: Order, timestamp: Optional[pd.Timestamp]):
        """Update validation state after successful validation."""
        if timestamp:
            # Update bar order count
            bar_key = f"{order.symbol}|{timestamp}"
            self.state.bar_order_count[bar_key] = self.state.bar_order_count.get(bar_key, 0) + 1
            
            # Update signal tracking
            if order.metadata:
                signal_id = order.metadata.get('signal_id')
                if signal_id:
                    idempotency_key = f"{signal_id}|{timestamp}|{order.symbol}"
                    self.state.executed_signal_ids.add(idempotency_key)
            
            # Update entry tracking
            if order.side == ActionType.BUY:
                self.state.last_entry_bar[order.symbol] = timestamp
                self.state.increment_position_entries(order.symbol)
    
    def on_position_closed(self, symbol: str):
        """Called when a position is closed to reset entry count."""
        self.state.reset_position_entries(symbol)
    
    def reset_bar_counts(self):
        """Reset per-bar order counts (call at start of new bar)."""
        self.state.bar_order_count.clear()
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get current validation state statistics."""
        return {
            'executed_signals': len(self.state.executed_signal_ids),
            'position_entries': dict(self.state.position_entries),
            'active_bars': len(self.state.bar_order_count)
        }
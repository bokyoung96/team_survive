from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass
from decimal import Decimal
from collections import deque

from backtest.types import ActionType, Signal


@dataclass
class TradingContext:
    """
    Context object passed to strategies containing current market state
    - portfolio: Portfolio object
    - position: Current position for the symbol
    - bar_index: Current bar index in the dataset
    - timestamp: Current timestamp
    - current_bar: Current OHLCV bar
    - lookback_data: Historical data with limited lookback
    - symbol: Trading symbol
    """
    portfolio: Any
    position: Optional[Any]
    bar_index: int
    timestamp: datetime
    current_bar: pd.Series
    lookback_data: pd.DataFrame
    symbol: str
    
    def get_current_price(self) -> Decimal:
        return Decimal(str(self.current_bar['close']))


class StreamingStrategy(ABC):
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None, lookback_periods: int = 200):
        self.name = name
        self.parameters = parameters or {}
        self.lookback_periods = lookback_periods
        self._trade_history: deque = deque(maxlen=1000)  # Store actual trade results
        self._indicator_cache: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def update_indicators(self, context: TradingContext) -> None:
        pass
    
    @abstractmethod
    def process_bar(self, context: TradingContext) -> Optional[Signal]:
        pass
    
    def reset_state(self) -> None:
        self._trade_history.clear()
        self._indicator_cache.clear()
        self._state.clear()
    
    def get_indicator_value(self, name: str, default: Any = None) -> Any:
        return self._indicator_cache.get(name, default)
    
    def set_indicator_value(self, name: str, value: Any) -> None:
        self._indicator_cache[name] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        self._state[key] = value

    def _get_lookback_data(self, full_data: pd.DataFrame, current_index: int) -> pd.DataFrame:
        start_idx = max(0, current_index - self.lookback_periods + 1)
        return full_data.iloc[start_idx:current_index + 1]


    def record_signal(self, timestamp: datetime, signal: Signal) -> None:
        """Backward compatibility - just pass for now"""
        pass

    def record_trade(self, timestamp: datetime, trade_data: Dict[str, Any]) -> None:
        """Record actual trade execution data"""
        self._trade_history.append((timestamp, trade_data))

    @property
    def signals_history(self) -> List:
        """Backward compatibility - return empty list"""
        return []

    @property
    def trade_history(self) -> List[Tuple[datetime, Dict[str, Any]]]:
        """Get actual trade execution history"""
        return list(self._trade_history)


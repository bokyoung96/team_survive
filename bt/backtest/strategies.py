from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
from dataclasses import dataclass
from decimal import Decimal

from backtest.types import ActionType, Signal


@dataclass
class TradingContext:
    """Context object passed to strategies containing current market state"""
    portfolio: Any  # Portfolio object
    position: Optional[Any]  # Current position for the symbol
    bar_index: int  # Current bar index in the dataset
    timestamp: datetime  # Current timestamp
    current_bar: pd.Series  # Current OHLCV bar
    lookback_data: pd.DataFrame  # Historical data with limited lookback
    symbol: str  # Trading symbol
    
    def get_current_price(self) -> Decimal:
        """Get current close price as Decimal"""
        return Decimal(str(self.current_bar['close']))


class StreamingStrategy(ABC):
    """Base class for streaming strategies that process bars sequentially"""
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None, lookback_periods: int = 200):
        self.name = name
        self.parameters = parameters or {}
        self.lookback_periods = lookback_periods
        self._signals_history: List[Tuple[datetime, Signal]] = []
        self._indicator_cache: Dict[str, Any] = {}
        self._state: Dict[str, Any] = {}

    @abstractmethod
    def update_indicators(self, context: TradingContext) -> None:
        """Update indicators incrementally with new bar data"""
        pass
    
    @abstractmethod
    def process_bar(self, context: TradingContext) -> Optional[Signal]:
        """Process a single bar and return signal if any"""
        pass
    
    def reset_state(self) -> None:
        """Reset strategy state for new backtest run"""
        self._signals_history.clear()
        self._indicator_cache.clear()
        self._state.clear()
    
    def get_indicator_value(self, name: str, default: Any = None) -> Any:
        """Get cached indicator value"""
        return self._indicator_cache.get(name, default)
    
    def set_indicator_value(self, name: str, value: Any) -> None:
        """Cache indicator value"""
        self._indicator_cache[name] = value
    
    def get_state(self, key: str, default: Any = None) -> Any:
        """Get strategy state value"""
        return self._state.get(key, default)
    
    def set_state(self, key: str, value: Any) -> None:
        """Set strategy state value"""
        self._state[key] = value

    def _get_lookback_data(self, full_data: pd.DataFrame, current_index: int) -> pd.DataFrame:
        """Get limited lookback data for current bar"""
        start_idx = max(0, current_index - self.lookback_periods + 1)
        return full_data.iloc[start_idx:current_index + 1].copy()

    def validate_signal(
        self,
        signal: Signal,
        context: TradingContext
    ) -> bool:
        """Validate signal before execution"""
        return True


    def record_signal(self, timestamp: datetime, signal: Signal) -> None:
        self._signals_history.append((timestamp, signal))

    @property
    def signals_history(self) -> List[Tuple[datetime, Signal]]:
        return self._signals_history.copy()


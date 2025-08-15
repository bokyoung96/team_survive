from abc import ABC, abstractmethod
from datetime import datetime
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd

from backtest.types import ActionType, Signal


class Strategy(ABC):
    def __init__(self, name: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.parameters = parameters or {}
        self._signals_history: List[Tuple[datetime, Signal]] = []

    @abstractmethod
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        pass

    def generate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_with_indicators = self.calculate_indicators(data)

        signals = []
        for i in range(len(data_with_indicators)):
            if i < 50:
                continue

            current_data = data_with_indicators.iloc[:i+1]
            signal = self.generate_signal(current_data)

            if signal:
                signals.append({
                    'timestamp': current_data.index[-1],
                    'type': signal.type.value,
                    'strength': signal.strength,
                    'metadata': signal.metadata
                })
        return pd.DataFrame(signals).set_index('timestamp') if signals else pd.DataFrame()

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: Optional[Any] = None,
        portfolio: Optional[Any] = None
    ) -> Optional[Signal]:
        pass

    def validate_signal(
        self,
        signal: Signal,
        data: pd.DataFrame,
        position: Optional[Any] = None,
        portfolio: Optional[Any] = None
    ) -> bool:
        return True

    def record_signal(self, timestamp: datetime, signal: Signal) -> None:
        self._signals_history.append((timestamp, signal))

    @property
    def signals_history(self) -> List[Tuple[datetime, Signal]]:
        return self._signals_history.copy()


class SignalCombiner:
    def __init__(self, strategies: List[Strategy], weights: Optional[Dict[str, float]] = None):
        self.strategies = strategies
        self.weights = weights or {s.name: 1.0 for s in strategies}

    def combine_signals(
        self,
        signals: Dict[str, Signal]
    ) -> Optional[Signal]:
        if not signals:
            return None

        buy_strength = 0.0
        sell_strength = 0.0

        for strategy_name, signal in signals.items():
            weight = self.weights.get(strategy_name, 1.0)

            if signal.type == ActionType.BUY:
                buy_strength += signal.strength * weight
            elif signal.type == ActionType.SELL:
                sell_strength += signal.strength * weight

        total_weight = sum(self.weights.values())
        buy_strength /= total_weight
        sell_strength /= total_weight

        if buy_strength > sell_strength and buy_strength > 0.5:
            return Signal(
                type=ActionType.BUY,
                strength=buy_strength,
                metadata={"combined": True}
            )
        elif sell_strength > buy_strength and sell_strength > 0.5:
            return Signal(
                type=ActionType.SELL,
                strength=sell_strength,
                metadata={"combined": True}
            )

        return None

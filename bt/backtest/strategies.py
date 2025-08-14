from abc import ABC, abstractmethod
from datetime import datetime
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd

from backtest.types import ActionType, Signal
from backtest.timeframe import MultiTimeframeData
from backtest.logger import get_logger
from indicators.indicators import MovingAverage


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


class GoldenCrossStrategy(Strategy):
    def __init__(
        self,
        multi_tf_data: 'MultiTimeframeData',
        ma_periods: Optional[Dict[str, List[int]]] = None,
        position_sizing: Optional[Dict[str, Any]] = None,
        exit_levels: Optional[Dict[str, float]] = None
    ):
        super().__init__("GoldenCrossStrategy")
        self.multi_tf_data = multi_tf_data
        self._logger = get_logger(__name__)
        
        # NOTE: Default parameters
        default_ma_periods = {
            "daily": [112, 224, 448],
            "3min": [360],
            "30min": [60],
            "60min": [60]
        }
        
        default_position_sizing = {
            "initial_size": 0.01,  # 1% of capital
            "scale_factor": 1.4,
            "max_entries": 10
        }
        
        default_exit_levels = {
            "tp1_level": 0.382,  # First take profit at 38.2%
            "tp1_size": 0.5,     # Exit 50% at first level
            "tp2_level": 0.5,    # Second take profit at 50%
            "tp2_size": 0.5      # Exit remaining 50%
        }
        
        self.parameters = {
            "ma_periods": ma_periods or default_ma_periods,
            "position_sizing": position_sizing or default_position_sizing,
            "exit_levels": exit_levels or default_exit_levels
        }
        
        self.entry_count = 0
        self.last_high = None
        self.entry_prices: List[Decimal] = []
    
    def _calculate_ma(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate moving average using MovingAverage indicator class."""
        if len(data) < period:
            return pd.Series(index=data.index, dtype=float)
        
        ma_indicator = MovingAverage(name=f"ma_{period}", length=period)
        ma_df = ma_indicator.calculate(data)
        return ma_df[f"ma_{period}"]
    
    def _check_price_touch(self, price: float, ma_value: float, tolerance: float = 0.001) -> bool:
        if pd.isna(ma_value) or pd.isna(price):
            return False
        if ma_value == 0:
            return False
        diff_pct = abs(price - ma_value) / ma_value
        return diff_pct <= tolerance
    
    def _get_timeframe_ma_from_df(self, df: pd.DataFrame, period: int) -> Optional[float]:
        if df.empty or 'close' not in df.columns:
            return None
        if len(df) < period:
            return None
        ma_series = self._calculate_ma(df, period)
        latest_ma = ma_series.iloc[-1]
        return float(latest_ma) if not pd.isna(latest_ma) else None
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        
        for period in self.parameters["ma_periods"]["daily"]:
            ma_indicator = MovingAverage(name=f"ma_{period}", length=period)
            ma_df = ma_indicator.calculate(data)
            df[f"ma_{period}"] = ma_df[f"ma_{period}"]
        
        return df
    
    def check_golden_cross(self, data: pd.DataFrame) -> bool:
        if len(data) < self.parameters["ma_periods"]["daily"][-1]:
            return False
        
        last_row = data.iloc[-1]
        ma_periods = self.parameters["ma_periods"]["daily"]
        
        # Check MA112 > MA224 > MA448
        ma_112 = last_row.get(f"ma_{ma_periods[0]}")
        ma_224 = last_row.get(f"ma_{ma_periods[1]}")
        ma_448 = last_row.get(f"ma_{ma_periods[2]}")
        
        if pd.isna(ma_112) or pd.isna(ma_224) or pd.isna(ma_448):
            return False
        
        return ma_112 > ma_224 > ma_448
    
    def check_multi_timeframe_touch(self, current_price: float) -> bool:
        tolerance = 0.001
        
        # NOTE: 3min 360 MA check
        if "3m" in self.multi_tf_data:
            ma_value = self._get_timeframe_ma_from_df(self.multi_tf_data["3m"], 360)
            if ma_value is not None and self._check_price_touch(current_price, ma_value, tolerance):
                return True
        
        # NOTE: 30min 60 MA check  
        if "30m" in self.multi_tf_data:
            ma_value = self._get_timeframe_ma_from_df(self.multi_tf_data["30m"], 60)
            if ma_value is not None and self._check_price_touch(current_price, ma_value, tolerance):
                return True
        
        # NOTE: 60min 60 MA check
        if "1h" in self.multi_tf_data:
            ma_value = self._get_timeframe_ma_from_df(self.multi_tf_data["1h"], 60)
            if ma_value is not None and self._check_price_touch(current_price, ma_value, tolerance):
                return True
        
        return False
    
    def calculate_position_size(
        self,
        portfolio_value: Decimal,
        entry_count: int
    ) -> Decimal:
        if entry_count >= self.parameters["position_sizing"]["max_entries"]:
            return Decimal("0")
        
        initial_size = Decimal(str(self.parameters["position_sizing"]["initial_size"]))
        scale_factor = Decimal(str(self.parameters["position_sizing"]["scale_factor"]))
        
        size_percent = initial_size * (scale_factor ** entry_count)
        return portfolio_value * size_percent
    
    def calculate_exit_levels(
        self,
        entry_price: Decimal,
        last_high: Decimal
    ) -> Dict[str, Decimal]:
        price_range = last_high - entry_price
        
        tp1_level = Decimal(str(self.parameters["exit_levels"]["tp1_level"]))
        tp2_level = Decimal(str(self.parameters["exit_levels"]["tp2_level"]))
        
        return {
            "tp1": entry_price + (price_range * tp1_level),
            "tp2": entry_price + (price_range * tp2_level),
            "stop_loss": entry_price * Decimal("0.95")  # 5% stop loss
        }
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: Optional[Any] = None,
        portfolio: Optional[Any] = None
    ) -> Optional[Signal]:
        if data.empty:
            self._logger.warning("Empty data provided to generate_signal")
            return None
        
        last_row = data.iloc[-1]
        current_price = Decimal(str(last_row["close"]))
        
        if self.last_high is None or current_price > self.last_high:
            self.last_high = current_price
        
        # NOTE: Exit logic for existing positions
        if position and position.is_open:
            exit_levels = self.calculate_exit_levels(
                position.entry_price,
                self.last_high
            )
            
            if current_price >= exit_levels["tp2"]:
                self._logger.info(f"Take profit 2 signal at ${current_price} (target: ${exit_levels['tp2']})")
                return Signal(
                    type=ActionType.CLOSE,
                    strength=1.0,
                    price=current_price,
                    metadata={"reason": "take_profit_2"}
                )
            elif current_price >= exit_levels["tp1"]:
                return Signal(
                    type=ActionType.CLOSE,
                    strength=0.5,
                    price=current_price,
                    quantity=position.open_quantity * Decimal("0.5"),
                    metadata={"reason": "take_profit_1"}
                )
            
            if current_price <= exit_levels["stop_loss"]:
                return Signal(
                    type=ActionType.CLOSE,
                    strength=1.0,
                    price=current_price,
                    metadata={"reason": "stop_loss"}
                )
        
        # NOTE: Entry logic: Golden cross (daily) + Multi-timeframe MA touch
        golden_cross = self.check_golden_cross(data)
        multi_tf_touch = self.check_multi_timeframe_touch(float(current_price))
        
        if golden_cross and multi_tf_touch:
            self._logger.info(f"Entry conditions met: Golden cross + Multi-TF touch at ${current_price}")
            
            if portfolio:
                portfolio_value = portfolio.total_value
                position_size = self.calculate_position_size(
                    portfolio_value,
                    self.entry_count
                )
                
                if position_size > 0:
                    quantity = position_size / current_price
                    
                    self.entry_count += 1
                    self.entry_prices.append(current_price)
                    
                    self._logger.info(f"BUY signal generated: ${current_price}, quantity: {quantity}, entry #{self.entry_count}")
                    return Signal(
                        type=ActionType.BUY,
                        strength=0.8,
                        price=current_price,
                        quantity=quantity,
                        metadata={
                            "entry_count": self.entry_count,
                            "golden_cross": True,
                            "multi_timeframe_touch": True
                        }
                    )
        elif golden_cross:
            self._logger.debug("Golden cross detected but no multi-timeframe touch")
        elif multi_tf_touch:
            self._logger.debug("Multi-timeframe touch detected but no golden cross")
        
        return None
    
    def get_strategy_conditions(self) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        # NOTE: Golden cross conditions (daily MA112 > MA224 > MA448)
        golden_cross_conditions = []
        ma_periods = self.parameters["ma_periods"]["daily"]
        for i, period in enumerate(ma_periods):
            golden_cross_conditions.append({
                "type": "golden_cross",
                "timeframe": "daily", 
                "ma_period": period,
                "order": i
            })
        
        # NOTE: Multi-timeframe touch conditions
        touch_conditions = [
            {"timeframe": "3min", "ma_period": 360, "tolerance": 0.001},
            {"timeframe": "30min", "ma_period": 60, "tolerance": 0.001}, 
            {"timeframe": "60min", "ma_period": 60, "tolerance": 0.001}
        ]
        return golden_cross_conditions, touch_conditions


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



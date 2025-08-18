from decimal import Decimal
from typing import Optional, Dict, Any, List
import pandas as pd

from backtest.types import ActionType, Signal
from backtest.timeframe import MultiTimeframeData
from backtest.logger import get_logger
from backtest.strategies import Strategy
from indicators.indicators import MovingAverage


class GoldenCrossStrategy(Strategy):
    """
    Golden Cross Strategy with Multi-Timeframe Confirmation
    
    Entry Conditions:
    - Golden Cross: MA20 > MA50 > MA200 (daily)
    - Multi-timeframe MA touch: Price within 2% of any MTF MA
    
    Exit Conditions:
    - Take Profit 1: 38.2% of (last_high - entry_price), exit 50%
    - Take Profit 2: 50% of (last_high - entry_price), exit remaining
    - Stop Loss: 5% below entry price
    
    Features:
    - Multi-entry scaling (up to 10 entries)
    - Position sizing with scaling factor
    - State management for entry tracking
    """
    def __init__(
        self,
        data: Optional['MultiTimeframeData'] = None,
        ma_periods: Optional[Dict[str, List[int]]] = None,
        position_sizing: Optional[Dict[str, Any]] = None,
        exit_levels: Optional[Dict[str, float]] = None
    ):
        super().__init__("GoldenCrossStrategy")
        self.data = data
        self._logger = get_logger(__name__)

        # NOTE: Strategy parameters
        self.parameters = {
            "ma_periods": ma_periods or {
                "daily": [112, 224, 448],
                "3min": [360], "30min": [60], "60min": [60]
            },
            "position_sizing": position_sizing or {
                "initial_size": 0.01, "scale_factor": 1.4, "max_entries": 10
            },
            "exit_levels": exit_levels or {
                "tp1_level": 0.382, "tp1_size": 0.5, "tp2_level": 0.5, "tp2_size": 0.5
            }
        }

        # NOTE: Strategy state - persistent across signal periods
        self.entry_count = 0
        self.last_high = None
        self.current_signal_period_id = None  # Track signal periods
        self.total_entries_made = 0  # Track total entries across all periods
        
        self._daily_ma_indicators = {
            period: MovingAverage(name=f"ma_{period}", length=period)
            for period in self.parameters["ma_periods"]["daily"]
        }
        
        # Cache MTF indicators for performance
        self._mtf_ma_indicators = {
            f"{tf}_{period}": MovingAverage(name=f"ma_{tf}_{period}", length=period)
            for tf, period in [("3m", 360), ("30m", 60), ("1h", 60)]
        }
        self._cached_mtf_data = {}

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        for period, indicator in self._daily_ma_indicators.items():
            ma_df = indicator.calculate(data)
            df[f"ma_{period}"] = ma_df[f"ma_{period}"]
        return df

    def check_golden_cross(self, data: pd.DataFrame) -> bool:
        ma_periods = self.parameters["ma_periods"]["daily"]
        if len(data) < max(ma_periods):
            return False

        last_row = data.iloc[-1]
        ma_values = []
        for period in ma_periods:
            ma_value = last_row.get(f"ma_{period}")
            if pd.isna(ma_value):
                return False
            ma_values.append(ma_value)
        
        return ma_values[0] > ma_values[1] > ma_values[2]

    def check_mtf_touch(self, current_price: Decimal, timestamp: Optional[pd.Timestamp] = None) -> bool:
        if not self.data:
            return False

        tolerance = Decimal("0.02")  # NOTE: 2% tolerance
        current_price_decimal = Decimal(str(current_price))
        
        # Cache key for performance optimization
        cache_key = f"{timestamp}_{current_price}" if timestamp else f"live_{current_price}"
        
        if cache_key in self._cached_mtf_data:
            return self._cached_mtf_data[cache_key]
        
        result = False
        for tf_key, period in [("3m", 360), ("30m", 60), ("1h", 60)]:
            if tf_key not in self.data:
                continue

            df = self.data[tf_key]
            if len(df) < period:
                continue
                
            if timestamp is not None:
                df = df[df.index <= timestamp]
                if len(df) < period:
                    continue

            # Use cached indicator if available
            indicator_key = f"{tf_key}_{period}"
            if indicator_key in self._mtf_ma_indicators:
                ma_indicator = self._mtf_ma_indicators[indicator_key]
            else:
                ma_indicator = MovingAverage(name=f"ma_{tf_key}_{period}", length=period)
                self._mtf_ma_indicators[indicator_key] = ma_indicator
                
            ma_df = ma_indicator.calculate(df)
            ma_value = ma_df[f"ma_{tf_key}_{period}"].iloc[-1]

            if not pd.isna(ma_value) and ma_value > 0:
                diff_pct = abs(current_price_decimal - Decimal(str(ma_value))) / Decimal(str(ma_value))
                if diff_pct <= tolerance:
                    result = True
                    break

        # Cache result but limit cache size
        if len(self._cached_mtf_data) > 1000:
            # Clear oldest half of cache
            items = list(self._cached_mtf_data.items())
            self._cached_mtf_data = dict(items[500:])
            
        self._cached_mtf_data[cache_key] = result
        return result

    def calculate_exit_levels(self, entry_price: Decimal, last_high: Decimal) -> Dict[str, Decimal]:
        # NOTE: Handle edge case where last_high < entry_price (losing position)
        effective_high = max(last_high, entry_price)
        price_range = effective_high - entry_price
        
        tp1_level = Decimal(str(self.parameters["exit_levels"]["tp1_level"]))
        tp2_level = Decimal(str(self.parameters["exit_levels"]["tp2_level"]))

        tp1_price = entry_price + (price_range * tp1_level)
        tp2_price = entry_price + (price_range * tp2_level)
        stop_loss_price = entry_price * Decimal("0.95")  # NOTE: 5% stop loss
        
        # NOTE: Ensure TP levels are above entry price
        tp1_price = max(tp1_price, entry_price * Decimal("1.01"))
        tp2_price = max(tp2_price, entry_price * Decimal("1.02"))
        
        return {
            "tp1": tp1_price,
            "tp2": tp2_price,
            "stop_loss": stop_loss_price
        }

    def generate_signal(self, data: pd.DataFrame, position: Optional[Any] = None, 
                       portfolio: Optional[Any] = None) -> Optional[Signal]:
        if data.empty:
            return None

        if not any(col.startswith('ma_') for col in data.columns):
            data = self.calculate_indicators(data)

        last_row = data.iloc[-1]
        current_price = Decimal(str(last_row["close"]))

        if self.last_high is None or current_price > self.last_high:
            self.last_high = current_price

        has_position = position and position.is_open
        # NOTE: Only reset state when explicitly requested, not when position closes
        # This allows proper tracking across multiple signal periods

        if has_position:
            return self._check_exit_conditions(position, current_price)

        return self._check_entry_conditions(data, current_price)

    def _check_exit_conditions(self, position, current_price: Decimal) -> Optional[Signal]:
        exit_levels = self.calculate_exit_levels(position.entry_price, self.last_high)

        if current_price >= exit_levels["tp2"]:
            # NOTE: Reset signal period but preserve total entry tracking
            self.current_signal_period_id = None
            return Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "take_profit_2"}
            )
        elif current_price >= exit_levels["tp1"]:
            return Signal(
                type=ActionType.CLOSE, strength=0.5, price=current_price,
                quantity=position.open_quantity * Decimal("0.5"),
                metadata={"reason": "take_profit_1"}
            )
        elif current_price <= exit_levels["stop_loss"]:
            # NOTE: Reset signal period but preserve total entry tracking
            self.current_signal_period_id = None
            return Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "stop_loss"}
            )
        return None

    def _check_entry_conditions(self, data: pd.DataFrame, current_price: Decimal) -> Optional[Signal]:
        max_entries = self.parameters["position_sizing"]["max_entries"]
        
        if not self.check_golden_cross(data):
            # No golden cross - end current signal period
            if self.current_signal_period_id is not None:
                self.current_signal_period_id = None
                self.entry_count = 0
            return None

        timestamp = data.index[-1] if not data.empty else None
        if not self.check_mtf_touch(current_price, timestamp):
            return None

        # Generate unique signal period ID based on timestamp
        signal_period_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Check if this is a new signal period
        if self.current_signal_period_id != signal_period_id:
            self.current_signal_period_id = signal_period_id
            self.entry_count = 0  # Reset for new period

        # Check if we can make more entries in this period
        if self.entry_count >= max_entries:
            return None

        self.entry_count += 1
        self.total_entries_made += 1
        
        return Signal(
            type=ActionType.BUY, strength=0.8, price=current_price,
            metadata={
                "entry_count": self.entry_count,
                "total_entries": self.total_entries_made,
                "signal_period_id": signal_period_id,
                "golden_cross": True,
                "multi_timeframe_touch": True,
                "position_sizing": self.parameters["position_sizing"]
            }
        )

    def reset_state(self):
        """Completely reset strategy state - use with caution"""
        self.entry_count = 0
        self.last_high = None
        self.current_signal_period_id = None
        self.total_entries_made = 0

    def generate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all signals for backtesting with proper signal period tracking.
        Each signal period gets a unique ID and proper entry counting.
        """
        data_with_indicators = self.calculate_indicators(data)
        ma_periods = self.parameters["ma_periods"]["daily"]
        
        if len(data_with_indicators) < max(ma_periods):
            return pd.DataFrame()
        
        # NOTE: Find golden cross points
        ma_cols = [f"ma_{p}" for p in ma_periods]
        golden_cross_mask = (data_with_indicators[ma_cols[0]] > data_with_indicators[ma_cols[1]]) & \
                           (data_with_indicators[ma_cols[1]] > data_with_indicators[ma_cols[2]])
        
        # NOTE: Generate signals with consistent signal period tracking
        signals = []
        entry_count = 0
        last_high = None
        current_signal_period_id = None
        total_entries_made = 0
        
        for i in range(max(ma_periods), len(data_with_indicators)):
            current_row = data_with_indicators.iloc[i]
            current_price = Decimal(str(current_row["close"]))
            timestamp = data_with_indicators.index[i]
            
            if last_high is None or current_price > last_high:
                last_high = current_price
            
            has_golden_cross = golden_cross_mask.iloc[i]
            
            if has_golden_cross:
                has_mtf_touch = self.check_mtf_touch(current_price, timestamp)
                
                if has_mtf_touch:
                    # Generate unique signal period ID
                    signal_period_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    
                    # Check if this is a new signal period
                    if current_signal_period_id != signal_period_id:
                        current_signal_period_id = signal_period_id
                        entry_count = 0  # Reset for new period
                        last_high = current_price
                    
                    max_entries = self.parameters["position_sizing"]["max_entries"]
                    if entry_count < max_entries:
                        entry_count += 1
                        total_entries_made += 1
                        
                        signals.append({
                            'timestamp': timestamp,
                            'type': ActionType.BUY.value,
                            'strength': 0.8,
                            'metadata': {
                                'golden_cross': True,
                                'multi_timeframe_touch': True,
                                'entry_count': entry_count,
                                'total_entries': total_entries_made,
                                'signal_period_id': signal_period_id,
                                'last_high': float(last_high),
                                'position_sizing': self.parameters["position_sizing"],
                                'signal_period_start': entry_count == 1
                            }
                        })
            else:
                # NOTE: No golden cross - end current signal period
                if current_signal_period_id is not None:
                    current_signal_period_id = None
                    entry_count = 0
                    last_high = None
        
        return pd.DataFrame(signals).set_index('timestamp') if signals else pd.DataFrame()
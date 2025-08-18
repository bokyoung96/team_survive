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

        # Strategy parameters
        self.parameters = {
            "ma_periods": ma_periods or {
                "daily": [20, 50, 200],
                "3min": [360], "30min": [60], "60min": [60]
            },
            "position_sizing": position_sizing or {
                "initial_size": 0.01, "scale_factor": 1.4, "max_entries": 10
            },
            "exit_levels": exit_levels or {
                "tp1_level": 0.382, "tp1_size": 0.5, "tp2_level": 0.5, "tp2_size": 0.5
            }
        }

        # Strategy state
        self.last_high = None
        self.last_pyramid_timestamp = None  # Track last pyramid entry time
        
        self._daily_ma_indicators = {
            period: MovingAverage(name=f"ma_{period}", length=period)
            for period in self.parameters["ma_periods"]["daily"]
        }
        
        # Cache MTF indicators
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

        tolerance = Decimal("0.50")  # 50% tolerance
        current_price_decimal = Decimal(str(current_price))
        
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

            indicator_key = f"{tf_key}_{period}"
            ma_indicator = self._mtf_ma_indicators.get(indicator_key)
            if not ma_indicator:
                ma_indicator = MovingAverage(name=f"ma_{tf_key}_{period}", length=period)
                self._mtf_ma_indicators[indicator_key] = ma_indicator
                
            ma_df = ma_indicator.calculate(df)
            ma_value = ma_df[f"ma_{tf_key}_{period}"].iloc[-1]

            if not pd.isna(ma_value) and ma_value > 0:
                diff_pct = abs(current_price_decimal - Decimal(str(ma_value))) / Decimal(str(ma_value))
                if diff_pct <= tolerance:
                    return True

        return False

    def calculate_exit_levels(self, entry_price: Decimal, last_high: Decimal) -> Dict[str, Decimal]:
        # Fixed TP/SL levels - ignore last_high for now to test pyramiding
        tp1_price = entry_price * Decimal("1.50")  # 50% profit
        tp2_price = entry_price * Decimal("2.00")  # 100% profit  
        stop_loss_price = entry_price * Decimal("0.50")  # 50% stop loss (very wide)
        
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
        timestamp = data.index[-1]

        # Update last high
        if self.last_high is None or current_price > self.last_high:
            self.last_high = current_price

        if position and position.is_open:
            # Check exit conditions first (TP/SL)
            exit_signal = self._check_exit_conditions(position, current_price)
            if exit_signal:
                return exit_signal
            
            # Check for pyramiding opportunities
            return self._check_pyramiding_entry(data, current_price, position, timestamp)

        # Check for new entry when no position
        return self._check_new_entry(data, current_price, timestamp)

    def _check_exit_conditions(self, position, current_price: Decimal) -> Optional[Signal]:
        exit_levels = self.calculate_exit_levels(position.entry_price, self.last_high)
        
        print(f"🔍 EXIT CHECK: price={current_price}, entry={position.entry_price}")
        print(f"   TP1={exit_levels['tp1']}, TP2={exit_levels['tp2']}, SL={exit_levels['stop_loss']}")

        if current_price >= exit_levels["tp2"]:
            print(f"   ✅ TP2 HIT: {current_price} >= {exit_levels['tp2']}")
            return Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "take_profit_2"}
            )
        elif current_price >= exit_levels["tp1"]:
            print(f"   ✅ TP1 HIT: {current_price} >= {exit_levels['tp1']}")
            return Signal(
                type=ActionType.CLOSE, strength=0.5, price=current_price,
                quantity=position.open_quantity * Decimal("0.5"),
                metadata={"reason": "take_profit_1"}
            )
        elif current_price <= exit_levels["stop_loss"]:
            print(f"   ✅ SL HIT: {current_price} <= {exit_levels['stop_loss']}")
            return Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "stop_loss"}
            )
        
        print(f"   ⏸️ No exit condition met")
        return None

    def _check_pyramiding_entry(self, data: pd.DataFrame, current_price: Decimal, 
                               position, timestamp: pd.Timestamp) -> Optional[Signal]:
        """Check for pyramiding opportunities when position is open."""
        max_entries = self.parameters["position_sizing"]["max_entries"]
        current_entries = position.metadata.get("entry_count", 1)
        
        print(f"🔺 PYRAMID CHECK: current_entries={current_entries}, max={max_entries}")
        
        # Check if we can add more entries
        if current_entries >= max_entries:
            print(f"   ❌ Max entries reached")
            return None
        
        # Check if golden cross is still active
        if not self.check_golden_cross(data):
            print(f"   ❌ No golden cross")
            return None
        
        print(f"   ✅ Golden cross active")
        
        # Check if MTF touch conditions are met  
        if not self.check_mtf_touch(current_price, timestamp):
            print(f"   ❌ No MTF touch")
            return None
        
        print(f"   ✅ MTF touch OK")
        
        # Get signal period from position
        signal_period_id = position.metadata.get("signal_period_id", 
                                                 f"{timestamp.strftime('%Y%m%d_%H%M%S')}")
        
        print(f"   🔥 PYRAMID SIGNAL GENERATED: Entry #{current_entries + 1}")
        return Signal(
            type=ActionType.BUY, strength=0.8, price=current_price,
            metadata={
                "entry_count": current_entries + 1,
                "signal_period_id": signal_period_id,
                "golden_cross": True,
                "multi_timeframe_touch": True,
                "position_sizing": self.parameters["position_sizing"],
                "pyramid_entry": True
            }
        )
    
    def _check_new_entry(self, data: pd.DataFrame, current_price: Decimal, 
                        timestamp: pd.Timestamp) -> Optional[Signal]:
        """Check for new entry signal when no position is open."""
        if not self.check_golden_cross(data):
            return None

        if not self.check_mtf_touch(current_price, timestamp):
            return None

        # Generate new signal period ID
        signal_period_id = f"{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        return Signal(
            type=ActionType.BUY, strength=0.8, price=current_price,
            metadata={
                "entry_count": 1,
                "signal_period_id": signal_period_id,
                "golden_cross": True,
                "multi_timeframe_touch": True,
                "position_sizing": self.parameters["position_sizing"]
            }
        )

    def reset_state(self):
        self.last_high = None
        self.last_pyramid_timestamp = None

    def generate_all_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        data_with_indicators = self.calculate_indicators(data)
        ma_periods = self.parameters["ma_periods"]["daily"]
        
        if len(data_with_indicators) < max(ma_periods):
            return pd.DataFrame()
        
        # Find golden cross points
        ma_cols = [f"ma_{p}" for p in ma_periods]
        golden_cross_mask = (data_with_indicators[ma_cols[0]] > data_with_indicators[ma_cols[1]]) & \
                           (data_with_indicators[ma_cols[1]] > data_with_indicators[ma_cols[2]])
        
        signals = []
        last_high = None
        current_signal_period_id = None
        entries_in_current_period = 0
        last_signal_timestamp = None
        
        for i in range(max(ma_periods), len(data_with_indicators)):
            current_row = data_with_indicators.iloc[i]
            current_price = Decimal(str(current_row["close"]))
            timestamp = data_with_indicators.index[i]
            
            if last_high is None or current_price > last_high:
                last_high = current_price
            
            # Golden Cross 활성화된 동안 계속 신호 생성
            if golden_cross_mask.iloc[i]:
                # 새로운 Golden Cross 기간 시작
                if current_signal_period_id is None:
                    current_signal_period_id = f"GC_{timestamp.strftime('%Y%m%d_%H%M%S')}"
                    entries_in_current_period = 0
                
                # MTF Touch 체크하고 피라미딩 신호 생성
                if self.check_mtf_touch(current_price, timestamp):
                    # 너무 빈번한 신호 방지 - 최소 2바 간격
                    if last_signal_timestamp is None or \
                       data_with_indicators.index.get_loc(timestamp) - data_with_indicators.index.get_loc(last_signal_timestamp) >= 2:
                        
                        max_entries = self.parameters["position_sizing"]["max_entries"]
                        if entries_in_current_period < max_entries:
                            entries_in_current_period += 1
                            last_signal_timestamp = timestamp
                            
                            signals.append({
                                'timestamp': timestamp,
                                'type': ActionType.BUY.value,
                                'strength': 0.8,
                                'metadata': {
                                    'golden_cross': True,
                                    'multi_timeframe_touch': True,
                                    'entry_count': entries_in_current_period,
                                    'signal_period_id': current_signal_period_id,
                                    'last_high': float(last_high),
                                    'position_sizing': self.parameters["position_sizing"]
                                }
                            })
            else:
                # Golden Cross 끝남 - 다음 기간을 위해 리셋
                if current_signal_period_id is not None:
                    current_signal_period_id = None
                    entries_in_current_period = 0
        
        return pd.DataFrame(signals).set_index('timestamp') if signals else pd.DataFrame()
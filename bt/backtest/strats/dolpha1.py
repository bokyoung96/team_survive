from decimal import Decimal
from typing import Optional, Dict, Any, List
import pandas as pd
from datetime import datetime
from functools import lru_cache
from collections import deque

from backtest.types import ActionType, Signal
from backtest.timeframe import MultiTimeframeData
from backtest.logger import get_logger
from backtest.strategies import StreamingStrategy, TradingContext
from indicators.indicators import MovingAverage


class GoldenCrossStrategy(StreamingStrategy):
    """
    Golden Cross Strategy with MTF
    
    Entry Conditions:
    - Golden Cross: MA112 > MA224 > MA448 (daily)
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
        exit_levels: Optional[Dict[str, float]] = None,
        lookback_periods: int = 448
    ):
        max_ma_period = max(ma_periods.get("daily", [112, 224, 448])) if ma_periods else 448
        if lookback_periods < max_ma_period:
            lookback_periods = max_ma_period
            self._logger.warning(f"Adjusting lookback_periods to {max_ma_period} to support MA calculations")
        
        super().__init__("GoldenCrossStrategy", lookback_periods=lookback_periods)
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

        self._initialize_indicators()
        # NOTE: Rolling buffers for incremental MTF data management
        self._mtf_buffers = {
            "3m": deque(maxlen=360),
            "30m": deque(maxlen=60), 
            "1h": deque(maxlen=60)
        }
        self._last_cache_timestamp = None
        
    
    def _initialize_indicators(self):
        # NOTE: Daily MA indicators
        self._daily_ma_indicators = {
            period: MovingAverage(name=f"ma_{period}", length=period)
            for period in self.parameters["ma_periods"]["daily"]
        }
        
        # NOTE: MTF MA indicators
        self._mtf_ma_indicators = {
            f"{tf}_{period}": MovingAverage(name=f"ma_{tf}_{period}", length=period)
            for tf, period in [("3m", 360), ("30m", 60), ("1h", 60)]
        }

    @lru_cache(maxsize=50)
    def _calculate_mtf_ma(self, timestamp_str: str, tf_key: str, period: int) -> Optional[float]:
        if not self.data or tf_key not in self.data:
            return None

        df = self.data[tf_key]
        if len(df) < period:
            return None
            
        timestamp = datetime.fromisoformat(timestamp_str)
        df = df[df.index <= timestamp]
        if len(df) < period:
            return None

        indicator_key = f"{tf_key}_{period}"
        ma_indicator = self._mtf_ma_indicators.get(indicator_key)
        if not ma_indicator:
            return None
            
        ma_df = ma_indicator.calculate(df)
        ma_value = ma_df[f"ma_{tf_key}_{period}"].iloc[-1]

        if pd.isna(ma_value) or ma_value <= 0:
            return None
            
        return float(ma_value)

    def update_indicators(self, context: TradingContext) -> None:
        lookback = context.lookback_data
        
        for period, indicator in self._daily_ma_indicators.items():
            ma_df = indicator.calculate(lookback)
            if not ma_df.empty and not ma_df[indicator.name].isna().iloc[-1]:
                self.set_indicator_value(f"ma_{period}", ma_df[indicator.name].iloc[-1])
        
        # NOTE: Update state variables
        current_price = context.get_current_price()
        last_high = self.get_state("last_high", current_price)
        if current_price > last_high:
            self.set_state("last_high", current_price)
        
    def process_bar(self, context: TradingContext) -> Optional[Signal]:
        current_price = context.get_current_price()
        
        if not self._has_sufficient_data(context):
            return None
        
        if context.position and context.position.is_open:
            # NOTE: Check exit conditions
            exit_signal = self._check_exit_conditions(context)
            if exit_signal:
                return exit_signal
            
            # NOTE: Check martingale entries
            return self._check_martingale_entry(context)
        
        # NOTE: Check new entry
        return self._check_new_entry(context)
    
    def _has_sufficient_data(self, context: TradingContext) -> bool:
        max_period = max(self.parameters["ma_periods"]["daily"])
        return len(context.lookback_data) >= max_period

    def _check_golden_cross(self, context: TradingContext) -> bool:
        ma_periods = self.parameters["ma_periods"]["daily"]
        
        ma_values = []
        for period in ma_periods:
            ma_value = self.get_indicator_value(f"ma_{period}")
            if ma_value is None or pd.isna(ma_value):
                return False
            ma_values.append(ma_value)
        
        return ma_values[0] > ma_values[1] > ma_values[2]

    def check_mtf_touch(self, current_price: Decimal, timestamp) -> bool:
        if not self.data:
            return False

        # NOTE: MTF touch tolerance (2%)
        tolerance = Decimal("0.02")
        timestamp_str = timestamp.isoformat()
        
        for tf_key, period in [("3m", 360), ("30m", 60), ("1h", 60)]:
            ma_value = self._calculate_mtf_ma(timestamp_str, tf_key, period)
            if ma_value is None:
                continue
                
            diff_pct = abs(current_price - Decimal(str(ma_value))) / Decimal(str(ma_value))
            if diff_pct <= tolerance:
                return True

        return False

    def _calculate_exit_levels(self, entry_price: Decimal, last_high: Decimal) -> Dict[str, Decimal]:
        # NOTE: Fibonacci levels - 0.382 (38.2%) / 0.5 (50%)
        price_range = last_high - entry_price
        tp1_price = entry_price + (price_range * Decimal(str(self.parameters["exit_levels"]["tp1_level"])))
        tp2_price = entry_price + (price_range * Decimal(str(self.parameters["exit_levels"]["tp2_level"])))
        
        # NOTE: SL 5% below entry price
        stop_loss_price = entry_price * Decimal("0.95")
        
        return {
            "tp1": tp1_price,
            "tp2": tp2_price,
            "stop_loss": stop_loss_price
        }


    def _check_exit_conditions(self, context: TradingContext) -> Optional[Signal]:
        position = context.position
        current_price = context.get_current_price()
        last_high = self.get_state("last_high", current_price)
        
        exit_levels = self._calculate_exit_levels(position.entry_price, last_high)
        
        self._logger.debug(
            f"EXIT CHECK: price={current_price}, entry={position.entry_price}, "
            f"TP1={exit_levels['tp1']}, TP2={exit_levels['tp2']}, SL={exit_levels['stop_loss']}"
        )

        if current_price >= exit_levels["tp2"]:
            self._logger.info(f"TP2 HIT: {current_price} >= {exit_levels['tp2']}")
            return Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "take_profit_2"}
            )
        elif current_price >= exit_levels["tp1"]:
            self._logger.info(f"TP1 HIT: {current_price} >= {exit_levels['tp1']}")
            return Signal(
                type=ActionType.CLOSE, strength=0.5, price=current_price,
                quantity=position.open_quantity * Decimal("0.5"),
                metadata={"reason": "take_profit_1"}
            )
        elif current_price <= exit_levels["stop_loss"]:
            self._logger.info(f"SL HIT: {current_price} <= {exit_levels['stop_loss']}")
            return Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "stop_loss"}
            )
        
        return None

    def _check_martingale_entry(self, context: TradingContext) -> Optional[Signal]:
        position = context.position
        current_price = context.get_current_price()
        
        max_entries = self.parameters["position_sizing"]["max_entries"]
        current_entries = position.metadata.get("entry_count", 1)
        
        self._logger.debug(f"MARTINGALE CHECK: current_entries={current_entries}, max={max_entries}")
        
        # NOTE: Check if we can add more entries
        if current_entries >= max_entries:
            self._logger.debug(f"Max entries reached: {current_entries}/{max_entries}")
            return None
        
        # NOTE: Check if golden cross is still active
        if not self._check_golden_cross(context):
            return None
        
        # NOTE: Check if MTF touch conditions are met  
        if not self.check_mtf_touch(context.get_current_price(), context.timestamp):
            return None
        
        # NOTE: Get signal period from position
        signal_period_id = position.metadata.get(
            "signal_period_id", 
            f"{context.timestamp.strftime('%Y%m%d_%H%M%S')}"
        )
        
        self._logger.info(f"MARTINGALE SIGNAL GENERATED: Entry #{current_entries + 1}")
        return Signal(
            type=ActionType.BUY, strength=0.8, price=current_price,
            metadata={
                "entry_count": current_entries + 1,
                "signal_period_id": signal_period_id,
                "golden_cross": True,
                "multi_timeframe_touch": True,
                "position_sizing": self.parameters["position_sizing"],
                "martingale_entry": True
            }
        )
    
    def _check_new_entry(self, context: TradingContext) -> Optional[Signal]:
        current_price = context.get_current_price()
        
        if not self._check_golden_cross(context):
            return None

        if not self.check_mtf_touch(context.get_current_price(), context.timestamp):
            return None

        # NOTE: Generate new signal period ID
        signal_period_id = f"{context.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        self._logger.info(f"NEW ENTRY SIGNAL: {signal_period_id}")
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
        super().reset_state()
        for indicator in self._daily_ma_indicators.values():
            indicator.reset()
        for indicator in self._mtf_ma_indicators.values():
            indicator.reset()
        
        # NOTE: Clear LRU cache and reset buffers
        self._calculate_mtf_ma.cache_clear()
        for buffer in self._mtf_buffers.values():
            buffer.clear()
        self._last_cache_timestamp = None

    

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
from indicators.indicators import SMA


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
    
    # NOTE: Strategy parameters
    MTF_KEYS = ("3m", "30m", "1h")
    MTF_PERIODS = {"3m": 360, "30m": 60, "1h": 60}
    MTF_BUFFER_SIZES = {"3m": 360, "30m": 60, "1h": 60}
    DAILY_MA_PERIODS = (112, 224, 448)
    TOUCH_TOLERANCE = 0.02
    STOP_LOSS_PCT = 0.95
    
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

        # NOTE: Strategy parameters with defaults
        self.parameters = {
            "ma_periods": ma_periods or {"daily": list(self.DAILY_MA_PERIODS)},
            "position_sizing": position_sizing or {
                "initial_size": 0.01, "scale_factor": 1.4, "max_entries": 10
            },
            "exit_levels": exit_levels or {
                "tp1_level": 0.382, "tp1_size": 0.5, "tp2_level": 0.5, "tp2_size": 0.5
            }
        }

        self._initialize_buffers()
        self._initialize_indicators()
    
    def _initialize_buffers(self):
        self._mtf_buffers = {
            key: deque(maxlen=size) 
            for key, size in self.MTF_BUFFER_SIZES.items()
        }
        
        self._mtf_last_idx = {key: -1 for key in self.MTF_KEYS}
        
        max_daily_period = max(self.parameters["ma_periods"]["daily"])
        self._daily_price_buffer = deque(maxlen=max_daily_period)
    
    def _initialize_indicators(self):
        # NOTE: Daily MA indicators with streaming support
        self._daily_ma_indicators = {
            period: SMA(name=f"ma_{period}", length=period)
            for period in self.parameters["ma_periods"]["daily"]
        }
        
        # NOTE: MTF MA indicators
        self._mtf_ma_indicators = {
            f"{tf}_{period}": SMA(name=f"ma_{tf}_{period}", length=period)
            for tf, period in self.MTF_PERIODS.items()
        }

    @lru_cache(maxsize=50)
    def _calculate_mtf_ma(self, timestamp_str: str, tf_key: str, period: int) -> Optional[float]:
        if tf_key not in self._mtf_buffers:
            return None
            
        buffer = self._mtf_buffers[tf_key]
        if len(buffer) < period:
            return None
            
        recent_prices = list(buffer)[-period:]
        ma_value = sum(recent_prices) / len(recent_prices)
        
        if ma_value <= 0:
            return None
            
        return float(ma_value)

    def _update_mtf_buffers(self, timestamp: datetime, price: Decimal) -> None:
        if not self.data:
            return
        
        for tf_key in self.MTF_KEYS:
            if tf_key not in self.data:
                continue
            df = self.data[tf_key]
            
            idx = self._get_next_mtf_index(df, tf_key, timestamp)
            if idx is None:
                continue
            
            latest_price = float(df.iloc[idx]['close'])
            buffer = self._mtf_buffers[tf_key]
            
            if not buffer or buffer[-1] != latest_price:
                buffer.append(latest_price)
            
            self._mtf_last_idx[tf_key] = idx
    
    def _get_next_mtf_index(self, df: pd.DataFrame, tf_key: str, timestamp: datetime) -> Optional[int]:
        last_idx = self._mtf_last_idx[tf_key]
        df_len = len(df)
        
        if 0 <= last_idx < df_len - 1:
            if df.index[last_idx + 1] <= timestamp:
                idx = last_idx + 1
                while idx < df_len and df.index[idx] <= timestamp:
                    idx += 1
                return idx - 1
            return None
        
        if last_idx == -1:
            try:
                return df.index.get_indexer([timestamp], method='ffill')[0]
            except:
                return None
        return None

    def update_indicators(self, context: TradingContext) -> None:
        current_price = context.get_current_price()
        current_price_float = float(current_price)
        
        self._daily_price_buffer.append(current_price_float)
        self._update_mtf_buffers(context.timestamp, current_price)
        
        self._calculate_daily_mas()
        
        if current_price > self.get_state("last_high", current_price):
            self.set_state("last_high", current_price)
    
    def _calculate_daily_mas(self) -> None:
        buffer_len = len(self._daily_price_buffer)
        if buffer_len == 0:
            return
        
        latest_price = self._daily_price_buffer[-1]
        
        for period, indicator in self._daily_ma_indicators.items():
            ma_value = indicator.update(latest_price)
            if ma_value is not None:
                self.set_indicator_value(f"ma_{period}", ma_value)
        
    def process_bar(self, context: TradingContext) -> Optional[Signal]:
        if not self._has_sufficient_data(context):
            return None
        
        position = context.position
        if position and position.is_open:
            # NOTE: Check exit conditions first
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

    def check_mtf_touch(self, current_price: Decimal, timestamp: datetime) -> bool:
        if not self.data:
            return False

        price_float = float(current_price)
        lower_bound = price_float * (1 - self.TOUCH_TOLERANCE)
        upper_bound = price_float * (1 + self.TOUCH_TOLERANCE)
        timestamp_str = timestamp.isoformat()
        
        for tf, period in self.MTF_PERIODS.items():
            ma_value = self._calculate_mtf_ma(timestamp_str, tf, period)
            if ma_value is None or not (lower_bound <= ma_value <= upper_bound):
                return False
        
        return True

    def _calculate_exit_levels(self, entry_price: Decimal, last_high: Decimal) -> Dict[str, Decimal]:
        price_range = last_high - entry_price
        exit_params = self.parameters["exit_levels"]
        
        return {
            "tp1": entry_price + (price_range * Decimal(str(exit_params["tp1_level"]))),
            "tp2": entry_price + (price_range * Decimal(str(exit_params["tp2_level"]))),
            "stop_loss": entry_price * Decimal(str(self.STOP_LOSS_PCT))
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

        exit_signal = None
        if current_price >= exit_levels["tp2"]:
            self._logger.info(f"TP2 HIT: {current_price} >= {exit_levels['tp2']}")
            exit_signal = Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "take_profit_2"}
            )
        elif current_price >= exit_levels["tp1"]:
            self._logger.info(f"TP1 HIT: {current_price} >= {exit_levels['tp1']}")
            exit_signal = Signal(
                type=ActionType.CLOSE, strength=0.5, price=current_price,
                quantity=position.open_quantity * Decimal("0.5"),
                metadata={"reason": "take_profit_1"}
            )
        elif current_price <= exit_levels["stop_loss"]:
            self._logger.info(f"SL HIT: {current_price} <= {exit_levels['stop_loss']}")
            exit_signal = Signal(
                type=ActionType.CLOSE, strength=1.0, price=current_price,
                metadata={"reason": "stop_loss"}
            )
        
        return exit_signal

    def _check_martingale_entry(self, context: TradingContext) -> Optional[Signal]:
        position = context.position
        current_price = context.get_current_price()
        current_timestamp = context.timestamp
        
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
        
        self._logger.info(
            f"MARTINGALE SIGNAL GENERATED: Entry #{current_entries + 1} at {current_timestamp}"
        )
        return Signal(
            type=ActionType.BUY, strength=0.8, price=current_price,
            metadata={
                "entry_count": current_entries + 1,
                "signal_period_id": signal_period_id,
                "entry_timestamp": current_timestamp.strftime('%Y%m%d_%H%M%S'),
                "golden_cross": True,
                "multi_timeframe_touch": True,
                "position_sizing": self.parameters["position_sizing"],
                "martingale_entry": True
            }
        )
    
    def _check_new_entry(self, context: TradingContext) -> Optional[Signal]:
        current_price = context.get_current_price()
        current_timestamp = context.timestamp
        
        if not self._check_golden_cross(context):
            return None

        if not self.check_mtf_touch(context.get_current_price(), context.timestamp):
            return None

        # NOTE: Generate new signal period ID
        signal_period_id = f"{context.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        self._logger.info(f"NEW ENTRY SIGNAL: {signal_period_id} at {current_timestamp}")
        return Signal(
            type=ActionType.BUY, strength=0.8, price=current_price,
            metadata={
                "entry_count": 1,
                "signal_period_id": signal_period_id,
                "entry_timestamp": current_timestamp.strftime('%Y%m%d_%H%M%S'),
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
        # NOTE: Reset MTF indices
        for tf_key in self._mtf_last_idx:
            self._mtf_last_idx[tf_key] = -1
        # NOTE: Reset daily buffers
        self._daily_price_buffer.clear()

    

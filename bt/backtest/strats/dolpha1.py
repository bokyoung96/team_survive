from decimal import Decimal
from typing import Optional, Dict, List, Any
import pandas as pd

from backtest.types import ActionType, Signal
from backtest.timeframe import MultiTimeframeData
from backtest.logger import get_logger
from backtest.strategies import Strategy
from indicators.indicators import MovingAverage


class GoldenCrossStrategy(Strategy):
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
        
        # NOTE: Initialize indicators
        self._clear_indicators()
    
    def _clear_indicators(self):
        if hasattr(self, '_daily_ma_indicators'):
            del self._daily_ma_indicators
        if hasattr(self, '_mtf_ma_indicators'):
            del self._mtf_ma_indicators
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if not hasattr(self, '_daily_ma_indicators'):
            self._daily_ma_indicators = {}
            for period in self.parameters["ma_periods"]["daily"]:
                self._daily_ma_indicators[period] = MovingAverage(name=f"ma_{period}", length=period)

        for period, indicator in self._daily_ma_indicators.items():
            ma_df = indicator.calculate(data)
            df[f"ma_{period}"] = ma_df[f"ma_{period}"]

        return df

    def check_golden_cross(self, data: pd.DataFrame) -> bool:
        if len(data) < self.parameters["ma_periods"]["daily"][-1]:
            return False

        last_row = data.iloc[-1]
        ma_periods = self.parameters["ma_periods"]["daily"]

        ma_112 = last_row.get(f"ma_{ma_periods[0]}")
        ma_224 = last_row.get(f"ma_{ma_periods[1]}")
        ma_448 = last_row.get(f"ma_{ma_periods[2]}")

        if pd.isna(ma_112) or pd.isna(ma_224) or pd.isna(ma_448):
            return False

        return ma_112 > ma_224 > ma_448

    def check_mtf_touch(self, current_price: Decimal) -> bool:
        if not self.data:
            return False

        tolerance = Decimal("0.01")  # 1% tolerance for mtf touch

        if not hasattr(self, '_mtf_ma_indicators'):
            self._mtf_ma_indicators = {}
        
        checks = [
            ("3m", 360),   # 3min 360 MA
            ("30m", 60),   # 30min 60 MA
            ("1h", 60)     # 60min 60 MA
        ]

        for tf_key, period in checks:
            if tf_key not in self.data:
                continue

            df = self.data[tf_key]
            if df.empty or len(df) < period:
                continue

            indicator_key = f"{tf_key}_{period}"
            
            if indicator_key not in self._mtf_ma_indicators:
                self._mtf_ma_indicators[indicator_key] = MovingAverage(name=f"ma_{period}", length=period)
            
            ma_series = self._mtf_ma_indicators[indicator_key].calculate(df)[f"ma_{period}"]
            ma_value = ma_series.iloc[-1]

            if not pd.isna(ma_value) and ma_value > 0:
                diff_pct = abs(Decimal(str(current_price)) - Decimal(str(ma_value))) / Decimal(str(ma_value))
                if diff_pct <= tolerance:
                    return True

        return False

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
        position: Optional[Any] = None
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
                self._logger.info(
                    f"Take profit 2 signal at ${current_price} (target: ${exit_levels['tp2']})")
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
        max_entries = self.parameters["position_sizing"]["max_entries"]
        if self.entry_count >= max_entries:
            return None
            
        golden_cross = self.check_golden_cross(data)
        multi_tf_touch = self.check_mtf_touch(current_price)

        if golden_cross and multi_tf_touch:
            self._logger.info(
                f"Entry conditions met: Golden cross + Multi-TF touch at ${current_price}")

            self.entry_count += 1
            self.entry_prices.append(current_price)
            
            self._logger.info(
                f"BUY signal generated: ${current_price}, entry #{self.entry_count}/{max_entries}")
            return Signal(
                type=ActionType.BUY,
                strength=0.8,
                price=current_price,
                metadata={
                    "entry_count": self.entry_count,
                    "golden_cross": True,
                    "multi_timeframe_touch": True,
                    "position_sizing": self.parameters["position_sizing"]
                }
            )
        elif golden_cross:
            self._logger.debug(
                "Golden cross detected but no multi-timeframe touch")
        elif multi_tf_touch:
            self._logger.debug(
                "Multi-timeframe touch detected but no golden cross")

        return None

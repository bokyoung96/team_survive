# Strategy Development Guide

## Table of Contents

1. [Overview](#overview)
2. [Strategy Architecture](#strategy-architecture)
3. [Creating New Strategies](#creating-new-strategies)
4. [Position Sizing Framework](#position-sizing-framework)
5. [Signal Generation Best Practices](#signal-generation-best-practices)
6. [Multi-Timeframe Strategy Patterns](#multi-timeframe-strategy-patterns)
7. [Indicator Caching Patterns](#indicator-caching-patterns)
8. [Testing and Validation Guidelines](#testing-and-validation-guidelines)
9. [Real Examples from Codebase](#real-examples-from-codebase)
10. [Advanced Patterns](#advanced-patterns)

## Overview

This guide provides comprehensive documentation for developing trading strategies in the backtesting system. It covers architecture patterns, best practices, and real examples from the codebase.

### Core Concepts

- **Strategy**: Core logic for generating trading signals
- **Signal**: Trading action with price, quantity, and metadata
- **Position Sizing**: Risk management through controlled capital allocation
- **Multi-Timeframe**: Analysis across different time horizons
- **Indicator Caching**: Performance optimization for technical indicators

## Strategy Architecture

### Base Strategy Class

All strategies inherit from the abstract `Strategy` class:

```python
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
        """Calculate and add technical indicators to the data"""
        pass

    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: Optional[Any] = None,
        portfolio: Optional[Any] = None
    ) -> Optional[Signal]:
        """Generate trading signal based on current market data"""
        pass
```

### Signal Structure

Signals use a comprehensive dataclass structure:

```python
@dataclass(frozen=True)
class Signal:
    type: ActionType              # BUY, SELL, CLOSE, etc.
    strength: float              # Signal strength (0.0 to 1.0)
    price: Optional[Decimal] = None
    quantity: Optional[Decimal] = None
    stop_loss: Optional[Decimal] = None
    take_profit: Optional[Decimal] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Action Types

```python
class ActionType(Enum):
    BUY = "buy"
    SELL = "sell"
    CLOSE = "close"
    CLOSE_LONG = "close_long"
    CLOSE_SHORT = "close_short"
    HOLD = "hold"
    ENTRY = "entry"
    EXIT = "exit"
```

## Creating New Strategies

### Step 1: Basic Strategy Template

```python
from decimal import Decimal
from typing import Optional, Dict, Any
import pandas as pd

from backtest.types import ActionType, Signal
from backtest.strategies import Strategy
from indicators.indicators import MovingAverage, RSI, MACD

class MyCustomStrategy(Strategy):
    def __init__(self, parameters: Optional[Dict[str, Any]] = None):
        super().__init__("MyCustomStrategy", parameters)

        # Default parameters
        default_params = {
            "ma_periods": [20, 50],
            "rsi_period": 14,
            "rsi_oversold": 30,
            "rsi_overbought": 70,
            "position_sizing": {
                "initial_size": 0.02,
                "scale_factor": 1.3,
                "max_entries": 5
            }
        }

        self.parameters = {**default_params, **(parameters or {})}

        # Initialize indicator cache
        self._indicators = {}
        self._setup_indicators()

    def _setup_indicators(self):
        """Initialize technical indicators"""
        self._indicators = {
            'ma_20': MovingAverage(name="ma_20", length=20),
            'ma_50': MovingAverage(name="ma_50", length=50),
            'rsi': RSI(name="rsi", length=self.parameters["rsi_period"])
        }

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add all indicators to the dataframe"""
        df = data.copy()

        for name, indicator in self._indicators.items():
            result = indicator.calculate(data)
            df = pd.concat([df, result], axis=1)

        return df

    def generate_signal(
        self,
        data: pd.DataFrame,
        position: Optional[Any] = None,
        portfolio: Optional[Any] = None
    ) -> Optional[Signal]:
        """Generate trading signals"""
        if data.empty or len(data) < 50:
            return None

        last_row = data.iloc[-1]
        current_price = Decimal(str(last_row["close"]))

        # Exit logic for existing positions
        if position and hasattr(position, 'is_open') and position.is_open:
            return self._generate_exit_signal(data, position, current_price)

        # Entry logic
        return self._generate_entry_signal(data, current_price)

    def _generate_entry_signal(self, data: pd.DataFrame, current_price: Decimal) -> Optional[Signal]:
        """Generate entry signals"""
        last_row = data.iloc[-1]

        ma_20 = last_row.get('ma_20')
        ma_50 = last_row.get('ma_50')
        rsi = last_row.get('rsi')

        if pd.isna(ma_20) or pd.isna(ma_50) or pd.isna(rsi):
            return None

        # Golden cross + RSI oversold
        if (ma_20 > ma_50 and
            rsi < self.parameters["rsi_oversold"]):

            return Signal(
                type=ActionType.BUY,
                strength=0.8,
                price=current_price,
                metadata={
                    "reason": "golden_cross_oversold",
                    "rsi_value": rsi,
                    "position_sizing": self.parameters["position_sizing"]
                }
            )

        return None

    def _generate_exit_signal(self, data: pd.DataFrame, position: Any, current_price: Decimal) -> Optional[Signal]:
        """Generate exit signals for open positions"""
        last_row = data.iloc[-1]
        rsi = last_row.get('rsi')

        if pd.isna(rsi):
            return None

        # Exit on RSI overbought
        if rsi > self.parameters["rsi_overbought"]:
            return Signal(
                type=ActionType.CLOSE,
                strength=1.0,
                price=current_price,
                metadata={"reason": "rsi_overbought", "rsi_value": rsi}
            )

        return None
```

### Step 2: Advanced Strategy Features

```python
class AdvancedStrategy(Strategy):
    def __init__(self, data: Optional['MultiTimeframeData'] = None, **kwargs):
        super().__init__("AdvancedStrategy", kwargs.get('parameters'))
        self.data = data  # Multi-timeframe data access
        self.entry_count = 0
        self.last_signal_timestamp = None

    def _check_multi_timeframe_alignment(self) -> bool:
        """Check alignment across multiple timeframes"""
        if not self.data:
            return False

        alignments = []

        # Check daily trend
        if "1d" in self.data:
            daily_data = self.data["1d"]
            if len(daily_data) >= 50:
                daily_ma = daily_data["close"].rolling(50).mean()
                current_price = daily_data["close"].iloc[-1]
                alignments.append(current_price > daily_ma.iloc[-1])

        # Check 4-hour momentum
        if "4h" in self.data:
            h4_data = self.data["4h"]
            if len(h4_data) >= 20:
                h4_ma = h4_data["close"].rolling(20).mean()
                current_price = h4_data["close"].iloc[-1]
                alignments.append(current_price > h4_ma.iloc[-1])

        return all(alignments) if alignments else False

    def _calculate_dynamic_stop_loss(self, entry_price: Decimal, data: pd.DataFrame) -> Decimal:
        """Calculate ATR-based stop loss"""
        if len(data) < 14:
            return entry_price * Decimal("0.95")  # Default 5% stop

        # Simple ATR calculation
        high = data["high"].rolling(14)
        low = data["low"].rolling(14)
        close = data["close"].rolling(14)

        tr1 = high.max() - low.min()
        tr2 = abs(high.max() - close.shift(1).iloc[-1])
        tr3 = abs(low.min() - close.shift(1).iloc[-1])

        atr = max(tr1.iloc[-1], tr2, tr3)

        return entry_price - (Decimal(str(atr)) * Decimal("2.0"))
```

## Position Sizing Framework

### Standard Position Sizing Format

The position sizing system uses a standardized format passed through signal metadata:

```python
position_sizing = {
    "initial_size": 0.01,      # First entry: 1% of total portfolio value
    "scale_factor": 1.4,       # Multiplier for subsequent entries
    "max_entries": 10          # Maximum number of entries allowed
}
```

### Calculation Logic

Position size = `initial_size * (scale_factor ^ (entry_count - 1))`

#### Example Progression

- Entry 1: 1.0% of portfolio
- Entry 2: 1.4% of portfolio (1.0% × 1.4¹)
- Entry 3: 1.96% of portfolio (1.0% × 1.4²)
- Entry 4: 2.74% of portfolio (1.0% × 1.4³)
- Entry 5: 3.84% of portfolio (1.0% × 1.4⁴)

### Implementation in Strategies

Include position sizing parameters in signal metadata:

```python
def generate_signal(self, data, position=None, portfolio=None):
    # ... signal logic ...

    return Signal(
        type=ActionType.BUY,
        price=current_price,
        metadata={
            "entry_count": self.entry_count,
            "position_sizing": self.parameters["position_sizing"],
            "reason": "strategy_condition_met"
        }
    )
```

### Position Sizing Strategies

#### Conservative Scaling

```python
position_sizing = {
    "initial_size": 0.005,     # 0.5%
    "scale_factor": 1.2,       # 1.2x growth
    "max_entries": 15
}
```

#### Aggressive Scaling

```python
position_sizing = {
    "initial_size": 0.02,      # 2%
    "scale_factor": 1.5,       # 1.5x growth
    "max_entries": 8
}
```

#### Fixed Size

```python
position_sizing = {
    "initial_size": 0.01,      # 1%
    "scale_factor": 1.0,       # No scaling
    "max_entries": 20
}
```

#### Dynamic Position Sizing

```python
def calculate_dynamic_position_size(self, portfolio_value: Decimal, volatility: float) -> Dict[str, Any]:
    """Adjust position size based on volatility"""
    base_size = Decimal("0.01")

    # Reduce size in high volatility
    if volatility > 0.03:  # 3%
        adjusted_size = base_size * Decimal("0.5")
    elif volatility > 0.02:  # 2%
        adjusted_size = base_size * Decimal("0.75")
    else:
        adjusted_size = base_size

    return {
        "initial_size": float(adjusted_size),
        "scale_factor": 1.3,
        "max_entries": 8,
        "volatility_adjusted": True
    }
```

### Fallback Behavior

If no position sizing is specified in signal metadata, the executor defaults to using 95% of available cash for the trade.

## Signal Generation Best Practices

### 1. Signal Strength Guidelines

Use signal strength to indicate confidence:

```python
# High confidence signals
if strong_condition_1 and strong_condition_2:
    strength = 0.9

# Medium confidence signals
elif moderate_condition:
    strength = 0.6

# Weak signals
elif weak_condition:
    strength = 0.3
```

### 2. Metadata Best Practices

Include comprehensive metadata for debugging and analysis:

```python
metadata = {
    "reason": "golden_cross_rsi_divergence",
    "ma_fast": float(ma_fast_value),
    "ma_slow": float(ma_slow_value),
    "rsi": float(rsi_value),
    "volume_ratio": float(volume_ratio),
    "entry_count": self.entry_count,
    "position_sizing": self.parameters["position_sizing"],
    "timeframe_analysis": {
        "daily_trend": "bullish",
        "h4_momentum": "strong"
    }
}
```

### 3. Signal Validation

Implement signal validation to prevent invalid trades:

```python
def validate_signal(self, signal: Signal, data: pd.DataFrame, position=None, portfolio=None) -> bool:
    """Validate signal before execution"""

    # Don't generate conflicting signals
    if position and position.is_open:
        if signal.type in [ActionType.BUY, ActionType.SELL]:
            return False

    # Check minimum time between signals
    if (self.last_signal_timestamp and
        data.index[-1] - self.last_signal_timestamp < pd.Timedelta(hours=1)):
        return False

    # Validate price is reasonable
    if signal.price and signal.price <= 0:
        return False

    return True
```

### 4. Signal Filtering

```python
def apply_signal_filters(self, signal: Signal, data: pd.DataFrame) -> Optional[Signal]:
    """Apply filters to reduce noise"""

    # Volume filter
    if len(data) >= 20:
        avg_volume = data["volume"].rolling(20).mean().iloc[-1]
        current_volume = data["volume"].iloc[-1]

        if current_volume < avg_volume * 0.5:  # Low volume
            return None

    # Time-based filters
    current_time = data.index[-1]
    if current_time.hour < 9 or current_time.hour > 16:  # Outside market hours
        return None

    # Trend filter
    if self._is_sideways_market(data):
        signal.strength *= 0.5  # Reduce strength in sideways markets

    return signal
```

## Multi-Timeframe Strategy Patterns

### Pattern 1: Trend Alignment Strategy

```python
class TrendAlignmentStrategy(Strategy):
    def __init__(self, data: 'MultiTimeframeData', **kwargs):
        super().__init__("TrendAlignment", kwargs.get('parameters'))
        self.data = data

    def check_trend_alignment(self) -> Dict[str, str]:
        """Check trend direction across timeframes"""
        trends = {}

        timeframes = ["1d", "4h", "1h", "15m"]
        for tf in timeframes:
            if tf in self.data:
                df = self.data[tf]
                if len(df) >= 50:
                    ma_short = df["close"].rolling(20).mean()
                    ma_long = df["close"].rolling(50).mean()

                    if ma_short.iloc[-1] > ma_long.iloc[-1]:
                        trends[tf] = "bullish"
                    else:
                        trends[tf] = "bearish"

        return trends

    def generate_signal(self, data, position=None, portfolio=None):
        trends = self.check_trend_alignment()

        # Enter only when all timeframes align
        if all(trend == "bullish" for trend in trends.values()):
            return Signal(
                type=ActionType.BUY,
                strength=0.9,
                metadata={
                    "reason": "trend_alignment",
                    "trends": trends,
                    "position_sizing": self.parameters["position_sizing"]
                }
            )

        return None
```

### Pattern 2: Multi-Timeframe Confluence

```python
def check_mtf_confluence(self, current_price: float) -> Dict[str, bool]:
    """Check for confluence across multiple timeframes"""
    confluence = {}

    # Check support/resistance on higher timeframes
    if "1d" in self.data:
        daily_data = self.data["1d"]
        daily_pivot = self._calculate_pivot_points(daily_data)
        confluence["daily_support"] = abs(current_price - daily_pivot["support"]) / current_price < 0.002

    # Check momentum on medium timeframes
    if "4h" in self.data:
        h4_data = self.data["4h"]
        rsi = self._calculate_rsi(h4_data, 14)
        confluence["h4_oversold"] = rsi.iloc[-1] < 30

    # Check entry precision on lower timeframes
    if "15m" in self.data:
        m15_data = self.data["15m"]
        ma_touch = self._check_ma_touch(m15_data, current_price, period=50, tolerance=0.001)
        confluence["m15_ma_touch"] = ma_touch

    return confluence
```

### Pattern 3: Timeframe-Specific Logic

```python
def generate_mtf_signal(self, primary_tf: str = "1h") -> Optional[Signal]:
    """Generate signals with timeframe-specific logic"""

    if primary_tf not in self.data:
        return None

    primary_data = self.data[primary_tf]

    # Higher timeframe trend
    higher_tf_bullish = self._check_higher_tf_trend(primary_tf)

    # Current timeframe setup
    current_setup = self._check_current_setup(primary_data)

    # Lower timeframe entry
    lower_tf_entry = self._check_lower_tf_entry(primary_tf)

    if higher_tf_bullish and current_setup and lower_tf_entry:
        return Signal(
            type=ActionType.BUY,
            strength=0.85,
            metadata={
                "reason": "mtf_confluence",
                "primary_timeframe": primary_tf,
                "higher_tf_trend": "bullish",
                "position_sizing": self.parameters["position_sizing"]
            }
        )

    return None
```

## Indicator Caching Patterns

### Pattern 1: Lazy Loading with Caching

```python
class CachedIndicatorStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__("CachedStrategy", kwargs.get('parameters'))
        self._indicator_cache = {}
        self._last_calculation_length = 0

    def _get_cached_indicator(self, data: pd.DataFrame, indicator_key: str, indicator_func) -> pd.Series:
        """Get indicator with caching to avoid recalculation"""

        current_length = len(data)
        cache_key = f"{indicator_key}_{current_length}"

        # Check if we need to recalculate
        if (cache_key not in self._indicator_cache or
            current_length != self._last_calculation_length):

            self._indicator_cache[cache_key] = indicator_func(data)
            self._last_calculation_length = current_length

        return self._indicator_cache[cache_key]

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        # Cached moving averages
        df["ma_20"] = self._get_cached_indicator(
            data, "ma_20",
            lambda d: d["close"].rolling(20).mean()
        )

        df["ma_50"] = self._get_cached_indicator(
            data, "ma_50",
            lambda d: d["close"].rolling(50).mean()
        )

        return df
```

### Pattern 2: Incremental Indicator Updates

```python
class IncrementalIndicatorStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__("IncrementalStrategy", kwargs.get('parameters'))
        self._indicator_states = {}

    def _update_incremental_ma(self, new_price: float, period: int) -> float:
        """Update moving average incrementally"""
        key = f"ma_{period}"

        if key not in self._indicator_states:
            self._indicator_states[key] = {
                "sum": 0.0,
                "count": 0,
                "values": []
            }

        state = self._indicator_states[key]
        state["values"].append(new_price)
        state["sum"] += new_price
        state["count"] += 1

        # Remove old values if beyond period
        if len(state["values"]) > period:
            old_value = state["values"].pop(0)
            state["sum"] -= old_value
            state["count"] -= 1

        return state["sum"] / min(state["count"], period)
```

### Pattern 3: Multi-Timeframe Indicator Sharing

```python
class SharedIndicatorStrategy(Strategy):
    def __init__(self, data: 'MultiTimeframeData', **kwargs):
        super().__init__("SharedIndicator", kwargs.get('parameters'))
        self.data = data
        self._shared_indicators = {}

    def _get_shared_indicator(self, timeframe: str, indicator_name: str, **params):
        """Share indicators across timeframes to reduce computation"""

        key = f"{timeframe}_{indicator_name}_{hash(str(params))}"

        if key not in self._shared_indicators:
            if timeframe in self.data:
                tf_data = self.data[timeframe]

                if indicator_name == "rsi":
                    indicator = RSI(name=f"rsi_{timeframe}", length=params.get("period", 14))
                elif indicator_name == "macd":
                    indicator = MACD(name=f"macd_{timeframe}", **params)

                self._shared_indicators[key] = indicator.calculate(tf_data)

        return self._shared_indicators.get(key)
```

## Testing and Validation Guidelines

### Unit Testing Strategy Components

```python
import unittest
import pandas as pd
from datetime import datetime, timedelta

class TestMyStrategy(unittest.TestCase):
    def setUp(self):
        """Set up test data and strategy"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        self.test_data = pd.DataFrame({
            'open': range(100, 200),
            'high': range(105, 205),
            'low': range(95, 195),
            'close': range(102, 202),
            'volume': range(1000, 1100)
        }, index=dates)

        self.strategy = MyCustomStrategy()

    def test_indicator_calculation(self):
        """Test indicator calculations"""
        result = self.strategy.calculate_indicators(self.test_data)

        # Verify indicators are present
        self.assertIn('ma_20', result.columns)
        self.assertIn('ma_50', result.columns)
        self.assertIn('rsi', result.columns)

        # Verify no NaN in recent data
        self.assertFalse(result['ma_20'].iloc[-10:].isna().any())

    def test_signal_generation(self):
        """Test signal generation logic"""
        data_with_indicators = self.strategy.calculate_indicators(self.test_data)
        signal = self.strategy.generate_signal(data_with_indicators)

        if signal:
            self.assertIn(signal.type, [ActionType.BUY, ActionType.SELL])
            self.assertGreaterEqual(signal.strength, 0.0)
            self.assertLessEqual(signal.strength, 1.0)

    def test_position_sizing_metadata(self):
        """Test position sizing metadata is included"""
        data_with_indicators = self.strategy.calculate_indicators(self.test_data)
        signal = self.strategy.generate_signal(data_with_indicators)

        if signal and signal.type in [ActionType.BUY, ActionType.SELL]:
            self.assertIn('position_sizing', signal.metadata)
            self.assertIn('initial_size', signal.metadata['position_sizing'])
```

### Integration Testing

```python
def test_strategy_integration():
    """Test strategy integration with backtesting engine"""
    from backtest.engine import BacktestEngine
    from core.models import Symbol, TimeFrame

    # Create test data
    symbol = Symbol("BTCUSDT")
    timeframe = TimeFrame.DAILY

    # Initialize strategy
    strategy = MyCustomStrategy()

    # Run backtest
    engine = BacktestEngine()
    results = engine.run(strategy, symbol, timeframe, start_date='2023-01-01', end_date='2023-12-31')

    # Validate results
    assert results is not None
    assert len(results.trades) > 0
    assert results.total_return is not None
```

### Performance Validation

```python
def validate_strategy_performance(strategy: Strategy, data: pd.DataFrame) -> Dict[str, Any]:
    """Validate strategy performance metrics"""

    signals_df = strategy.generate_all_signals(data)

    metrics = {
        "total_signals": len(signals_df),
        "buy_signals": len(signals_df[signals_df['type'] == 'buy']),
        "sell_signals": len(signals_df[signals_df['type'] == 'sell']),
        "avg_signal_strength": signals_df['strength'].mean(),
        "signal_frequency": len(signals_df) / len(data),  # Signals per bar
    }

    # Validate reasonable ranges
    assert 0 <= metrics["signal_frequency"] <= 0.1, "Signal frequency too high"
    assert 0.3 <= metrics["avg_signal_strength"] <= 1.0, "Average signal strength out of range"

    return metrics
```

## Real Examples from Codebase

### GoldenCrossStrategy Analysis

The `GoldenCrossStrategy` in `/Users/bkchoi/Desktop/GitHub/team_survive/bt/backtest/strats/dolpha1.py` demonstrates several advanced patterns:

#### 1. Multi-Timeframe Analysis

```python
def check_mtf_touch(self, current_price: float) -> bool:
    """Check for confluence across multiple timeframes"""
    tolerance = 0.001

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

        # Calculate MA and check for price touch
        ma_value = self._calculate_ma_stored(df, period, f"{tf_key}_{period}")
        if abs(current_price - ma_value) / ma_value <= tolerance:
            return True

    return False
```

#### 2. Indicator Caching Implementation

```python
def _calculate_ma_stored(self, data: pd.DataFrame, period: int, key: str) -> pd.Series:
    """Cached moving average calculation"""
    if len(data) < period:
        return pd.Series(index=data.index, dtype=float)

    if not hasattr(self, '_ma_results'):
        self._ma_results = {}

    if key not in self._ma_results:
        self._ma_results[key] = MovingAverage(name=f"ma_{period}", length=period)

    ma_df = self._ma_results[key].calculate(data)
    return ma_df[f"ma_{period}"]
```

#### 3. Comprehensive Exit Strategy

```python
def calculate_exit_levels(self, entry_price: Decimal, last_high: Decimal) -> Dict[str, Decimal]:
    """Calculate Fibonacci-based exit levels"""
    price_range = last_high - entry_price

    tp1_level = Decimal(str(self.parameters["exit_levels"]["tp1_level"]))  # 0.382
    tp2_level = Decimal(str(self.parameters["exit_levels"]["tp2_level"]))  # 0.5

    return {
        "tp1": entry_price + (price_range * tp1_level),
        "tp2": entry_price + (price_range * tp2_level),
        "stop_loss": entry_price * Decimal("0.95")  # 5% stop loss
    }
```

#### 4. Scaling Position Management

```python
# In signal metadata
"position_sizing": {
    "initial_size": 0.01,   # 1% of capital
    "scale_factor": 1.4,    # 40% increase per entry
    "max_entries": 10       # Maximum 10 entries
}
```

### Signal Combiner Pattern

The `SignalCombiner` class shows how to combine multiple strategies:

```python
class SignalCombiner:
    def __init__(self, strategies: List[Strategy], weights: Optional[Dict[str, float]] = None):
        self.strategies = strategies
        self.weights = weights or {s.name: 1.0 for s in strategies}

    def combine_signals(self, signals: Dict[str, Signal]) -> Optional[Signal]:
        """Combine signals from multiple strategies with weighted averaging"""
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
```

## Advanced Patterns

### 1. Adaptive Strategy Parameters

```python
class AdaptiveStrategy(Strategy):
    def __init__(self, **kwargs):
        super().__init__("AdaptiveStrategy", kwargs.get('parameters'))
        self.performance_window = 30  # Days to track performance
        self.adaptation_frequency = 7  # Adapt every 7 days

    def adapt_parameters(self, recent_performance: Dict[str, float]) -> None:
        """Adapt strategy parameters based on recent performance"""

        win_rate = recent_performance.get('win_rate', 0.5)
        avg_return = recent_performance.get('avg_return', 0.0)

        # Adapt signal threshold based on win rate
        if win_rate < 0.4:
            self.parameters['signal_threshold'] = 0.8  # Be more selective
        elif win_rate > 0.6:
            self.parameters['signal_threshold'] = 0.6  # Be more aggressive

        # Adapt position sizing based on performance
        if avg_return < 0:
            # Reduce position size in poor performance
            self.parameters['position_sizing']['initial_size'] *= 0.8
        elif avg_return > 0.02:  # 2% avg return
            # Increase position size in good performance
            self.parameters['position_sizing']['initial_size'] *= 1.1
```

### 2. Portfolio-Aware Strategy

```python
class PortfolioAwareStrategy(Strategy):
    def generate_signal(self, data, position=None, portfolio=None):
        """Generate signals considering overall portfolio state"""

        if not portfolio:
            return self._generate_basic_signal(data)

        # Check portfolio concentration
        if portfolio.concentration_risk > 0.3:  # More than 30% in one asset
            return None  # Don't add more to concentrated positions

        # Check correlation with existing positions
        if self._high_correlation_with_portfolio(data, portfolio):
            return None  # Avoid correlated positions

        # Adjust position size based on portfolio volatility
        portfolio_vol = portfolio.calculate_volatility()
        adjusted_sizing = self._adjust_sizing_for_volatility(portfolio_vol)

        signal = self._generate_basic_signal(data)
        if signal:
            signal.metadata['position_sizing'] = adjusted_sizing

        return signal
```

### 3. Machine Learning Integration

```python
class MLEnhancedStrategy(Strategy):
    def __init__(self, model_path: str = None, **kwargs):
        super().__init__("MLEnhanced", kwargs.get('parameters'))
        self.model = self._load_ml_model(model_path) if model_path else None

    def _extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for ML model"""
        features = []

        # Technical indicators
        ma_short = data["close"].rolling(10).mean()
        ma_long = data["close"].rolling(30).mean()
        features.append((ma_short.iloc[-1] - ma_long.iloc[-1]) / ma_long.iloc[-1])

        # Price momentum
        momentum = (data["close"].iloc[-1] - data["close"].iloc[-5]) / data["close"].iloc[-5]
        features.append(momentum)

        # Volume ratio
        vol_ratio = data["volume"].iloc[-1] / data["volume"].rolling(20).mean().iloc[-1]
        features.append(vol_ratio)

        return np.array(features).reshape(1, -1)

    def generate_signal(self, data, position=None, portfolio=None):
        """Generate signal using ML model + traditional analysis"""

        # Traditional signal
        traditional_signal = self._generate_traditional_signal(data)

        if not self.model or not traditional_signal:
            return traditional_signal

        # ML enhancement
        features = self._extract_features(data)
        ml_probability = self.model.predict_proba(features)[0][1]  # Probability of success

        # Combine traditional and ML signals
        enhanced_strength = traditional_signal.strength * ml_probability

        return Signal(
            type=traditional_signal.type,
            strength=enhanced_strength,
            price=traditional_signal.price,
            metadata={
                **traditional_signal.metadata,
                "ml_probability": ml_probability,
                "traditional_strength": traditional_signal.strength
            }
        )
```

### 4. Event-Driven Strategy

```python
class EventDrivenStrategy(Strategy):
    def __init__(self, event_sources: List[str] = None, **kwargs):
        super().__init__("EventDriven", kwargs.get('parameters'))
        self.event_sources = event_sources or []
        self.recent_events = []

    def process_event(self, event: Dict[str, Any]) -> None:
        """Process external events (news, economic data, etc.)"""
        self.recent_events.append({
            "timestamp": datetime.now(),
            "type": event.get("type"),
            "sentiment": event.get("sentiment", 0),  # -1 to 1
            "importance": event.get("importance", 0.5)  # 0 to 1
        })

        # Keep only recent events (last 24 hours)
        cutoff = datetime.now() - timedelta(hours=24)
        self.recent_events = [e for e in self.recent_events if e["timestamp"] > cutoff]

    def calculate_event_sentiment(self) -> float:
        """Calculate aggregated sentiment from recent events"""
        if not self.recent_events:
            return 0.0

        weighted_sentiment = sum(
            event["sentiment"] * event["importance"]
            for event in self.recent_events
        )
        total_weight = sum(event["importance"] for event in self.recent_events)

        return weighted_sentiment / total_weight if total_weight > 0 else 0.0

    def generate_signal(self, data, position=None, portfolio=None):
        """Generate signal incorporating event sentiment"""

        # Technical analysis signal
        technical_signal = self._generate_technical_signal(data)

        # Event sentiment adjustment
        event_sentiment = self.calculate_event_sentiment()

        if technical_signal:
            # Boost signal strength with positive sentiment
            if technical_signal.type == ActionType.BUY and event_sentiment > 0.3:
                technical_signal.strength = min(1.0, technical_signal.strength + event_sentiment * 0.2)
            # Reduce signal strength with negative sentiment
            elif technical_signal.type == ActionType.BUY and event_sentiment < -0.3:
                technical_signal.strength = max(0.0, technical_signal.strength + event_sentiment * 0.2)

            technical_signal.metadata["event_sentiment"] = event_sentiment
            technical_signal.metadata["recent_events_count"] = len(self.recent_events)

        return technical_signal
```

## Best Practices Summary

### 1. Code Organization

- Inherit from the base `Strategy` class
- Use clear, descriptive method names
- Separate indicator calculation from signal generation
- Cache expensive calculations
- Use type hints and docstrings

### 2. Signal Quality

- Include comprehensive metadata for debugging
- Validate signals before generation
- Use meaningful signal strengths (0.0 to 1.0)
- Implement proper exit logic for risk management

### 3. Position Sizing

- Always include position sizing in signal metadata
- Use conservative initial sizes (0.5% - 2%)
- Limit maximum number of entries
- Consider portfolio-level risk

### 4. Multi-Timeframe Analysis

- Use higher timeframes for trend direction
- Use lower timeframes for precise entry/exit
- Check for confluence across timeframes
- Cache calculations to avoid redundancy

### 5. Testing and Validation

- Write unit tests for each strategy component
- Test with various market conditions
- Validate performance metrics
- Use proper data handling (avoid look-ahead bias)

### 6. Performance Optimization

- Cache indicator calculations
- Use incremental updates where possible
- Minimize data copying
- Profile strategy performance regularly

This comprehensive guide provides the foundation for developing robust, scalable trading strategies within the backtesting framework. Follow these patterns and practices to create effective strategies that can be thoroughly tested and validated.

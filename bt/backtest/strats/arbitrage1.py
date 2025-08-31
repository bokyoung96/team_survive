import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Optional, Dict, Any, List, Tuple
from collections import deque
from statsmodels.tsa.stattools import coint

from backtest.types import ActionType, Signal
from backtest.timeframe import MultiTimeframeData
from backtest.strategies import StreamingStrategy, TradingContext
from backtest.logger import get_logger


class PairsTradingStrategy(StreamingStrategy):
    """
    Pairs Trading Strategy with Statistical Arbitrage
    
    How Pairs Trading Works:
    -------------------------
    When z-score > +0.7 (spread is too HIGH):
      → SHORT symbol1 (expecting it to fall)
      → LONG symbol2 * hedge_ratio (expecting it to rise)
      → Profit when spread narrows back to mean
    
    When z-score < -0.7 (spread is too LOW):
      → LONG symbol1 (expecting it to rise)  
      → SHORT symbol2 * hedge_ratio (expecting it to fall)
      → Profit when spread widens back to mean
    
    Example with BTC/ETH pair (hedge_ratio=15):
    - If BTC=$50,000, ETH=$3,000, normal spread = 50,000 - 15*3,000 = $5,000
    - If spread rises to $8,000 (z-score > 0.7): SHORT BTC, LONG 15 ETH
    - When spread returns to $5,000: Close both positions for profit
    
    Entry Conditions:
    - Z-score > ±0.7 standard deviations for spread divergence
    - Cointegration test passed (p-value < 0.05)
    - Volatility filter: current vol ≤ 1.5x average volatility
    
    Exit Conditions:
    - Mean reversion: z-score returns to near 0 (±0.1)
    - Trailing stop-loss: 2.5% drawdown from peak/trough
    
    Note: Due to backtest limitations, we track the synthetic spread position
    rather than individual asset positions, but the logic represents true
    pairs trading with hedged long/short positions.
    """
    
    # NOTE: Strategy parameters
    ENTRY_Z_THRESHOLD = 0.7        # ±0.7 standard deviations for entry
    EXIT_Z_THRESHOLD = 0.1         # Exit zone around mean (±0.1)
    STOP_LOSS_PCT = 0.025          # 2.5% trailing stop-loss
    VOLATILITY_MULTIPLIER = 1.5    # Volatility filter threshold
    COINTEGRATION_P_VALUE = 0.05   # Significance level for cointegration
    MIN_LOOKBACK = 30              # Minimum lookback period
    MAX_LOOKBACK = 360             # Maximum lookback period
    PAIR_RESELECTION_FREQ = 30    # Bars between pair reselection
    MIN_DATA_POINTS = 100          # Minimum data for cointegration test
    VOLATILITY_WINDOW = 30         # Days for volatility calculation
    LOOKBACK_STEP_SIZE = 30        # Step size for lookback optimization
    Z_SCORE_STRENGTH_DIVISOR = 3  # Divisor for signal strength calculation
    
    def __init__(
        self, 
        data: Optional[MultiTimeframeData] = None, 
        crypto_universe: Optional[List[str]] = None, 
        **kwargs: Any
    ):
        super().__init__(name="PairsTradingStrategy", lookback_periods=self.MAX_LOOKBACK)
        
        self.data = data
        self._logger = get_logger(__name__)
        
        # NOTE: Strategy parameters with defaults
        self.parameters = {
            "crypto_universe": crypto_universe or [
                "BTC/USDT", "ETH/USDT", "BNB/USDT", "ADA/USDT",
                "SOL/USDT", "DOT/USDT", "LTC/USDT", "BCH/USDT",
                "DOGE/USDT", "ETC/USDT"
            ],
            "entry_z_threshold": self.ENTRY_Z_THRESHOLD,
            "exit_z_threshold": self.EXIT_Z_THRESHOLD,
            "stop_loss_pct": self.STOP_LOSS_PCT,
            "volatility_multiplier": self.VOLATILITY_MULTIPLIER
        }
        
        self._initialize_buffers()
        self._initialize_state()
        
        self._logger.info(
            f"Initialized pairs trading with {len(self.parameters['crypto_universe'])} assets"
        )
    
    def _initialize_buffers(self):
        """Initialize price data buffers for all assets"""
        self._price_buffers = {
            symbol: deque(maxlen=self.MAX_LOOKBACK) 
            for symbol in self.parameters["crypto_universe"]
        }
        # NOTE: Volatility window for trade filtering
        self._spread_volatility_buffer = deque(maxlen=self.VOLATILITY_WINDOW)
    
    def _initialize_state(self):
        # NOTE: Pair selection and cointegration state
        self._best_pair = None
        self._hedge_ratio = 1.0
        self._optimized_lookback_period = None
        
        # NOTE: Position and risk management state
        self._position_type = None   # 1 for long spread, -1 for short spread
        self._entry_spread = None
        self._highest_spread = None  # For trailing stop-loss (long)
        self._lowest_spread = None   # For trailing stop-loss (short)
    
    def update_indicators(self, context: TradingContext) -> None:
        current_bar_index = context.bar_index
        
        # NOTE: Update prices for all symbols in universe
        self._update_price_buffers(current_bar_index)
        
        # NOTE: Log data availability for debugging
        self._log_data_availability(current_bar_index)
        
        # NOTE: Select best pair periodically
        if self._should_reselect_pair(current_bar_index):
            self._select_and_optimize_pair()
    
    def _update_price_buffers(self, current_bar_index: int) -> None:
        for symbol in self.parameters["crypto_universe"]:
            if self.data and self.data.has_symbol(symbol, "1d"):
                symbol_data = self.data.get_symbol_data(symbol, "1d")
                if symbol_data is not None and current_bar_index < len(symbol_data):
                    price = float(symbol_data.iloc[current_bar_index]['close'])
                    self._price_buffers[symbol].append(price)
    
    def _log_data_availability(self, current_bar_index: int) -> None:
        if current_bar_index == self.MIN_DATA_POINTS:
            for symbol in self.parameters["crypto_universe"]:
                self._logger.info(f"{symbol}: {len(self._price_buffers[symbol])} prices")
            if hasattr(self.data, '_symbol_data'):
                self._logger.info(f"Data has symbols: {list(self.data._symbol_data.keys())}")
    
    def _should_reselect_pair(self, current_bar_index: int) -> bool:
        return current_bar_index % self.PAIR_RESELECTION_FREQ == 0 and current_bar_index > self.MIN_DATA_POINTS
    
    def _select_and_optimize_pair(self) -> None:
        result = self._select_best_pair()
        if result:
            self._best_pair = (result[0], result[1])
            self._hedge_ratio = result[2]
            self._optimized_lookback_period = self._optimize_lookback_period(
                result[0], result[1], result[2]
            )
            self._logger.info(
                f"Selected pair: {result[0]}-{result[1]}, "
                f"Hedge ratio: {result[2]:.3f}, "
                f"Lookback: {self._optimized_lookback_period}"
            )
        else:
            self._logger.warning("No cointegrated pairs found!")
    
    def _find_cointegrated_pairs(self) -> List[Tuple[str, str, float, float]]:
        cointegrated_pairs = []
        crypto_universe = self.parameters["crypto_universe"]
        
        for i, symbol1 in enumerate(crypto_universe):
            prices1 = list(self._price_buffers[symbol1])
            if len(prices1) < self.MIN_DATA_POINTS:
                continue
                
            for symbol2 in crypto_universe[i+1:]:
                prices2 = list(self._price_buffers[symbol2])
                if len(prices2) < self.MIN_DATA_POINTS:
                    continue
                
                # NOTE: Test for cointegration using log prices
                try:
                    log_prices1 = np.log(prices1[-self.MIN_DATA_POINTS:])
                    log_prices2 = np.log(prices2[-self.MIN_DATA_POINTS:])
                    _, p_value, _ = coint(log_prices1, log_prices2)
                    
                    if p_value < self.COINTEGRATION_P_VALUE:
                        # NOTE: Calculate hedge ratio using OLS on log prices
                        X = sm.add_constant(log_prices2)
                        model = sm.OLS(log_prices1, X).fit()
                        hedge_ratio = model.params[1]
                        
                        # NOTE: Calculate Sharpe ratio using log spread
                        log_spread = log_prices1 - hedge_ratio * log_prices2
                        returns = np.diff(log_spread)
                        if len(returns) > 0 and np.std(returns) > 0:
                            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                        else:
                            sharpe = 0
                        
                        cointegrated_pairs.append((symbol1, symbol2, hedge_ratio, sharpe))
                except Exception as e:
                    self._logger.debug(f"Cointegration test failed for {symbol1}-{symbol2}: {e}")
                    continue
        
        return cointegrated_pairs
    
    def _optimize_lookback_period(self, symbol1: str, symbol2: str, hedge_ratio: float) -> int:
        best_sharpe = -np.inf
        best_lookback = 60
        
        prices1 = np.array(list(self._price_buffers[symbol1]))
        prices2 = np.array(list(self._price_buffers[symbol2]))
        
        if len(prices1) < self.MAX_LOOKBACK or len(prices2) < self.MAX_LOOKBACK:
            return best_lookback
        
        for lookback in range(self.MIN_LOOKBACK, min(self.MAX_LOOKBACK + 1, len(prices1)), self.LOOKBACK_STEP_SIZE):
            try:
                # NOTE: Use log prices for consistency
                log_spread = np.log(prices1[-lookback:]) - hedge_ratio * np.log(prices2[-lookback:])
                returns = np.diff(log_spread)
                if len(returns) > 0 and np.std(returns) > 0:
                    sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_lookback = lookback
            except:
                continue
        
        return best_lookback
    
    def _select_best_pair(self) -> Optional[Tuple[str, str, float]]:
        """Select pair with highest in-sample Sharpe ratio"""
        cointegrated_pairs = self._find_cointegrated_pairs()
        
        if not cointegrated_pairs:
            return None
        
        # NOTE: Sort by Sharpe ratio and select best performing pair
        best = max(cointegrated_pairs, key=lambda x: x[3])
        return (best[0], best[1], best[2])  # symbol1, symbol2, hedge_ratio
    
    def _has_sufficient_data(self, context: TradingContext) -> bool:
        if not self._best_pair:
            return False
        
        symbol1, symbol2 = self._best_pair
        lookback = self._optimized_lookback_period or 60
        
        return (len(self._price_buffers[symbol1]) >= lookback and 
                len(self._price_buffers[symbol2]) >= lookback)
    
    def _calculate_volatility_filter(self, spread_returns: np.ndarray) -> bool:
        if len(self._spread_volatility_buffer) < self.VOLATILITY_WINDOW:
            return True  # Allow trading when insufficient volatility history
        
        current_vol = np.std(spread_returns[-self.VOLATILITY_WINDOW:]) if len(spread_returns) >= self.VOLATILITY_WINDOW else 0
        avg_vol = np.mean(list(self._spread_volatility_buffer))
        
        volatility_threshold = self.parameters["volatility_multiplier"] * avg_vol
        
        if avg_vol > 0 and current_vol > volatility_threshold:
            self._logger.debug(f"Trading suppressed: vol={current_vol:.4f} > threshold={volatility_threshold:.4f}")
            return False
        
        return True
    
    def _check_trailing_stop_loss(self, current_spread: float) -> bool:
        if self._entry_spread is None:
            return False
        
        stop_loss_pct = self.parameters["stop_loss_pct"]
        
        # NOTE: Long position stop-loss logic
        if self._position_type == 1:
            if self._highest_spread is not None:
                self._highest_spread = max(self._highest_spread, current_spread)
                if current_spread < self._highest_spread * (1 - stop_loss_pct):
                    self._logger.info(f"Trailing stop-loss triggered (long): {current_spread:.4f}")
                    return True
        
        # NOTE: Short position stop-loss logic
        elif self._position_type == -1:
            if self._lowest_spread is not None:
                self._lowest_spread = min(self._lowest_spread, current_spread)
                if current_spread > self._lowest_spread * (1 + stop_loss_pct):
                    self._logger.info(f"Trailing stop-loss triggered (short): {current_spread:.4f}")
                    return True
        
        return False
    
    def _calculate_z_score(self, symbol1: str, symbol2: str) -> Optional[float]:
        prices1 = self._price_buffers[symbol1]
        prices2 = self._price_buffers[symbol2]
        
        lookback = self._optimized_lookback_period or 60
        
        if len(prices1) < lookback or len(prices2) < lookback:
            return None
        
        # NOTE: Calculate spread using log prices (consistent with cointegration)
        p1 = np.array(list(prices1)[-lookback:])
        p2 = np.array(list(prices2)[-lookback:])
        log_spread = np.log(p1) - self._hedge_ratio * np.log(p2)
        
        # NOTE: Update volatility tracking
        if len(log_spread) > 1:
            returns = np.diff(log_spread)
            if len(returns) > 0:
                self._spread_volatility_buffer.append(np.std(returns))
        
        # NOTE: Calculate z-score
        mean = np.mean(log_spread)
        std = np.std(log_spread)
        
        if std < 1e-6:
            return None
        
        # NOTE: Current spread in log space for z-score
        current_log_spread = np.log(p1[-1]) - self._hedge_ratio * np.log(p2[-1])
        # NOTE: Actual spread in price space for position tracking
        current_spread = p1[-1] - self._hedge_ratio * p2[-1]
        
        # NOTE: Update spread tracking for trailing stop-loss
        if self._position_type is not None:
            if self._position_type == 1 and self._highest_spread is not None:
                self._highest_spread = max(self._highest_spread, current_spread)
            elif self._position_type == -1 and self._lowest_spread is not None:
                self._lowest_spread = min(self._lowest_spread, current_spread)
        
        z_score = (current_log_spread - mean) / std
        return z_score
    
    def process_bar(self, context: TradingContext) -> Optional[Signal]:
        if not self._has_sufficient_data(context):
            return None
        
        position = context.position
        if position and position.is_open:
            # NOTE: Check exit conditions first
            exit_signal = self._check_exit_conditions(context)
            if exit_signal:
                return exit_signal
        else:
            # NOTE: Check new entry
            return self._check_new_entry(context)
        
        return None
    
    def _check_exit_conditions(self, context: TradingContext) -> Optional[Signal]:
        symbol1, symbol2 = self._best_pair
        z_score = self._calculate_z_score(symbol1, symbol2)
        
        if z_score is None:
            return None
        
        # NOTE: Check trailing stop-loss first
        stop_loss_signal = self._check_stop_loss_exit(context, symbol1, symbol2)
        if stop_loss_signal:
            return stop_loss_signal
        
        # NOTE: Check mean reversion exit
        return self._check_mean_reversion_exit(context, z_score, symbol1, symbol2)
    
    def _check_stop_loss_exit(self, context: TradingContext, symbol1: str, symbol2: str) -> Optional[Signal]:
        prices1 = np.array(list(self._price_buffers[symbol1]))
        prices2 = np.array(list(self._price_buffers[symbol2]))
        
        if len(prices1) == 0 or len(prices2) == 0:
            return None
        
        current_spread = prices1[-1] - self._hedge_ratio * prices2[-1]
        
        if self._check_trailing_stop_loss(current_spread):
            self._reset_position_state()
            return Signal(
                type=ActionType.CLOSE,
                strength=1.0,
                price=context.get_current_price(),
                metadata={
                    "reason": "trailing_stop_loss",
                    "pair": f"{symbol1}-{symbol2}",
                    "action": "CLOSE ALL POSITIONS"
                }
            )
        
        return None
    
    def _check_mean_reversion_exit(self, context: TradingContext, z_score: float, 
                                   symbol1: str, symbol2: str) -> Optional[Signal]:
        exit_threshold = self.parameters["exit_z_threshold"]
        
        # NOTE: Exit at mean reversion (z-score returns to near 0)
        if self._position_type == 1 and abs(z_score) <= exit_threshold:
            # NOTE: Long position - exit when z-score returns to near 0
            self._reset_position_state()
            self._logger.info(f"PAIRS LONG EXIT: Close positions | z-score={z_score:.2f}")
            return Signal(
                type=ActionType.CLOSE,
                strength=1.0,
                price=context.get_current_price(),
                metadata={
                    "reason": f"pairs_long_exit_z={z_score:.2f}",
                    "pair": f"{symbol1}-{symbol2}",
                    "action": f"CLOSE LONG {symbol1}, CLOSE SHORT {symbol2}"
                }
            )
        
        elif self._position_type == -1 and abs(z_score) <= exit_threshold:
            # NOTE: Short position - exit when z-score returns to near 0
            self._reset_position_state()
            self._logger.info(f"PAIRS SHORT EXIT: Close positions | z-score={z_score:.2f}")
            return Signal(
                type=ActionType.CLOSE,
                strength=1.0,
                price=context.get_current_price(),
                metadata={
                    "reason": f"pairs_short_exit_z={z_score:.2f}",
                    "pair": f"{symbol1}-{symbol2}",
                    "action": f"CLOSE SHORT {symbol1}, CLOSE LONG {symbol2}"
                }
            )
        
        return None
    
    def _check_new_entry(self, context: TradingContext) -> Optional[Signal]:
        symbol1, symbol2 = self._best_pair
        z_score = self._calculate_z_score(symbol1, symbol2)
        
        if z_score is None:
            return None
        
        # NOTE: Check volatility filter before entry
        if not self._passes_volatility_filter(symbol1, symbol2):
            return None
        
        entry_threshold = self.parameters["entry_z_threshold"]
        
        # NOTE: Entry logic following paper thresholds
        if z_score > entry_threshold:
            # NOTE: Spread is too high - short it (expecting mean reversion down)
            return self._create_short_signal(context, z_score, symbol1, symbol2)
        
        elif z_score < -entry_threshold:
            # NOTE: Spread is too low - buy it (expecting mean reversion up)
            return self._create_long_signal(context, z_score, symbol1, symbol2)
        
        return None
    
    def _passes_volatility_filter(self, symbol1: str, symbol2: str) -> bool:
        prices1 = np.array(list(self._price_buffers[symbol1]))
        prices2 = np.array(list(self._price_buffers[symbol2]))
        
        if len(prices1) <= 1 or len(prices2) <= 1:
            return True
        
        # NOTE: Calculate spread returns using log prices for consistency
        log_spread = np.log(prices1) - self._hedge_ratio * np.log(prices2)
        spread_returns = np.diff(log_spread)
        
        return self._calculate_volatility_filter(spread_returns)
    
    def _create_short_signal(self, context: TradingContext, z_score: float, 
                           symbol1: str, symbol2: str) -> Signal:
        prices1 = np.array(list(self._price_buffers[symbol1]))
        prices2 = np.array(list(self._price_buffers[symbol2]))
        current_spread = prices1[-1] - self._hedge_ratio * prices2[-1]
        
        self._position_type = -1
        self._entry_spread = current_spread
        self._lowest_spread = current_spread
        self._highest_spread = None
        
        # NOTE: In real pairs trading:
        # - SHORT symbol1 (expecting it to decrease)
        # - LONG symbol2 with hedge_ratio amount (expecting it to increase)
        # - We profit when spread narrows (returns to mean)
        self._logger.info(
            f"PAIRS SHORT: {symbol1}↓ / {symbol2}↑ | z-score={z_score:.2f}, spread={current_spread:.4f}"
        )
        
        signal_period_id = f"{context.timestamp.strftime('%Y%m%d_%H%M%S')}"
        entry_timestamp = context.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        return Signal(
            type=ActionType.SHORT,
            strength=min(abs(z_score) / self.Z_SCORE_STRENGTH_DIVISOR, 1.0),
            price=context.get_current_price(),
            metadata={
                "reason": f"pairs_short_z={z_score:.2f}",
                "pair": f"{symbol1}-{symbol2}",
                "action": f"SHORT {symbol1}, LONG {symbol2}*{self._hedge_ratio:.4f}",
                "signal_period_id": signal_period_id,
                "entry_timestamp": entry_timestamp,
                "entry_count": 1
            }
        )
    
    def _create_long_signal(self, context: TradingContext, z_score: float, 
                          symbol1: str, symbol2: str) -> Signal:
        prices1 = np.array(list(self._price_buffers[symbol1]))
        prices2 = np.array(list(self._price_buffers[symbol2]))
        current_spread = prices1[-1] - self._hedge_ratio * prices2[-1]
        
        self._position_type = 1
        self._entry_spread = current_spread
        self._highest_spread = current_spread
        self._lowest_spread = None
        
        # NOTE: In real pairs trading:
        # - LONG symbol1 (expecting it to increase) 
        # - SHORT symbol2 with hedge_ratio amount (expecting it to decrease)
        # - We profit when spread widens (returns to mean)
        self._logger.info(
            f"PAIRS LONG: {symbol1}↑ / {symbol2}↓ | z-score={z_score:.2f}, spread={current_spread:.4f}"
        )
        
        signal_period_id = f"{context.timestamp.strftime('%Y%m%d_%H%M%S')}"
        entry_timestamp = context.timestamp.strftime('%Y-%m-%d %H:%M:%S')
        
        return Signal(
            type=ActionType.BUY,
            strength=min(abs(z_score) / self.Z_SCORE_STRENGTH_DIVISOR, 1.0),
            price=context.get_current_price(),
            metadata={
                "reason": f"pairs_long_z={z_score:.2f}",
                "pair": f"{symbol1}-{symbol2}",
                "action": f"LONG {symbol1}, SHORT {symbol2}*{self._hedge_ratio:.4f}",
                "signal_period_id": signal_period_id,
                "entry_timestamp": entry_timestamp,
                "entry_count": 1
            }
        )
    
    def _reset_position_state(self) -> None:
        self._position_type = None
        self._entry_spread = None
        self._highest_spread = None
        self._lowest_spread = None
    
    def reset_state(self) -> None:
        super().reset_state()
        
        # NOTE: Reset pair selection and cointegration state
        self._best_pair = None
        self._hedge_ratio = 1.0
        self._optimized_lookback_period = None
        
        # NOTE: Reset position and risk management state
        self._reset_position_state()
        
        # NOTE: Clear all buffers
        for symbol in self._price_buffers:
            self._price_buffers[symbol].clear()
        self._spread_volatility_buffer.clear()
        
        self._logger.info("Strategy state reset completed")
from abc import ABC, abstractmethod
from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, List, Any, Tuple
import pandas as pd
from tqdm import tqdm

from backtest.models import (
    Portfolio,
    BacktestResult
)
from backtest.types import ActionType, TransactionCost
from backtest.executors import OrderCreator, OrderValidator, OrderExecutor
from backtest.performance import PerformanceAnalyzer
from backtest.logger import get_logger


@dataclass
class BacktestContext:
    """Contains all the context needed for backtest execution."""
    portfolio: Portfolio
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    orders: List[Any]
    symbol: str
    current_prices: Dict[str, Decimal]
    timestamp: Any
    current_row: pd.Series
    

class SignalProcessor(ABC):
    """Abstract base class for signal processing strategies."""
    
    @abstractmethod
    def process_signal(
        self, 
        context: BacktestContext, 
        signal: Optional[Dict[str, Any]]
    ) -> bool:
        """Process a signal and return True if processing should continue."""
        pass


class BaseSignalProcessor(SignalProcessor):
    """Base implementation for signal processors with common functionality."""
    
    def __init__(self, order_creator: OrderCreator, order_validator: OrderValidator, 
                 order_executor: OrderExecutor, logger):
        self.order_creator = order_creator
        self.order_validator = order_validator
        self.order_executor = order_executor
        self.logger = logger
    
    def _execute_signal_order(
        self, 
        context: BacktestContext, 
        signal: Dict[str, Any]
    ) -> bool:
        """Execute an order based on signal."""
        # Handle both ActionType enum and string values
        if isinstance(signal['type'], ActionType):
            signal_type = signal['type']
        else:
            signal_type = ActionType(signal['type'])
        signal_strength = signal['strength']
        signal_metadata = signal['metadata']

        position_before = context.portfolio.get_position(context.symbol)
        position_was_open = position_before is not None and position_before.is_open

        order = self.order_creator.create_order(
            signal_type=signal_type,
            strength=signal_strength,
            symbol=context.symbol,
            timestamp=context.timestamp,
            portfolio=context.portfolio,
            metadata=signal_metadata
        )

        if not order:
            return False
            
        if not self.order_validator.validate_order(order, context.portfolio, context.current_row):
            return False
            
        success, _ = self.order_executor.execute_order(
            order,
            context.current_row,
            context.portfolio,
            context.timestamp
        )

        if success:
            context.orders.append(order)
            self._update_trades_if_position_closed(
                context, position_before, position_was_open
            )
            return True
            
        return False
    
    def _update_trades_if_position_closed(
        self, 
        context: BacktestContext, 
        position_before: Any, 
        position_was_open: bool
    ) -> None:
        """Update trades list if position was closed."""
        position_after = context.portfolio.get_position(context.symbol)
        position_is_open_after = position_after is not None and position_after.is_open

        if position_was_open and not position_is_open_after and position_before:
            context.trades.append({
                "entry_time": position_before.entry_time,
                "exit_time": context.timestamp,
                "entry_price": float(position_before.entry_price),
                "exit_price": float(context.current_row["close"]),
                "quantity": float(position_before.quantity),
                "side": position_before.side.value,
                "pnl": float(position_before.realized_pnl),
                "commission": float(position_before.commission),
                "slippage": float(position_before.slippage)
            })


class ExitSignalProcessor(BaseSignalProcessor):
    """Processes exit signals (TP/SL) from strategy."""
    
    def process_signal(
        self, 
        context: BacktestContext, 
        signal: Optional[Dict[str, Any]]
    ) -> bool:
        """Process exit signal - returns False to stop processing for this iteration."""
        if not signal:
            return True
            
        # Handle both ActionType enum and string values
        if isinstance(signal['type'], ActionType):
            signal_type = signal['type']
        else:
            signal_type = ActionType(signal['type'])
        
        # Handle exit signals with priority
        if signal_type == ActionType.CLOSE:
            success = self._execute_signal_order(context, signal)
            return not success  # Return False to stop processing if successful
            
        return True


class EntrySignalProcessor(BaseSignalProcessor):
    """Processes entry signals pre-generated by strategy."""
    
    def process_signal(
        self, 
        context: BacktestContext, 
        signal: Optional[Dict[str, Any]]
    ) -> bool:
        """Process entry signal from pre-generated signals."""
        if not signal:
            return True
            
        if not self._should_process_signal(context, signal):
            return True
            
        self._execute_signal_order(context, signal)
        return True  # Always continue processing after entry signals
    
    def _should_process_signal(
        self, 
        context: BacktestContext, 
        signal: Dict[str, Any]
    ) -> bool:
        """Check if signal should be processed based on metadata."""
        current_position = context.portfolio.get_position(context.symbol)
        metadata = signal.get('metadata', {})
        
        # Handle signal period transitions with open positions
        if current_position and current_position.is_open:
            signal_period_id = metadata.get('signal_period_id')
            position_signal_period = current_position.metadata.get('signal_period_id')
            
            # Allow entries from the same signal period, skip different periods
            if (signal_period_id and position_signal_period and 
                signal_period_id != position_signal_period):
                return False
                
        return True


class DataProcessor:
    """Handles data preparation and combination."""
    
    def __init__(self, logger):
        self.logger = logger
    
    def prepare_data(
        self, 
        signals: pd.DataFrame, 
        ohlcv_data: pd.DataFrame, 
        strategy: Any = None, 
        intraday_timeframe: str = "3m"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Prepare and combine signals with market data."""
        signals_indexed = self._index_signals(signals)
        combined_data = self._combine_data(
            signals_indexed, ohlcv_data, strategy, intraday_timeframe
        )
        return signals_indexed, combined_data
    
    def _index_signals(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Index signals by timestamp if needed."""
        if 'timestamp' in signals.columns:
            return signals.set_index('timestamp')
        return signals
    
    def _combine_data(
        self, 
        signals_indexed: pd.DataFrame, 
        ohlcv_data: pd.DataFrame, 
        strategy: Any, 
        intraday_timeframe: str
    ) -> pd.DataFrame:
        """Combine signals with market data."""
        if strategy and hasattr(strategy, 'data') and strategy.data:
            if intraday_timeframe in strategy.data:
                intraday_data = strategy.data[intraday_timeframe]
                # Use reindex with forward fill to propagate daily signals to intraday bars
                combined_data = intraday_data.join(
                    signals_indexed.reindex(intraday_data.index, method='ffill'), 
                    how='left'
                )
                self.logger.info(
                    f"Using {intraday_timeframe} data for intraday backtesting: {len(combined_data)} bars"
                )
            else:
                combined_data = ohlcv_data.join(signals_indexed, how='left')
                self.logger.warning(
                    f"Intraday timeframe {intraday_timeframe} not available, using daily data"
                )
        else:
            combined_data = ohlcv_data.join(signals_indexed, how='left')
            
        return combined_data


class EquityTracker:
    """Tracks equity curve during backtest."""
    
    def update_equity_curve(
        self, 
        context: BacktestContext
    ) -> None:
        """Update equity curve with current portfolio metrics."""
        metrics = context.portfolio.calculate_metrics(context.current_prices)
        
        context.equity_curve.append({
            "timestamp": context.timestamp,
            "total_value": float(metrics["total_value"]),
            "cash": float(metrics["cash"]),
            "position_count": metrics["position_count"],
            "realized_pnl": float(metrics["realized_pnl"]),
            "unrealized_pnl": float(metrics["unrealized_pnl"]),
            "drawdown": 0.0
        })


class FinalPositionHandler:
    """Handles final position closing and trade recording."""
    
    def close_final_position(
        self, 
        portfolio: Portfolio, 
        symbol: str, 
        combined_data: pd.DataFrame, 
        trades: List[Dict[str, Any]]
    ) -> None:
        """Close any remaining open position at the end of backtest."""
        final_position = portfolio.get_position(symbol)
        if not final_position or not final_position.is_open:
            return
            
        final_price = Decimal(str(combined_data.iloc[-1]["close"]))
        final_timestamp = combined_data.index[-1]

        pnl = portfolio.close_position(
            final_position,
            final_price,
            timestamp=final_timestamp
        )

        trades.append({
            "entry_time": final_position.entry_time,
            "exit_time": final_timestamp,
            "entry_price": float(final_position.entry_price),
            "exit_price": float(final_price),
            "quantity": float(final_position.quantity),
            "side": final_position.side.value,
            "pnl": float(pnl),
            "commission": float(final_position.commission),
            "slippage": float(final_position.slippage)
        })


class ResultBuilder:
    """Builds the final backtest result."""
    
    def __init__(self, analyzer: PerformanceAnalyzer):
        self.analyzer = analyzer
    
    def build_result(
        self, 
        portfolio: Portfolio,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        signals: pd.DataFrame,
        symbol: str,
        initial_capital: Decimal,
        transaction_cost: TransactionCost
    ) -> BacktestResult:
        """Build the final backtest result."""
        equity_df = self._build_equity_dataframe(equity_curve)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        signals_df = signals.copy() if not signals.empty else pd.DataFrame()
        
        metrics_dict = self._calculate_performance_metrics(
            equity_df, trades_df, initial_capital
        )
        
        return BacktestResult(
            portfolio=portfolio,
            equity_curve=equity_df,
            trades=trades_df,
            signals=signals_df,
            metrics=metrics_dict,
            metadata={
                "symbol": symbol,
                "initial_capital": float(initial_capital),
                "transaction_costs": {
                    "maker_fee": float(transaction_cost.maker_fee),
                    "taker_fee": float(transaction_cost.taker_fee),
                    "slippage": float(transaction_cost.slippage)
                }
            }
        )
    
    def _build_equity_dataframe(self, equity_curve: List[Dict[str, Any]]) -> pd.DataFrame:
        """Build equity DataFrame with drawdown calculations."""
        equity_df = pd.DataFrame(equity_curve)
        if equity_df.empty:
            return equity_df
            
        equity_df.set_index("timestamp", inplace=True)
        running_max = equity_df["total_value"].expanding().max()
        equity_df["drawdown"] = (
            equity_df["total_value"] - running_max) / running_max
            
        return equity_df
    
    def _calculate_performance_metrics(
        self, 
        equity_df: pd.DataFrame, 
        trades_df: pd.DataFrame, 
        initial_capital: Decimal
    ) -> Dict[str, Any]:
        """Calculate performance metrics."""
        if not equity_df.empty and not trades_df.empty:
            performance_metrics = self.analyzer.analyze_performance(
                equity_df,
                trades_df,
                float(initial_capital)
            )

            return {
                "total_return": performance_metrics.total_return,
                "annualized_return": performance_metrics.annualized_return,
                "sharpe_ratio": performance_metrics.sharpe_ratio,
                "sortino_ratio": performance_metrics.sortino_ratio,
                "max_drawdown": performance_metrics.max_drawdown,
                "win_rate": performance_metrics.win_rate,
                "profit_factor": performance_metrics.profit_factor,
                "total_trades": performance_metrics.total_trades
            }
        else:
            return {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }


class SignalExtractor:
    """Extracts signals from various sources."""
    
    def get_exit_signal(
        self, 
        strategy: Any, 
        current_position: Any, 
        ohlcv_data: pd.DataFrame, 
        timestamp: Any, 
        i: int, 
        intraday_timeframe: str
    ) -> Optional[Dict[str, Any]]:
        """Get exit signal (TP/SL) from strategy if position is open."""
        if not strategy:
            return None
            
        signal = self._generate_signal_for_position(
            strategy, current_position, ohlcv_data, timestamp, i, intraday_timeframe
        )
        
        if signal:
            return {
                'type': signal.type.value,
                'strength': signal.strength,
                'metadata': signal.metadata
            }
            
        return None
    
    def _generate_signal_for_position(
        self, 
        strategy: Any, 
        current_position: Any, 
        ohlcv_data: pd.DataFrame, 
        timestamp: Any, 
        i: int, 
        intraday_timeframe: str
    ) -> Optional[Any]:
        """Generate signal based on current position state."""
        # Check for exit signals if we have a position
        if current_position and current_position.is_open:
            return self._generate_exit_signal(
                strategy, ohlcv_data, timestamp, i, intraday_timeframe, current_position
            )
        
        # Check for entry signals if no position
        elif not current_position:
            return self._generate_entry_signal(
                strategy, ohlcv_data, timestamp, i, intraday_timeframe
            )
        
        return None
    
    def _generate_exit_signal(
        self, 
        strategy: Any, 
        ohlcv_data: pd.DataFrame, 
        timestamp: Any, 
        i: int, 
        intraday_timeframe: str, 
        current_position: Any
    ) -> Optional[Any]:
        """Generate exit signal."""
        data = self._get_strategy_data(strategy, ohlcv_data, timestamp, i, intraday_timeframe)
        return strategy.generate_signal(data, current_position)
    
    def _generate_entry_signal(
        self, 
        strategy: Any, 
        ohlcv_data: pd.DataFrame, 
        timestamp: Any, 
        i: int, 
        intraday_timeframe: str
    ) -> Optional[Any]:
        """Generate entry signal."""
        data = self._get_strategy_data(strategy, ohlcv_data, timestamp, i, intraday_timeframe)
        return strategy.generate_signal(data, None)
    
    def _get_strategy_data(
        self, 
        strategy: Any, 
        ohlcv_data: pd.DataFrame, 
        timestamp: Any, 
        i: int, 
        intraday_timeframe: str
    ) -> pd.DataFrame:
        """Get appropriate data slice for strategy."""
        if hasattr(strategy, 'data') and strategy.data and intraday_timeframe in strategy.data:
            daily_data_up_to_now = ohlcv_data[ohlcv_data.index <= timestamp]
            if daily_data_up_to_now.empty:
                daily_data_up_to_now = ohlcv_data.iloc[:1]
            return daily_data_up_to_now
        else:
            return ohlcv_data.iloc[:i+1]
    
    def extract_entry_signal(
        self, 
        current_row: pd.Series
    ) -> Optional[Dict[str, Any]]:
        """Extract entry signal from pre-generated signals DataFrame."""
        if pd.notna(current_row.get('type')):
            metadata = current_row.get('metadata', {})
            if isinstance(metadata, str):
                metadata = {}
                
            return {
                'type': current_row['type'],
                'strength': current_row.get('strength', 1.0),
                'metadata': metadata
            }
            
        return None


class BacktestEngine:
    """Main backtest engine with improved separation of concerns."""
    
    def __init__(
        self,
        transaction_cost: Optional[TransactionCost] = None,
        analyzer: Optional[PerformanceAnalyzer] = None
    ):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.order_creator = OrderCreator()
        self.order_validator = OrderValidator()
        self.order_executor = OrderExecutor(self.transaction_cost)
        
        # Initialize processors and handlers
        self.data_processor = DataProcessor(self.logger)
        self.equity_tracker = EquityTracker()
        self.final_position_handler = FinalPositionHandler()
        self.result_builder = ResultBuilder(self.analyzer)
        self.signal_extractor = SignalExtractor()
        
        # Initialize signal processors
        self.exit_signal_processor = ExitSignalProcessor(
            self.order_creator, self.order_validator, self.order_executor, self.logger
        )
        self.entry_signal_processor = EntrySignalProcessor(
            self.order_creator, self.order_validator, self.order_executor, self.logger
        )

    def run_backtest(
        self,
        signals: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
        initial_capital: Decimal,
        symbol: str,
        strategy: Any = None,
        intraday_timeframe: str = "3m"
    ) -> BacktestResult:
        """Run backtest with improved structure and separation of concerns."""
        # Validate inputs
        if signals.empty:
            raise ValueError("Signals DataFrame cannot be empty")
        if ohlcv_data.empty:
            raise ValueError("OHLCV data DataFrame cannot be empty")
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not symbol:
            raise ValueError("Symbol cannot be empty")
            
        self._log_backtest_start(symbol, initial_capital, signals, ohlcv_data)
        
        # Initialize state
        portfolio = Portfolio(initial_capital=initial_capital)
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        orders: List[Any] = []
        
        # Prepare data
        signals_indexed, combined_data = self.data_processor.prepare_data(
            signals, ohlcv_data, strategy, intraday_timeframe
        )
        
        # Run main backtest loop
        self._run_backtest_loop(
            combined_data, symbol, portfolio, equity_curve, trades, orders,
            ohlcv_data, strategy, intraday_timeframe
        )
        
        # Handle final position
        self.final_position_handler.close_final_position(
            portfolio, symbol, combined_data, trades
        )
        
        # Build and return result
        return self.result_builder.build_result(
            portfolio, equity_curve, trades, signals, symbol, 
            initial_capital, self.transaction_cost
        )
    
    def _log_backtest_start(
        self, 
        symbol: str, 
        initial_capital: Decimal, 
        signals: pd.DataFrame, 
        ohlcv_data: pd.DataFrame
    ) -> None:
        """Log backtest initialization information."""
        self.logger.info(
            f"Starting backtest for {symbol} with ${initial_capital} initial capital"
        )
        self.logger.info(
            f"Processing {len(signals)} signals and {len(ohlcv_data)} price bars"
        )
    
    def _run_backtest_loop(
        self, 
        combined_data: pd.DataFrame, 
        symbol: str, 
        portfolio: Portfolio, 
        equity_curve: List[Dict[str, Any]], 
        trades: List[Dict[str, Any]], 
        orders: List[Any],
        ohlcv_data: pd.DataFrame,
        strategy: Any,
        intraday_timeframe: str
    ) -> None:
        """Run the main backtest iteration loop."""
        iterator = tqdm(range(len(combined_data)), desc="Running backtest")
        
        for i in iterator:
            current_row = combined_data.iloc[i]
            timestamp = current_row.name
            current_prices = {symbol: Decimal(str(current_row["close"]))}
            
            # Create context for this iteration
            context = BacktestContext(
                portfolio=portfolio,
                equity_curve=equity_curve,
                trades=trades,
                orders=orders,
                symbol=symbol,
                current_prices=current_prices,
                timestamp=timestamp,
                current_row=current_row
            )
            
            # Update equity curve
            self.equity_tracker.update_equity_curve(context)
            
            # Process signals
            if not self._process_signals(context, ohlcv_data, strategy, i, intraday_timeframe):
                continue  # Skip to next iteration if signal processing indicates to do so
    
    def _process_signals(
        self, 
        context: BacktestContext, 
        ohlcv_data: pd.DataFrame, 
        strategy: Any, 
        i: int, 
        intraday_timeframe: str
    ) -> bool:
        """Process both exit and entry signals. Returns False to skip iteration."""
        current_position = context.portfolio.get_position(context.symbol)
        
        # Check for exit signals (TP/SL) if position is open
        exit_signal = self.signal_extractor.get_exit_signal(
            strategy, current_position, ohlcv_data, context.timestamp, i, intraday_timeframe
        )
        
        # Process exit signal with priority
        if exit_signal:
            should_continue = self.exit_signal_processor.process_signal(context, exit_signal)
            if not should_continue:
                return False  # Exit signal was processed, skip to next iteration
        
        # Process entry signals from pre-generated signals
        if not exit_signal:
            entry_signal = self.signal_extractor.extract_entry_signal(context.current_row)
            if entry_signal:
                self.entry_signal_processor.process_signal(context, entry_signal)
        
        return True


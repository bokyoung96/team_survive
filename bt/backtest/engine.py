from dataclasses import dataclass
from decimal import Decimal
from typing import Optional, Dict, List, Any
import pandas as pd
from tqdm import tqdm

from backtest.models import (
    Portfolio,
    BacktestResult,
    Order,
    Position
)
from backtest.types import ActionType, TransactionCost
from backtest.executors import OrderCreator, OrderValidator, OrderExecutor, SignalProcessor, SignalContext
from backtest.strategies import TradingContext, StreamingStrategy
from backtest.performance import PerformanceAnalyzer
from backtest.logger import get_logger


@dataclass
class BacktestContext:
    portfolio: Portfolio
    equity_curve: List[Dict[str, Any]]
    trades: List[Dict[str, Any]]
    orders: List[Any]
    symbol: str
    current_prices: Dict[str, Decimal]
    timestamp: Any
    current_row: pd.Series


class BacktestEngine:
    def __init__(
        self,
        transaction_cost: Optional[TransactionCost] = None,
        analyzer: Optional[PerformanceAnalyzer] = None,
        max_trades_limit: int = 5000
    ):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.max_trades_limit = max_trades_limit
        self.storage = None
        self.logger = get_logger(__name__)
        
        self.order_creator = OrderCreator()
        self.order_validator = OrderValidator()
        self.order_executor = OrderExecutor(self.transaction_cost)
        self.signal_processor = SignalProcessor(self.order_creator)

    
    
    def _validate_inputs(
        self,
        ohlcv_data: pd.DataFrame,
        initial_capital: Decimal,
        symbol: str,
        strategy: Any
    ) -> None:
        if ohlcv_data.empty:
            raise ValueError("OHLCV data DataFrame cannot be empty")
        if initial_capital <= 0:
            raise ValueError("Initial capital must be positive")
        if not symbol:
            raise ValueError("Symbol cannot be empty")
        if strategy is None:
            raise ValueError("Strategy cannot be None")
    
    
    
    def _close_final_position(
        self, 
        portfolio: Portfolio, 
        symbol: str, 
        data: pd.DataFrame, 
        trades: List[Dict[str, Any]]
    ) -> None:
        final_position = portfolio.get_position(symbol)
        if not final_position or not final_position.is_open:
            return
        
        final_price = Decimal(str(data.iloc[-1]["close"]))
        final_timestamp = data.index[-1]
        
        pnl = portfolio.close_position(final_position, final_price, timestamp=final_timestamp)
        
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
    
    def _build_result(
        self,
        portfolio: Portfolio,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        symbol: str,
        initial_capital: Decimal,
        strategy: Optional[Any] = None
    ) -> BacktestResult:
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)
            running_max = equity_df["total_value"].expanding().max()
            equity_df["drawdown"] = (equity_df["total_value"] - running_max) / running_max
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        signals_data = []
        if strategy and hasattr(strategy, 'signals_history'):
            for timestamp, signal in strategy.signals_history:
                signals_data.append({
                    "timestamp": timestamp,
                    "type": signal.type.value,
                    "strength": signal.strength,
                    "price": float(signal.price) if signal.price else None,
                    "quantity": float(signal.quantity) if signal.quantity else None,
                    "metadata": signal.metadata
                })
        signals_df = pd.DataFrame(signals_data)
        
        if not equity_df.empty and not trades_df.empty:
            performance_metrics = self.analyzer.analyze_performance(
                equity_df, trades_df, float(initial_capital)
            )
            metrics_dict = {
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
            metrics_dict = {
                "total_return": 0.0,
                "annualized_return": 0.0,
                "sharpe_ratio": 0.0,
                "sortino_ratio": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0
            }
        
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
                    "maker_fee": float(self.transaction_cost.maker_fee),
                    "taker_fee": float(self.transaction_cost.taker_fee),
                    "slippage": float(self.transaction_cost.slippage)
                }
            }
        )

    def run_backtest(
        self,
        strategy: StreamingStrategy,
        ohlcv_data: pd.DataFrame,
        initial_capital: Decimal,
        symbol: str,
        warmup_periods: int = 50,
        storage=None
    ) -> BacktestResult:
        self._validate_inputs(ohlcv_data, initial_capital, symbol, strategy)
        
        # NOTE: Set trade storage if provided
        self.storage = storage
        
        portfolio = Portfolio(initial_capital=initial_capital)
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        
        
        strategy.reset_state()
        
        # NOTE: Run backtest loop
        self._run_loop(
            strategy, 
            ohlcv_data, 
            symbol, 
            portfolio, 
            equity_curve, 
            trades, 
            warmup_periods
        )
        
        self._close_final_position(portfolio, symbol, ohlcv_data, trades)
        
        return self._build_result(portfolio, equity_curve, trades, symbol, initial_capital, strategy)
    
    def _run_loop(
        self,
        strategy: StreamingStrategy,
        ohlcv_data: pd.DataFrame,
        symbol: str,
        portfolio: Portfolio,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        warmup_periods: int
    ) -> None:        
        total_bars = len(ohlcv_data)
        
        # NOTE: Optimize progress bar updates
        update_frequency = max(100, total_bars // 1000)
        
        # NOTE: Prevent multiple signals in same bar
        last_signal_timestamp = None
        last_equity_date = None
        
        with tqdm(total=total_bars, desc="Running Backtest") as pbar:
            for i, (timestamp, bar) in enumerate(ohlcv_data.iterrows()):
                
                current_position = portfolio.get_position(symbol)
                
                lookback_data = strategy._get_lookback_data(ohlcv_data, i)
                
                # NOTE: Create trading context
                context = TradingContext(
                    portfolio=portfolio,
                    position=current_position,
                    bar_index=i,
                    timestamp=timestamp,
                    current_bar=bar,
                    lookback_data=lookback_data,
                    symbol=symbol
                )
                
                strategy.update_indicators(context)
                
                if i < warmup_periods:
                    current_date = timestamp.date()
                    if last_equity_date != current_date:
                        self._update_equity(context, equity_curve)
                        last_equity_date = current_date
                    if i % update_frequency == 0:
                        pbar.update(update_frequency)
                    continue
                
                signal = strategy.process_bar(context)
                
                # NOTE: Process signal if generated
                if signal:
                    if last_signal_timestamp == timestamp:
                        self.logger.warning(
                            f"Multiple signals in same bar blocked at {timestamp}. "
                            f"Only first signal processed."
                        )
                    else:
                        strategy.record_signal(context.timestamp, signal)
                        
                        self._process_signal(
                            signal, context, trades, strategy
                        )
                        
                        context.position = portfolio.get_position(symbol)
                            
                        last_signal_timestamp = timestamp
                
                current_date = timestamp.date()
                if last_equity_date != current_date or signal:
                    self._update_equity(context, equity_curve)
                    last_equity_date = current_date
                
                if i % update_frequency == 0:
                    pbar.update(update_frequency)
            
            remaining = total_bars - (total_bars // update_frequency) * update_frequency
            if remaining > 0:
                pbar.update(remaining)
    
    def _process_signal(
        self,
        signal,
        context: TradingContext,
        trades: List[Dict[str, Any]],
        strategy: Any = None
    ) -> None:        
        # NOTE: Create signal context
        signal_context = SignalContext(
            signal=signal,
            portfolio=context.portfolio,
            position=context.position,
            current_bar=context.current_bar,
            timestamp=context.timestamp,
            symbol=context.symbol
        )
        
        # NOTE: Process signal to get orders
        generated_orders = self.signal_processor.process_signal(signal_context)
        
        # NOTE: Execute each order
        for order in generated_orders:
            success, position = self.order_executor.execute_order(
                order, context.current_bar, context.portfolio, context.timestamp
            )
            
            if success:
                if order.side == ActionType.SELL:
                    self._record_completed_trades(order, context.portfolio, trades, strategy)
                    
                self.logger.info(
                    f"Executed {order.side.value} order: "
                    f"{order.quantity} @ {order.filled_price}"
                )
    
    def _update_equity(
        self, 
        context: TradingContext, 
        equity_curve: List[Dict[str, Any]]
    ) -> None:
        current_prices = {context.symbol: context.get_current_price()}
        metrics = context.portfolio.calculate_metrics(current_prices)
        
        equity_curve.append({
            "timestamp": context.timestamp,
            "total_value": float(metrics["total_value"]),
            "cash": float(metrics["cash"]),
            "position_count": metrics["position_count"],
            "realized_pnl": float(metrics["realized_pnl"]),
            "unrealized_pnl": float(metrics["unrealized_pnl"]),
            "drawdown": 0.0
        })
    
    def _record_completed_trades(
        self, 
        order: Order, 
        portfolio: Portfolio, 
        trades: List[Dict[str, Any]],
        strategy: Any = None
    ) -> None:
        # NOTE: Trade info stored in order metadata by executor
        if order.metadata and "entry_time" in order.metadata:
            trade_data = {
                "entry_time": order.metadata["entry_time"],
                "exit_time": order.timestamp,
                "entry_price": order.metadata["entry_price"],
                "exit_price": float(order.filled_price),
                "quantity": order.metadata["quantity"],
                "side": order.metadata["side"],
                "pnl": order.metadata["realized_pnl"],
                "commission": order.metadata["commission"],
                "slippage": order.metadata["slippage"],
                "metadata": order.metadata.get("position_metadata", {})
            }
            trades.append(trade_data)
            
            if len(trades) > self.max_trades_limit:
                trades.pop(0)
            
            # NOTE: Record all trades to storage if available
            if self.storage:
                from backtest.storage import Trade
                storage_trade = Trade(
                    timestamp=int(order.timestamp.timestamp() * 1000),
                    symbol=order.symbol,
                    side=order.metadata["side"],
                    price=float(order.metadata["entry_price"]),
                    quantity=float(order.metadata["quantity"]),
                    pnl=float(order.metadata["realized_pnl"]),
                    cumulative_pnl=float(portfolio.total_realized_pnl),
                    position=float(order.metadata["quantity"]),
                    trade_id=len(trades)
                )
                self.storage.record_trade(storage_trade)
            
            if strategy and hasattr(strategy, 'record_trade'):
                strategy.record_trade(order.timestamp, trade_data)
            
            self.logger.info(
                f"Recorded trade: {trade_data['side']} "
                f"{trade_data['quantity']} @ ${trade_data['entry_price']} -> "
                f"${order.filled_price}, PnL: ${trade_data['pnl']}"
            )

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
        analyzer: Optional[PerformanceAnalyzer] = None
    ):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.logger = get_logger(__name__)
        
        self.order_creator = OrderCreator()
        self.order_validator = OrderValidator()
        self.order_executor = OrderExecutor(self.transaction_cost)

    def run_backtest(
        self,
        strategy: Any,
        ohlcv_data: pd.DataFrame,
        initial_capital: Decimal,
        symbol: str,
        intraday_timeframe: str = "3m"
    ) -> BacktestResult:
        self._validate_inputs(ohlcv_data, initial_capital, symbol, strategy)
        
        portfolio = Portfolio(initial_capital=initial_capital)
        equity_curve: List[Dict[str, Any]] = []
        trades: List[Dict[str, Any]] = []
        orders: List[Any] = []
        
        primary_data = self._select_primary_data(ohlcv_data, strategy, intraday_timeframe)
        
        self._run_backtest_loop(
            primary_data, symbol, portfolio, 
            equity_curve, trades, orders, ohlcv_data, strategy, intraday_timeframe
        )
        
        self._close_final_position(portfolio, symbol, primary_data, trades)
        
        return self._build_result(
            portfolio, equity_curve, trades, symbol, initial_capital
        )
    
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
    
    
    def _select_primary_data(
        self, 
        ohlcv_data: pd.DataFrame, 
        strategy: Any, 
        intraday_frequency: str
    ) -> pd.DataFrame:
        if strategy and hasattr(strategy, 'data') and strategy.data:
            if intraday_frequency in strategy.data:
                self.logger.info(f"Using {intraday_frequency} data")
                return strategy.data[intraday_frequency]
        
        self.logger.info("Using daily data")
        return ohlcv_data
    
    def _run_backtest_loop(
        self,
        primary_data: pd.DataFrame,
        symbol: str,
        portfolio: Portfolio,
        equity_curve: List[Dict[str, Any]],
        trades: List[Dict[str, Any]],
        orders: List[Any],
        ohlcv_data: pd.DataFrame,
        strategy: Any,
        intraday_timeframe: str
    ) -> None:
        for i, (timestamp, row) in enumerate(tqdm(primary_data.iterrows(), desc="Running backtest")):
            context = BacktestContext(
                portfolio=portfolio,
                equity_curve=equity_curve,
                trades=trades,
                orders=orders,
                symbol=symbol,
                current_prices={symbol: Decimal(str(row["close"]))},
                timestamp=timestamp,
                current_row=row
            )
            
            self._update_equity_curve(context)
            
            current_position = portfolio.get_position(symbol)
            
            # Get current data slice for strategy
            data_slice = ohlcv_data.iloc[:i+1]
            
            # Generate signal from strategy with current position context
            signal = strategy.generate_signal(data_slice, current_position, portfolio)
            
            if signal:
                signal_dict = {
                    'type': signal.type.value if hasattr(signal.type, 'value') else signal.type,
                    'strength': signal.strength,
                    'metadata': signal.metadata
                }
                
                # Process the signal (entry or exit)
                if signal.type in [ActionType.BUY, ActionType.ENTRY]:
                    self._process_entry_signal(context, signal_dict)
                elif signal.type in [ActionType.SELL, ActionType.EXIT, ActionType.CLOSE]:
                    self._process_exit_signal(context, signal_dict)
    
    def _update_equity_curve(self, context: BacktestContext) -> None:
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
    
    
    
    
    def _process_exit_signal(self, context: BacktestContext, signal: Dict[str, Any]) -> bool:
        signal_type = self._normalize_signal_type(signal['type'])
        
        if signal_type == ActionType.CLOSE:
            return self._execute_signal_order(context, signal)
        
        return False
    
    def _process_entry_signal(self, context: BacktestContext, signal: Dict[str, Any]) -> bool:
        if not self._should_process_entry_signal(context, signal):
            return False
        
        return self._execute_signal_order(context, signal)
    
    def _should_process_entry_signal(
        self, 
        context: BacktestContext, 
        signal: Dict[str, Any]
    ) -> bool:
        current_position = context.portfolio.get_position(context.symbol)
        metadata = signal.get('metadata', {})
        
        if current_position and current_position.is_open:
            signal_period_id = metadata.get('signal_period_id')
            position_signal_period = current_position.metadata.get('signal_period_id')
            
            if (signal_period_id and position_signal_period and 
                signal_period_id != position_signal_period):
                return False
        
        return True
    
    def _execute_signal_order(self, context: BacktestContext, signal: Dict[str, Any]) -> bool:
        signal_type = self._normalize_signal_type(signal['type'])
        
        order = self.order_creator.create_order(
            signal_type=signal_type,
            strength=signal['strength'],
            symbol=context.symbol,
            timestamp=context.timestamp,
            portfolio=context.portfolio,
            metadata=signal['metadata']
        )
        
        if not order:
            return False
        
        if not self.order_validator.validate_order(order, context.portfolio, context.current_row):
            return False
        
        success, _ = self.order_executor.execute_order(
            order, context.current_row, context.portfolio, context.timestamp
        )
        
        if success:
            context.orders.append(order)
            return True
        
        return False
    
    def _normalize_signal_type(self, signal_type) -> ActionType:
        if isinstance(signal_type, ActionType):
            return signal_type
        return ActionType(signal_type)
    
    
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
        initial_capital: Decimal
    ) -> BacktestResult:
        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)
            running_max = equity_df["total_value"].expanding().max()
            equity_df["drawdown"] = (equity_df["total_value"] - running_max) / running_max
        
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        signals_df = pd.DataFrame()  # No pre-generated signals in new approach
        
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
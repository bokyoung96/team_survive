from decimal import Decimal
from typing import Optional
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


class BacktestEngine:
    def __init__(
        self,
        transaction_cost: Optional[TransactionCost] = None,
        analyzer: Optional[PerformanceAnalyzer] = None
    ):
        self.transaction_cost = transaction_cost or TransactionCost()
        self.analyzer = analyzer or PerformanceAnalyzer()
        self.order_creator = OrderCreator()
        self.order_validator = OrderValidator()
        self.order_executor = OrderExecutor(self.transaction_cost)
        self.logger = get_logger(__name__)

    def run_backtest(
        self,
        signals: pd.DataFrame,
        ohlcv_data: pd.DataFrame,
        initial_capital: Decimal,
        symbol: str
    ) -> BacktestResult:
        self.logger.info(
            f"Starting backtest for {symbol} with ${initial_capital} initial capital")
        self.logger.info(
            f"Processing {len(signals)} signals and {len(ohlcv_data)} price bars")

        portfolio = Portfolio(initial_capital=initial_capital)

        equity_curve = []
        trades = []
        orders = []

        if 'timestamp' in signals.columns:
            signals_indexed = signals.set_index('timestamp')
        else:
            signals_indexed = signals

        combined_data = ohlcv_data.join(signals_indexed, how='left')

        iterator = tqdm(range(len(combined_data)), desc="Running backtest")

        for i in iterator:
            current_row = combined_data.iloc[i]
            timestamp = current_row.name

            current_prices = {symbol: Decimal(str(current_row["close"]))}
            metrics = portfolio.calculate_metrics(current_prices)

            equity_curve.append({
                "timestamp": timestamp,
                "total_value": float(metrics["total_value"]),
                "cash": float(metrics["cash"]),
                "position_count": metrics["position_count"],
                "realized_pnl": float(metrics["realized_pnl"]),
                "unrealized_pnl": float(metrics["unrealized_pnl"]),
                "drawdown": 0.0
            })

            if pd.notna(current_row.get('type')):
                signal_type = ActionType(current_row['type'])
                signal_strength = current_row.get('strength', 1.0)
                signal_metadata = current_row.get('metadata', {})

                position_before = portfolio.get_position(symbol)
                position_was_open = position_before is not None and position_before.is_open

                order = self.order_creator.create_order(
                    signal_type=signal_type,
                    strength=signal_strength,
                    symbol=symbol,
                    timestamp=timestamp,
                    portfolio=portfolio,
                    metadata=signal_metadata
                )

                if order:
                    if self.order_validator.validate_order(order, portfolio, current_row):
                        success, new_position = self.order_executor.execute_order(
                            order,
                            current_row,
                            portfolio,
                            timestamp
                        )

                        if success:
                            orders.append(order)

                            position_after = portfolio.get_position(symbol)
                            position_is_open_after = position_after is not None and position_after.is_open

                            if position_was_open and not position_is_open_after and position_before:
                                trades.append({
                                    "entry_time": position_before.entry_time,
                                    "exit_time": timestamp,
                                    "entry_price": float(position_before.entry_price),
                                    "exit_price": float(current_row["close"]),
                                    "quantity": float(position_before.quantity),
                                    "side": position_before.side.value,
                                    "pnl": float(position_before.realized_pnl),
                                    "commission": float(position_before.commission),
                                    "slippage": float(position_before.slippage)
                                })

        final_position = portfolio.get_position(symbol)
        if final_position and final_position.is_open:
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

        equity_df = pd.DataFrame(equity_curve)
        if not equity_df.empty:
            equity_df.set_index("timestamp", inplace=True)

            running_max = equity_df["total_value"].expanding().max()
            equity_df["drawdown"] = (
                equity_df["total_value"] - running_max) / running_max

        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        signals_df = signals.copy() if not signals.empty else pd.DataFrame()

        if not equity_df.empty and not trades_df.empty:
            performance_metrics = self.analyzer.analyze_performance(
                equity_df,
                trades_df,
                float(initial_capital)
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

        result = BacktestResult(
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
        return result

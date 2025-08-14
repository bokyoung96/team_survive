"""Backtesting module for trading strategies."""
from backtest.engine import BacktestEngine
from backtest.models import (
    Order,
    Position,
    Portfolio,
    BacktestResult
)
from backtest.types import ActionType, Signal, TransactionCost
from backtest.strategies import Strategy
from backtest.executors import OrderExecutor
from backtest.performance import PerformanceAnalyzer

# Backward compatibility
SignalType = ActionType

__all__ = [
    'BacktestEngine',
    'Order',
    'Position',
    'Portfolio',
    'BacktestResult',
    'TransactionCost',
    'Strategy',
    'Signal',
    'ActionType',
    'SignalType',  # For backward compatibility
    'OrderExecutor',
    'PerformanceAnalyzer'
]
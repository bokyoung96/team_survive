"""
Simple main script for running backtests.
Demonstrates the core backtesting workflow with real data.
"""

import pandas as pd
from decimal import Decimal
from datetime import datetime

from backtest.engine import BacktestEngine
from backtest.strategies import GoldenCrossStrategy
from backtest.types import TransactionCost

# Import data loader and models from core
from ..core.loader import DataLoader
from ..core.models import Symbol, TimeFrame, TimeRange, DataType, Exchange
from ..core import Factory
from pathlib import Path


def generate_signals(strategy, data: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals from strategy efficiently."""
    return strategy.generate_all_signals(data)


def run_simple_backtest():
    """Run a basic backtest example using real data."""
    print("Loading data...")
    
    # Setup data loader
    exchange = Exchange(id="binance")
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    # Define symbol and timeframe
    symbol = Symbol.from_string("BTC/USDT")
    timeframe = TimeFrame.D1
    time_range = TimeRange.days(365)  # Last 365 days
    
    # Load real market data
    data = loader.load(
        symbol=symbol,
        timeframe=timeframe,
        data_type=DataType.OHLCV,
        time_range=time_range
    )
    
    print(f"Loaded {len(data)} price bars for {symbol.base}/{symbol.quote}")
    
    # Setup strategy
    strategy = GoldenCrossStrategy()
    print(f"Using strategy: {strategy.name}")
    
    # Generate signals
    print("Generating signals...")
    signals = generate_signals(strategy, data)
    print(f"Generated {len(signals)} signals")
    
    if signals.empty:
        print("No signals generated. Try with longer data period.")
        return
    
    # Setup backtest
    engine = BacktestEngine(
        transaction_cost=TransactionCost(
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001")
        )
    )
    
    # Run backtest
    print("Running backtest...")
    result = engine.run_backtest(
        signals=signals,
        ohlcv_data=data,
        initial_capital=Decimal("10000"),
        symbol=f"{symbol.base}{symbol.quote}",
        progress_bar=True
    )
    
    # Show results
    print("\nBacktest Results:")
    print(f"Total Return: {result.metrics['total_return']:.2%}")
    print(f"Sharpe Ratio: {result.metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")
    print(f"Total Trades: {result.metrics['total_trades']}")
    print(f"Win Rate: {result.metrics['win_rate']:.2%}")
    
    return result


if __name__ == "__main__":
    try:
        result = run_simple_backtest()
        print("\nBacktest completed successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have price data or check your strategy configuration.")
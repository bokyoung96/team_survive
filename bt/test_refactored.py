#!/usr/bin/env python3
"""Test script to verify Entry #1 repetition bug is fixed."""

from decimal import Decimal
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

from core.factory import Factory
from core.models import Symbol, TimeFrame, TimeRange, Exchange, MarketType
from core.loader import DataLoader
from utils import KST
from backtest.timeframe import MultiTimeframeData
from backtest.strats.dolpha1 import GoldenCrossStrategy
from backtest.types import TransactionCost
from backtest.engine import BacktestEngine


def main():
    print("Starting test for Entry #1 repetition bug...")
    
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    symbol = Symbol.from_string("BTC/USDT:USDT")
    # Shorter date range for quick test
    start_date = KST.localize(datetime(2023, 1, 1))
    end_date = KST.localize(datetime(2024, 1, 1))  # 1 year
    date_range = TimeRange(start_date, end_date)
    
    data = (MultiTimeframeData(loader)
            .add(symbol, TimeFrame.D1, date_range)
            .add(symbol, TimeFrame.M3, date_range)
            .add(symbol, TimeFrame.M30, date_range)
            .add(symbol, TimeFrame.H1, date_range))
    
    strategy = GoldenCrossStrategy(data=data)
    
    print(f"Generating signals...")
    signals = strategy.generate_all_signals(data["1d"])
    print(f"Generated {len(signals)} signals")
    
    # Filter to first few signals for quick test
    if len(signals) > 5:
        signals = signals.head(5)
        print(f"Using first 5 signals for quick test")
    
    engine = BacktestEngine(
        transaction_cost=TransactionCost(
            maker_fee=Decimal("0.000"),
            taker_fee=Decimal("0.000"),
            slippage=Decimal("0.000")
        )
    )
    
    print("\n--- Running backtest ---")
    result = engine.run_backtest(
        signals=signals,
        ohlcv_data=data["1d"],
        initial_capital=Decimal("10000"),
        symbol=f"{symbol.base}{symbol.quote}",
        strategy=strategy
    )
    
    print("\n--- Test Results ---")
    trades = result.trades
    if not trades.empty:
        print(f"Total trades: {len(trades)}")
        print("\nTrade details:")
        for idx, trade in trades.iterrows():
            print(f"Trade {idx+1}:")
            print(f"  Entry: {trade['entry_time']:%Y-%m-%d %H:%M}")
            print(f"  Exit: {trade['exit_time']:%Y-%m-%d %H:%M}")
            print(f"  PnL: ${trade['pnl']:.2f}")
    else:
        print("No trades executed")
    
    # Check logs for Entry patterns
    print("\n--- Checking for Entry #1 repetition ---")
    print("If you see multiple Entry #1 without Entry #2, #3, etc., the bug persists.")
    print("If you see Entry #1, #2, #3... up to #10, then EXIT, the bug is fixed.")
    
    return result


if __name__ == "__main__":
    result = main()
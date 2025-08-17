from decimal import Decimal
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.factory import Factory
from core.models import Symbol, TimeFrame, TimeRange, Exchange, MarketType
from core.loader import DataLoader
from utils import KST
from backtest.timeframe import MultiTimeframeData
from backtest.strats.dolpha1 import GoldenCrossStrategy

def debug_signals():
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    symbol = Symbol.from_string("BTC/USDT:USDT")
    start_date = KST.localize(datetime(2022, 1, 1))
    end_date = KST.localize(datetime(2025, 8, 15))
    date_range = TimeRange(start_date, end_date)
    
    data = (MultiTimeframeData(loader)
            .add(symbol, TimeFrame.D1, date_range)
            .add(symbol, TimeFrame.M3, date_range)
            .add(symbol, TimeFrame.M30, date_range)
            .add(symbol, TimeFrame.H1, date_range))
    
    strategy = GoldenCrossStrategy(data=data)
    
    print("Generating signals...")
    signals = strategy.generate_all_signals(data["1d"])
    print(f"Generated {len(signals)} signals")
    
    if not signals.empty:
        print("\n=== SIGNAL STRUCTURE ===")
        print(f"Columns: {list(signals.columns)}")
        print(f"Index type: {type(signals.index)}")
        print(f"First 5 signals:")
        print(signals.head())
        
        print("\n=== SIGNAL TYPES ===")
        print(signals['type'].value_counts())
        
        print("\n=== SIGNAL METADATA SAMPLE ===")
        if 'metadata' in signals.columns:
            print(signals['metadata'].iloc[0])
        
        print("\n=== DAILY DATA SAMPLE ===")
        daily_data = data["1d"]
        print(f"Daily data shape: {daily_data.shape}")
        print(f"Daily data columns: {list(daily_data.columns)}")
        print(f"Daily data index type: {type(daily_data.index)}")
        print("Sample daily data:")
        print(daily_data.head())
        
        # Check if signals are being joined properly
        print("\n=== SIGNAL JOIN TEST ===")
        combined = daily_data.join(signals, how='left')
        print(f"Combined data shape: {combined.shape}")
        print(f"Combined data columns: {list(combined.columns)}")
        
        # Count non-null signal entries
        signal_mask = combined['type'].notna()
        print(f"Non-null signals in combined data: {signal_mask.sum()}")
        
        if signal_mask.sum() > 0:
            print("\nFirst few rows with signals:")
            print(combined[signal_mask].head())
    else:
        print("No signals generated!")

if __name__ == "__main__":
    debug_signals()
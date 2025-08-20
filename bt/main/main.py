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
from backtest.strats.dolpha2 import GoldenCrossOnlyStrategy
from backtest.types import TransactionCost
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.plot import create_backtest_report
from backtest.storage import TradeStorage


def main(strategy_choice=None):
    if strategy_choice is None:
        try:
            choice = input("Enter strategy number (1 or 2): ").strip()
        except EOFError:
            choice = "2"
    else:
        choice = str(strategy_choice)
    
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    symbol = Symbol.from_string("BTC/USDT:USDT")
    start_date = KST.localize(datetime(2023, 1, 1))
    end_date = KST.localize(datetime(2025, 8, 18))
    date_range = TimeRange(start_date, end_date)
    
    if choice == "1":
        # NOTE: Dolpha1
        data = (MultiTimeframeData(loader)
                .add(symbol, TimeFrame.D1, date_range)
                .add(symbol, TimeFrame.M3, date_range)
                .add(symbol, TimeFrame.M30, date_range)
                .add(symbol, TimeFrame.H1, date_range))
        
        strategy = GoldenCrossStrategy(data=data)
        strategy_name = "GoldenCrossStrategy"
        
    elif choice == "2":
        # NOTE: Dolpha2
        data = (MultiTimeframeData(loader)
                .add(symbol, TimeFrame.D1, date_range))
        
        strategy = GoldenCrossOnlyStrategy()
        strategy_name = "GoldenCrossOnlyStrategy"
        
    else:
        print("Invalid choice. Defaulting to Dolpha2.")
        data = (MultiTimeframeData(loader)
                .add(symbol, TimeFrame.D1, date_range))
        
        strategy = GoldenCrossOnlyStrategy()
        strategy_name = "GoldenCrossOnlyStrategy"
    
    engine = BacktestEngine(
        transaction_cost=TransactionCost(
            maker_fee=Decimal("0.000"),
            taker_fee=Decimal("0.000"),
            slippage=Decimal("0.000")
        )
    )
    
    storage = TradeStorage(base_dir="bt_results")
    session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage.initialize_session(session_id)
    
    print(f"Running real-time backtest...")
    print(f"Session ID: {session_id}")
    
    result = engine.run_backtest(
        strategy=strategy,
        ohlcv_data=data["1d"],
        initial_capital=Decimal("100000000"),
        symbol=f"{symbol.base}{symbol.quote}"
    )
    
    if hasattr(result, 'trades') and not result.trades.empty:
        print(f"Saving {len(result.trades)} trades to disk...")
    
    analyzer = PerformanceAnalyzer()
    detailed_metrics = analyzer.analyze_performance(
        equity_curve=result.equity_curve,
        trades=result.trades,
        initial_capital=float(result.portfolio.initial_capital)
    )
    
    print(analyzer.generate_report(detailed_metrics))
    
    try:
        create_backtest_report(
            result=result,
            benchmark_data=data["1d"],
            strategy_name=strategy_name,
            output_dir="bt_results",
            show_plots=False,
            session_id=session_id
        )
        print(f"\nPlots saved to bt_results/")
        print(f"  - performance_{session_id.split('_', 1)[1]}.png")
        print(f"  - trades_{session_id.split('_', 1)[1]}.png")
        print(f"Session data: bt_results/{session_id}/")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
    
    storage.close()
    return result


if __name__ == "__main__":
    result = main()

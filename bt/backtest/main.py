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
from backtest.types import TransactionCost
from backtest.engine import BacktestEngine
from backtest.performance import PerformanceAnalyzer
from backtest.plot import create_comprehensive_report_plots


def main():
    print("Setting up backtest...")
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)

    symbol = Symbol.from_string("BTC/USDT:USDT")

    print("Loading multi-timeframe data...")
    start_date = KST.localize(datetime(2022, 1, 1))
    end_date = KST.localize(datetime(2025, 8, 15))
    date_range = TimeRange(start_date, end_date)
    
    data = (MultiTimeframeData(loader)
            .add(symbol, TimeFrame.D1, date_range)
            .add(symbol, TimeFrame.M3, date_range)
            .add(symbol, TimeFrame.M30, date_range)
            .add(symbol, TimeFrame.H1, date_range))

    print(
        f"Loaded {len(data['1d'])} daily bars for {symbol.base}/{symbol.quote}")

    strategy = GoldenCrossStrategy(data=data)

    signals = strategy.generate_all_signals()
    print(f"Generated {len(signals)} signals")

    if signals.empty:
        print("No signals generated. Need more data for moving averages.")
        return

    engine = BacktestEngine(
        transaction_cost=TransactionCost(
            maker_fee=Decimal("0.001"),
            taker_fee=Decimal("0.001")
        )
    )

    result = engine.run_backtest(
        signals=signals,
        ohlcv_data=data["1d"],
        initial_capital=Decimal("10000"),
        symbol=f"{symbol.base}{symbol.quote}"
    )

    print("\n=== Generating Performance Report ===")
    analyzer = PerformanceAnalyzer()
    detailed_metrics = analyzer.analyze_performance(
        equity_curve=result.equity_curve,
        trades=result.trades,
        initial_capital=float(result.portfolio.initial_capital)
    )
    report = analyzer.generate_report(detailed_metrics)
    print(report)

    print("\n=== Generating Visualizations ===")
    try:
        plots = create_comprehensive_report_plots(
            result=result,
            metrics=detailed_metrics,
            price_data=data["1d"]
        )
        print(f"Generated {len(plots)} visualization plots in backtest_results/")
    except Exception as e:
        print(f"Could not generate plots: {e}")
        print("Install required packages: pip install matplotlib seaborn")

    return result


if __name__ == "__main__":
    try:
        result = main()
        if result:
            print("\nBacktest completed successfully!")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
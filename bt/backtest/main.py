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
from backtest.plot import create_backtest_report


def main():
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    symbol = Symbol.from_string("BTC/USDT:USDT")
    start_date = KST.localize(datetime(2020, 1, 1))
    end_date = KST.localize(datetime(2025, 8, 18))
    date_range = TimeRange(start_date, end_date)
    
    data = (MultiTimeframeData(loader)
            .add(symbol, TimeFrame.D1, date_range)
            .add(symbol, TimeFrame.M3, date_range)
            .add(symbol, TimeFrame.M30, date_range)
            .add(symbol, TimeFrame.H1, date_range))
    
    strategy = GoldenCrossStrategy(data=data)
    
    engine = BacktestEngine(
        transaction_cost=TransactionCost(
            maker_fee=Decimal("0.000"),
            taker_fee=Decimal("0.000"),
            slippage=Decimal("0.000")
        )
    )
    
    print(f"Running real-time backtest...")
    result = engine.run_backtest(
        strategy=strategy,
        ohlcv_data=data["1d"],
        initial_capital=Decimal("100000000"),
        symbol=f"{symbol.base}{symbol.quote}"
    )
    
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
            strategy_name="GoldenCrossStrategy",
            output_dir="bt_results",
            show_plots=False
        )
        print("\nPlots saved to bt_results/")
    except Exception as e:
        print(f"\nCould not generate plots: {e}")
    
    return result


if __name__ == "__main__":
    result = main()
    
    # NOTE: For detailed portfolio datas
    print(result.portfolio.summary())
    print(result.trades.summary())
    print(result.equity_curve.summary())
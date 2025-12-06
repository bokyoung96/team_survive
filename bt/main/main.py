import sys
from datetime import datetime, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from backtest.engine import BacktestEngine
from backtest.plot import generate_plots
from backtest.storage import TradeStorage
from backtest.strats.dolpha3 import Dolpha3Strategy
from backtest.timeframe import MultiTimeframeData
from backtest.types import TransactionCost
from core.factory import Factory
from core.loader import DataLoader
from core.models import Exchange, MarketType, Symbol, TimeFrame, TimeRange
from main.save import generate_report
from utils import KST


def main():
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)

    symbol = Symbol.from_string("BTC/USDT:USDT")
    end_date = KST.localize(datetime(2025, 12, 1))
    desired_start = KST.localize(datetime(2024, 1, 1))
    long_start = KST.localize(datetime(2020, 1, 1))

    def cached_start(tf: TimeFrame) -> datetime:
        path = base_path / "binance" / "ohlcv" / f"{symbol.base}_{symbol.quote}" / f"{tf.value}.parquet"
        if not path.exists():
            return desired_start
        try:
            df = pd.read_parquet(path, columns=["timestamp"])
            ts = pd.to_datetime(df["timestamp"], unit="ms")
            ts = ts.dt.tz_localize(KST)
            return max(desired_start, ts.min())
        except Exception:
            return desired_start

    start_m = cached_start(TimeFrame.M3)
    start_15 = cached_start(TimeFrame.M15)
    start_30 = cached_start(TimeFrame.M30)
    start_1h = max(long_start, cached_start(TimeFrame.H1))
    start_4h = max(long_start, cached_start(TimeFrame.H4))

    data = (MultiTimeframeData(loader)
            .add(symbol, TimeFrame.M3, TimeRange(start_m, end_date))
            .add(symbol, TimeFrame.M15, TimeRange(start_15, end_date))
            .add(symbol, TimeFrame.M30, TimeRange(start_30, end_date))
            .add(symbol, TimeFrame.H1, TimeRange(start_1h, end_date))
            .add(symbol, TimeFrame.H4, TimeRange(start_4h, end_date)))

    strategy = Dolpha3Strategy(data=data, use_aggressive_exits=False)
    strategy_name = "TrendPullback"

    transaction_cost = TransactionCost(
        maker_fee=Decimal("0.0000"),
        taker_fee=Decimal("0.0000"),
        slippage=Decimal("0.0000")
    )

    engine = BacktestEngine(transaction_cost=transaction_cost)

    storage = TradeStorage(base_dir="bt_results")
    session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage.initialize_session(session_id)

    ohlcv_data = data["15m"]

    result = engine.run_backtest(
        strategy=strategy,
        ohlcv_data=ohlcv_data,
        initial_capital=Decimal("100000000"),
        symbol=f"{symbol.base}{symbol.quote}",
        storage=storage
    )

    report_start = min(start_m, start_15, start_30, start_1h, start_4h)
    generate_report(result, session_id,
                    strategy_name=strategy_name,
                    symbol_str=f"{symbol.base}/{symbol.quote}",
                    start=report_start.strftime('%Y-%m-%d'),
                    end=end_date.strftime('%Y-%m-%d'))

    generate_plots(
        result=result,
        benchmark_data=ohlcv_data,
        strategy_name=strategy_name,
        output_dir="bt_results",
        show_plots=False,
        session_id=session_id
    )

    storage.close()
    return result


if __name__ == "__main__":
    result = main()

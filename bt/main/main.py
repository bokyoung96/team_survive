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
from backtest.strats.dolpha3 import Dolpha3Strategy
from backtest.types import TransactionCost
from backtest.engine import BacktestEngine
from backtest.plot import generate_plots
from backtest.storage import TradeStorage
from main.save import generate_report


def main(strategy_choice=2):
    choice = str(strategy_choice)
    
    exchange = Exchange(id="binance", default_type=MarketType.SWAP)
    base_path = Path(__file__).parent.parent.parent / "fetch"
    factory = Factory(exchange, base_path)
    loader = DataLoader(factory)
    
    symbol = Symbol.from_string("BTC/USDT:USDT")
    start_date = KST.localize(datetime(2020, 1, 1))
    end_date = KST.localize(datetime(2025, 8, 18))
    date_range = TimeRange(start_date, end_date)
    
    if choice == "1":
        data = (MultiTimeframeData(loader)
                .add(symbol, TimeFrame.D1, date_range)
                .add(symbol, TimeFrame.M3, date_range)
                .add(symbol, TimeFrame.M30, date_range)
                .add(symbol, TimeFrame.H1, date_range))
        strategy = GoldenCrossStrategy(data=data)
        strategy_name = "GoldenCrossStrategy"
    elif choice == "2":
        data = MultiTimeframeData(loader).add(symbol, TimeFrame.D1, date_range)
        strategy = GoldenCrossOnlyStrategy()
        strategy_name = "GoldenCrossOnlyStrategy"
    
    engine = BacktestEngine(
        transaction_cost=TransactionCost(
            maker_fee=Decimal("0.0001"),
            taker_fee=Decimal("0.0001"),
            slippage=Decimal("0.0001")
        )
    )
    
    storage = TradeStorage(base_dir="bt_results")
    session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage.initialize_session(session_id)
    
    if choice == "1":
        ohlcv_data = data["3m"]
    elif choice == "2":
        ohlcv_data = data["1d"]
    
    result = engine.run_backtest(
        strategy=strategy,
        ohlcv_data=ohlcv_data,
        initial_capital=Decimal("100000000"),
        symbol=f"{symbol.base}{symbol.quote}",
        storage=storage
    )
    
    generate_report(result, session_id,
                    strategy_name=strategy_name,
                    symbol_str=f"{symbol.base}/{symbol.quote}",
                    start=date_range.start.strftime('%Y-%m-%d'),
                    end=date_range.end.strftime('%Y-%m-%d'))
    
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
    strategy_choice = 1
    result = main(strategy_choice)

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
from backtest.strats.arbitrage1 import PairsTradingStrategy
from backtest.types import TransactionCost
from backtest.engine import BacktestEngine
from backtest.plot import generate_plots
from backtest.storage import TradeStorage
from main.save import generate_report


def main(strategy_choice=1):
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
    elif choice == "3":
        # NOTE: PoW: BTC, LTC, BCH, DOGE, ETC (Mining-based consensus)
        # NOTE: PoS: ETH, BNB, ADA, SOL, DOT (Staking-based consensus)
        crypto_universe = [
            "BTC/USDT",  # Bitcoin (PoW) - Market leader
            "ETH/USDT",  # Ethereum (PoS) - Smart contracts leader
            "BNB/USDT",  # Binance Coin (PoS) - Exchange token
            "ADA/USDT",  # Cardano (PoS) - Academic blockchain
            "SOL/USDT",  # Solana (PoS) - High-performance blockchain
            "DOT/USDT",  # Polkadot (PoS) - Interoperability
            "LTC/USDT",  # Litecoin (PoW) - Bitcoin fork
            "BCH/USDT",  # Bitcoin Cash (PoW) - Scalability fork
            "DOGE/USDT", # Dogecoin (PoW) - Meme/community coin
            "ETC/USDT"   # Ethereum Classic (PoW) - Original Ethereum
        ]
        
        print(f"Loading data for {len(crypto_universe)} cryptocurrencies...")
        data = MultiTimeframeData(loader)
        
        for crypto_symbol in crypto_universe:
            try:
                crypto_sym = Symbol.from_string(crypto_symbol)
                data.add(crypto_sym, TimeFrame.D1, date_range)
                print(f"✓ Loaded {crypto_symbol}")
            except Exception as e:
                print(f"✗ Failed to load {crypto_symbol}: {e}")
        
        strategy = PairsTradingStrategy(
            data=data,
            crypto_universe=crypto_universe,
            max_pairs=5,
            rebalance_frequency=30,
            optimize_lookback=True,
            use_volatility_filter=True,
            use_trailing_stop=True
        )
        strategy_name = "MultiPairTradingStrategy"
    else:
        data = MultiTimeframeData(loader).add(symbol, TimeFrame.D1, date_range)
        strategy = GoldenCrossOnlyStrategy()
        strategy_name = "GoldenCrossOnlyStrategy"
    

    transaction_cost = TransactionCost(
        maker_fee=Decimal("0.0000"),
        taker_fee=Decimal("0.0000"),
        slippage=Decimal("0.0000")
    )
    
    engine = BacktestEngine(transaction_cost=transaction_cost)
    
    storage = TradeStorage(base_dir="bt_results")
    session_id = f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    storage.initialize_session(session_id)
    
    if choice == "1":
        ohlcv_data = data["3m"]
    elif choice == "2" or choice == "3":
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
    strategy_choice = 3
    result = main(strategy_choice)

import sys
from pathlib import Path
from decimal import Decimal

# NOTE: For interactive window mode / debugging purpose
sys.path.insert(0, str(Path(__file__).parent.parent))


from core.factory import Factory
from core.loader import DataLoader
from core.models import Symbol, TimeFrame, TimeRange, Exchange
from backtest.timeframe import MultiTimeframeData
from backtest.strategies import GoldenCrossStrategy


def main():
    exchange = Exchange(id="binance")
    base_path = Path(__file__).parent.parent.parent / "fetch"
    
    factory = Factory(exchange, base_path)
    
    loader = DataLoader(factory)
    symbol = Symbol.from_string("BTC/USDT:USDT")
    
    multi_tf_data = (MultiTimeframeData(loader)
                     .add(symbol, TimeFrame.D1, TimeRange.days(365))
                     .add(symbol, TimeFrame.M3, TimeRange.days(365))
                     .add(symbol, TimeFrame.M30, TimeRange.days(365))
                     .add(symbol, TimeFrame.H1, TimeRange.days(365)))
    
    strategy = GoldenCrossStrategy(multi_tf_data)
    daily_data = multi_tf_data["1d"]
    data_with_indicators = strategy.calculate_indicators(daily_data)
    current_price = float(daily_data['close'].iloc[-1])
    
    golden_cross = strategy.check_golden_cross(data_with_indicators)
    multi_tf_touch = strategy.check_multi_timeframe_touch(current_price)
    
    print(f"Price: ${current_price:,.2f}")
    print(f"Golden Cross: {'✅' if golden_cross else '❌'}")
    print(f"Multi-TF Touch: {'✅' if multi_tf_touch else '❌'}")
    
    class MockPortfolio:
        total_value = Decimal("10000")
    
    signal = strategy.generate_signal(data_with_indicators, None, MockPortfolio())
    print(f"Signal: {signal.type.value if signal else 'None'}")


if __name__ == "__main__":
    main()
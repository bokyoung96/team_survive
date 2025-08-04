from datetime import datetime

from analyzer import Analyzer
from models import TimeFrame, TimeRange, Exchange, MarketType
from tools import parse_symbol
from ti import (
    IchimokuCloud,
    MovingAverage,
    RSI,
)


def main(id, default_type, symbol, timeframe, time_range, indicators, *args, **kwargs):
    analyzer = Analyzer(
        exchange=Exchange(id=id, default_type=default_type),
        symbol=parse_symbol(symbol),
        timeframe=timeframe,
        time_range=time_range
    )

    analyzer.add_indicators(indicators)

    res = analyzer.run(force_download=False)
    return res


if __name__ == "__main__":
    id = "binance"
    default_type = MarketType.SWAP
    symbol = "BTC/USDT:USDT"
    timeframe = TimeFrame.M5
    time_range = TimeRange(start=datetime(2025, 1, 1),
                           end=datetime(2025, 8, 3))
    indicators = [
        IchimokuCloud(name='ichimoku'),
        MovingAverage(name='ma_50', length=50),
        RSI(name='rsi_14', length=14)
    ]
    res = main(id, default_type, symbol, timeframe, time_range, indicators)

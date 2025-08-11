from __future__ import annotations
from datetime import datetime

from analysis import Analyzer
from core.models import TimeFrame, TimeRange
from indicators import IchimokuCloud, MovingAverage, RSI
from utils.tools import parse_symbol


def main() -> dict[str, object]:
    analyzer = Analyzer(
        symbol=parse_symbol("BTC/USDT:USDT"),
        timeframe=TimeFrame.M5,
        time_range=TimeRange(start=datetime(2025, 1, 1),
                             end=datetime(2025, 8, 3)),
    )

    analyzer.add_indicators([
        IchimokuCloud(name="ichimoku"),
        MovingAverage(name="ma_50", length=50),
        RSI(name="rsi_14", length=14),
    ])

    res = analyzer.run(force_download=False)
    return res


if __name__ == "__main__":
    main()

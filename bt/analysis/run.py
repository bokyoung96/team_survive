from __future__ import annotations
import sys
from pathlib import Path
from datetime import datetime

# NOTE: For interactive window mode / debugging purpose
sys.path.insert(0, str(Path(__file__).parent.parent))

from analysis import Analyzer
from core.models import Symbol, TimeFrame, TimeRange
from indicators import IchimokuCloud, MovingAverage, RSI


def run() -> dict[str, object]:
    analyzer = Analyzer(
        symbol=Symbol.from_string("BTC/USDT:USDT"),
        timeframe=TimeFrame.D1,
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
    res = run()
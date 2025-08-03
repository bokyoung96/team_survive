from __future__ import annotations
import ccxt
from enum import Enum, unique
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Optional, Dict, Any


from tools import KST


@unique
class DataType(Enum):
    OHLCV = "ohlcv"


@unique
class TimeFrame(Enum):
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1w"


@dataclass(frozen=True)
class TimeRange:
    start: datetime
    end: datetime

    def __post_init__(self):
        if self.start.tzinfo is None:
            object.__setattr__(self, "start", KST.localize(self.start))
        if self.end.tzinfo is None:
            object.__setattr__(self, "end", KST.localize(self.end))

        if self.start >= self.end:
            raise ValueError("Start time must be before end time")

    @classmethod
    def days(cls, n: int, end: Optional[datetime] = None) -> TimeRange:
        end = end or datetime.now(KST)
        return cls(end - timedelta(days=n), end)

    @classmethod
    def hours(cls, n: int, end: Optional[datetime] = None) -> TimeRange:
        end = end or datetime.now(KST)
        return cls(end - timedelta(hours=n), end)


@unique
class MarketType(Enum):
    SPOT = "spot"
    SWAP = "swap"
    FUTURE = "future"


@dataclass(frozen=True)
class Symbol:
    base: str
    quote: str
    settle: Optional[str] = None

    def to_string(self, market_type: MarketType) -> str:
        pair = f"{self.base}/{self.quote}"
        if market_type in (MarketType.SWAP, MarketType.FUTURE) and self.settle:
            return f"{pair}:{self.settle}"
        return pair

    def __str__(self) -> str:
        return f"{self.base}/{self.quote}"


@dataclass(frozen=True)
class Exchange:
    id: str
    name: Optional[str] = None
    default_type: MarketType = MarketType.SWAP
    enable_rate_limit: bool = True
    options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            object.__setattr__(self, "name", self.id.capitalize())

    @property
    def config(self) -> Dict[str, Any]:
        config = {
            "enableRateLimit": self.enable_rate_limit,
            "defaultType": self.default_type.value,
        }
        config.update(self.options)
        return config

    def client(self) -> ccxt.Exchange:
        exch_class = getattr(ccxt, self.id)
        return exch_class(self.config)

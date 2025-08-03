from __future__ import annotations
import pytz
from datetime import datetime
from typing import Optional

KST = pytz.timezone("Asia/Seoul")


def convert_ts_to_dt(
    ts: int, tz: Optional[pytz.timezone] = KST
) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts / 1000, tz=pytz.utc).astimezone(tz)


def parse_symbol(symbol_str: str):
    from models import Symbol
    parts = symbol_str.split("/")
    if len(parts) != 2:
        raise ValueError(f"Invalid symbol format: {symbol_str}")
    base = parts[0]
    if ":" in parts[1]:
        quote, settle = parts[1].split(":", 1)
    else:
        quote, settle = parts[1], None
    return Symbol(base=base, quote=quote, settle=settle)

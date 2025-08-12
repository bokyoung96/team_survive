from __future__ import annotations
import pytz
from datetime import datetime
from typing import Optional

KST = pytz.timezone("Asia/Seoul")


def convert_ts_to_dt(ts: int, tz: Optional[pytz.timezone] = KST) -> Optional[datetime]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts / 1000, tz=pytz.utc).astimezone(tz)

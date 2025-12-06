from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.config import TradingConfig
from execution.engine import TradingEngine
from execution.events import EventBus

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class TradingWorkflow:
    engine_factory: Callable[[], TradingEngine]
    config: TradingConfig
    bus: EventBus | None = None

    def run(self, cycles: int | None = None) -> None:
        engine = self.engine_factory()
        if self.bus is not None:
            engine.bus = self.bus

        iterations = 0
        while cycles is None or iterations < cycles:
            iterations += 1
            try:
                signal = engine.run_cycle()
                if signal:
                    logger.info(
                        "Cycle %s | signal=%s side=%s confidence=%s",
                        iterations,
                        type(signal).__name__,
                        getattr(signal, "side", "n/a"),
                        getattr(signal, "confidence", "n/a"),
                    )
                else:
                    logger.debug("No trade signal produced.")
            except Exception as exc:
                logger.exception("Cycle %s | failure=%s", iterations, exc)

            time.sleep(self.config.poll_interval.total_seconds())

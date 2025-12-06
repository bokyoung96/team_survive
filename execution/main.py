from __future__ import annotations

import importlib
import inspect
import logging
import os
import sys
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from Strategies.cpo1.execution.risk import MaxLeverageGuard

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.config import TradingConfig, load_api_credentials
from execution.engine import TradingEngine
from execution.events import EventBus, OrderLogger
from execution.exchange import BinanceGateway, ExchangeGateway
from execution.params import (StrategyConfig, StrategyDefinition,
                              StrategyParamsError, TradingParams,
                              load_strategy_config)
from execution.risk import (CompositeRiskManager, MaxLeverageGuard,
                            MaxPositionGuard)
from execution.sizing import DynamicFractionSizer
from execution.strategy import TradingStrategy
from execution.workflow import TradingWorkflow

DEFAULT_STRATEGY = "test"


def make_strategy(
    spec: StrategyDefinition,
    exchange: ExchangeGateway,
    trade_cfg: TradingConfig,
    trade: TradingParams,
) -> TradingStrategy:
    try:
        module = importlib.import_module(spec.module)
        cls = getattr(module, spec.class_name)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise StrategyParamsError(
            f"Strategy '{spec.class_name}' not found in {spec.module}"
        ) from exc

    kwargs = dict[str, Any](spec.params)
    params = inspect.signature(cls).parameters
    if "exchange" in params:
        kwargs.setdefault("exchange", exchange)
    if "config" in params:
        kwargs.setdefault("config", trade_cfg)
    if "symbol" in params:
        kwargs.setdefault("symbol", trade.symbol)

    for name, param in params.items():
        if name not in kwargs:
            continue
        hint = param.annotation
        if not isinstance(hint, type) or not issubclass(hint, Enum):
            continue
        value = kwargs[name]
        if isinstance(value, hint):
            continue
        try:
            kwargs[name] = hint(value)
        except ValueError as exc:
            raise StrategyParamsError(
                f"Invalid enum value '{value}' for '{name}'"
            ) from exc

    try:
        return cls(**kwargs)
    except TypeError as exc:
        raise StrategyParamsError(
            f"Could not create strategy '{spec.class_name}': {exc}"
        ) from exc


def make_engine(cfg: StrategyConfig) -> TradingEngine:
    creds = load_api_credentials(
        Path(__file__).parent / "config.json", provider="binance")
    trade = cfg.trading

    trade_cfg = TradingConfig(
        symbol=trade.symbol,
        leverage=trade.leverage,
        poll_interval=timedelta(seconds=trade.poll_interval_seconds),
        testnet=creds.testnet,
    )

    exchange = BinanceGateway(config=trade_cfg, credentials=creds)
    strategy = make_strategy(cfg.strategy, exchange, trade_cfg, trade)

    sizer = DynamicFractionSizer(
        config=trade_cfg,
        exchange=exchange,
        fraction=trade.order_fraction,
        quote_currency=trade.quote_currency,
        include_leverage=True,
        balance_bucket=trade.balance_bucket,
    )

    guards = [MaxLeverageGuard(max_leverage=trade_cfg.leverage)]
    if trade.max_position is not None:
        guards.append(MaxPositionGuard(max_position=trade.max_position))
    risk = CompositeRiskManager(checks=tuple[MaxLeverageGuard, ...](guards))

    return TradingEngine(
        config=trade_cfg,
        exchange=exchange,
        strategy=strategy,
        sizer=sizer,
        risk=risk,
    )


def print_snapshot(engine: TradingEngine, trade: TradingParams) -> None:
    log = logging.getLogger(__name__)
    cfg = engine.config
    exch = engine.exchange
    sizer = engine.sizer

    symbol = cfg.symbol
    fraction = getattr(sizer, "fraction", None)
    include_leverage = getattr(sizer, "include_leverage", None)
    bucket = getattr(sizer, "balance_bucket", trade.balance_bucket)

    min_qty = None
    fetch_min_qty = getattr(exch, "minimum_order_quantity", None)
    if callable(fetch_min_qty):
        try:
            candidate = fetch_min_qty(symbol)
            min_qty = float(candidate) if candidate else None
        except Exception:
            min_qty = None

    balance = float("nan")
    fetch_balance = getattr(exch, "account_balance", None)
    if callable(fetch_balance):
        try:
            balance = float(fetch_balance(trade.quote_currency, bucket=bucket))
        except Exception:
            pass

    price = float("nan")
    fetch_price = getattr(exch, "current_price", None)
    if callable(fetch_price):
        try:
            price = float(fetch_price(symbol))
        except Exception:
            pass

    log.info("+----------------------+------------------------------+")
    log.info("| Field                | Value                        |")
    log.info("+----------------------+------------------------------+")
    log.info("| symbol               | %-28s |", symbol)
    log.info("| price                | %-28.6f |", price)
    log.info("| leverage             | %-28s |", cfg.leverage)
    log.info("| testnet              | %-28s |", cfg.testnet)
    log.info("| quote_currency       | %-28s |", trade.quote_currency)
    log.info("| balance_bucket       | %-28s |", bucket)
    log.info("| balance              | %-28.6f |", balance)
    log.info("| order_fraction       | %-28s |", fraction)
    log.info("| min_order_qty        | %-28s |",
             f"{min_qty:.6f}" if min_qty is not None else "n/a")
    log.info("| include_leverage     | %-28s |", include_leverage)
    log.info("+----------------------+------------------------------+")


def confirm_run() -> bool:
    return input("Proceed with automated trading? [y/N]: ").strip().lower() == "y"


def main() -> None:
    name = DEFAULT_STRATEGY

    try:
        cfg = load_strategy_config(name)
    except StrategyParamsError as exc:
        print(f"Unable to load strategy '{name}': {exc}", file=sys.stderr)
        sys.exit(1)

    level = getattr(logging, cfg.trading.log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    log = logging.getLogger(__name__)
    log.info("Using strategy '%s'", cfg.name)

    engine = make_engine(cfg)

    bus = EventBus()
    bus.subscribe(OrderLogger(), None)

    print_snapshot(engine, cfg.trading)
    if not confirm_run():
        log.info("Execution aborted by user.")
        return

    flow = TradingWorkflow(engine_factory=lambda: engine,
                           config=engine.config, bus=bus)

    try:
        flow.run()
    except KeyboardInterrupt:
        log.info("Interrupted by user.")


if __name__ == "__main__":
    main()

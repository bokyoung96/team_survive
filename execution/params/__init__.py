from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

__all__ = [
    "StrategyParamsError",
    "StrategyDefinition",
    "TradingParams",
    "StrategyConfig",
    "load_strategy_config",
]


class StrategyParamsError(RuntimeError):
    """Raised when a strategy parameter file is missing or malformed."""


@dataclass(frozen=True)
class TradingParams:
    symbol: str
    leverage: int
    poll_interval_seconds: float
    quote_currency: str
    order_fraction: float
    balance_bucket: str
    log_level: str
    max_position: float | None


@dataclass(frozen=True)
class StrategyDefinition:
    module: str
    class_name: str
    params: Mapping[str, Any]


@dataclass(frozen=True)
class StrategyConfig:
    name: str
    trading: TradingParams
    strategy: StrategyDefinition


def _params_dir() -> Path:
    return Path(__file__).resolve().parent


def _load_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError as exc:
        raise StrategyParamsError(f"Parameter file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise StrategyParamsError(
            f"Failed to decode JSON from {path}: {exc}") from exc


def load_strategy_config(name: str) -> StrategyConfig:
    path = _params_dir() / f"{name}.json"
    payload = _load_json(path)

    strategy_data = payload.get("strategy")
    trading_data = payload.get("trading")

    if not isinstance(strategy_data, dict):
        raise StrategyParamsError(
            f"'strategy' section missing or invalid in {path}")
    if not isinstance(trading_data, dict):
        raise StrategyParamsError(
            f"'trading' section missing or invalid in {path}")

    class_name = strategy_data.get("class")
    if not isinstance(class_name, str) or not class_name:
        raise StrategyParamsError(
            f"'class' field missing or invalid in {path}")

    module_name = strategy_data.get("module", "execution.strategy")
    if not isinstance(module_name, str) or not module_name:
        raise StrategyParamsError(
            f"'module' field missing or invalid in {path}")

    params = strategy_data.get("params", {})
    if not isinstance(params, dict):
        raise StrategyParamsError(
            f"'params' section must be a mapping in {path}")

    symbol = trading_data.get("symbol")
    if not isinstance(symbol, str) or not symbol:
        raise StrategyParamsError(
            f"'symbol' field missing or invalid in {path}")

    leverage = trading_data.get("leverage")
    if leverage is None:
        raise StrategyParamsError(
            f"'leverage' field missing in {path}")

    poll_interval = trading_data.get("poll_interval_seconds")
    if poll_interval is None:
        raise StrategyParamsError(
            f"'poll_interval_seconds' field missing in {path}")

    try:
        leverage_int = int(leverage)
    except (TypeError, ValueError) as exc:
        raise StrategyParamsError(
            f"'leverage' must be an integer in {path}") from exc

    try:
        poll_interval_float = float(poll_interval)
    except (TypeError, ValueError) as exc:
        raise StrategyParamsError(
            f"'poll_interval_seconds' must be numeric in {path}") from exc

    quote_currency = trading_data.get("quote_currency", "USDT")
    if not isinstance(quote_currency, str) or not quote_currency:
        raise StrategyParamsError(
            f"'quote_currency' must be a non-empty string in {path}")

    order_fraction = trading_data.get("order_fraction", 0.01)
    try:
        order_fraction_float = float(order_fraction)
    except (TypeError, ValueError) as exc:
        raise StrategyParamsError(
            f"'order_fraction' must be numeric in {path}") from exc

    balance_bucket = trading_data.get("balance_bucket", "total")
    if not isinstance(balance_bucket, str) or not balance_bucket:
        raise StrategyParamsError(
            f"'balance_bucket' must be a non-empty string in {path}")

    log_level = trading_data.get("log_level", "INFO")
    if not isinstance(log_level, str) or not log_level:
        raise StrategyParamsError(
            f"'log_level' must be a non-empty string in {path}")

    max_position = trading_data.get("max_position")
    if max_position is not None:
        try:
            max_position = float(max_position)
        except (TypeError, ValueError) as exc:
            raise StrategyParamsError(
                f"'max_position' must be numeric or null in {path}") from exc

    trading_params = TradingParams(
        symbol=symbol,
        leverage=leverage_int,
        poll_interval_seconds=poll_interval_float,
        quote_currency=quote_currency,
        order_fraction=order_fraction_float,
        balance_bucket=balance_bucket,
        log_level=log_level.upper(),
        max_position=max_position,
    )

    strategy_definition = StrategyDefinition(
        module=module_name,
        class_name=class_name,
        params=params,
    )

    return StrategyConfig(
        name=name,
        trading=trading_params,
        strategy=strategy_definition,
    )

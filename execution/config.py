from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ApiCredentials:
    api_key: str
    api_secret: str
    testnet: bool
    market: str


@dataclass(frozen=True)
class TradingConfig:
    symbol: str
    leverage: int
    poll_interval: timedelta = timedelta(seconds=5)
    testnet: bool = False


def load_api_credentials(
    path: str | Path,
    *,
    provider: str | None = "binance",
) -> ApiCredentials:
    section = load_section(path, provider)

    api_key = section["api_key"]
    api_secret = section.get("api_secret") or section["secret_key"]

    if "testnet" not in section:
        raise KeyError("Configuration value 'testnet' is required")
    testnet = section["testnet"]
    if not isinstance(testnet, bool):
        raise TypeError("Configuration value 'testnet' must be a boolean")

    if "default_type" not in section:
        raise KeyError("Configuration value 'default_type' is required")
    market_value = section["default_type"]
    if not isinstance(market_value, str):
        raise TypeError("Configuration value 'default_type' must be a string")
    market = market_value.strip().lower()
    if not market:
        raise ValueError("Configuration value 'default_type' cannot be blank")

    return ApiCredentials(
        api_key=str(api_key),
        api_secret=str(api_secret),
        testnet=testnet,
        market=market,
    )


def load_section(path: str | Path, provider: str | None = None) -> Mapping[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise TypeError("Configuration payload must be a mapping object")
    if provider is None:
        section: Any = data
    else:
        try:
            section = data[provider]
        except KeyError as exc:
            raise KeyError(
                f"Provider '{provider}' not found in configuration file") from exc

    if not isinstance(section, Mapping):
        raise TypeError("Configuration section must be a mapping object")
    return section

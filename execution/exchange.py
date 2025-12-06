from __future__ import annotations

import abc
import os
import sys
from dataclasses import dataclass, field
from typing import (Any, Callable, ClassVar, Dict, Iterable, Iterator, Mapping,
                    Sequence, Tuple, TypeVar)

import ccxt
from ccxt.base.errors import BaseError as ExchangeClientError
from ccxt.base.exchange import Exchange as ExchangeClient

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.config import ApiCredentials, TradingConfig
from execution.types import (BestBidAsk, OrderRequest, OrderResult, OrderType,
                             PositionSnapshot)

T = TypeVar("T")


class ExchangeError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, payload: Any | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.payload = payload

    @classmethod
    def from_payload(cls, message: str, *, code: int | None, payload: Any | None) -> "ExchangeError":
        return cls(message, code=code, payload=payload)

    @classmethod
    def from_client(cls, error: ExchangeClientError) -> "ExchangeError":
        code = getattr(error, "code", None)
        payload = getattr(error, "last_http_response",
                          None) or getattr(error, "args", None)
        return cls(str(error), code=code, payload=payload)


class ExchangeGateway(abc.ABC):
    @abc.abstractmethod
    def current_price(self, symbol: str) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def position(self, symbol: str) -> PositionSnapshot:
        raise NotImplementedError

    @abc.abstractmethod
    def submit_order(self, request: OrderRequest) -> OrderResult:
        raise NotImplementedError

    @abc.abstractmethod
    def best_bid_ask(self, symbol: str) -> BestBidAsk:
        raise NotImplementedError

    @abc.abstractmethod
    def account_balance(self, currency: str, *, bucket: str = "total") -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        limit: int = 100,
    ) -> Sequence[Sequence[float | int]]:
        raise NotImplementedError


@dataclass(slots=True)
class ClientExchangeGateway(ExchangeGateway):
    config: TradingConfig
    credentials: ApiCredentials
    client: ExchangeClient = field(init=False)
    _symbol_cache: Dict[str, str] = field(default_factory=dict, init=False)
    _leverage_cache: Dict[str, int] = field(default_factory=dict, init=False)

    exchange_id: ClassVar[str] = ""
    market_type: ClassVar[str] = "future"
    supports_sandbox: ClassVar[bool] = True
    _TICKER_KEYS: ClassVar[Tuple[str, ...]] = ("last", "close")
    _TICKER_INFO_KEYS: ClassVar[Tuple[str, ...]] = ("lastPrice", "markPrice")

    def __post_init__(self) -> None:
        if not self.exchange_id:
            raise ValueError(
                "ClientExchangeGateway subclass must define exchange_id")

        self.client = self._build_client()
        self._call(self.client.load_markets)

    def current_price(self, symbol: str) -> float:
        venue_symbol = self._client_symbol(symbol)
        ticker = self._call(self.client.fetch_ticker, venue_symbol)

        for key in self._TICKER_KEYS:
            price = self._to_float(ticker.get(key))
            if price is not None:
                return price

        info: Mapping[str, Any] = ticker.get("info") or {}
        for key in self._TICKER_INFO_KEYS:
            price = self._to_float(info.get(key))
            if price is not None:
                return price

        raise ExchangeError.from_payload(
            "Ticker payload missing price data", code=None, payload=ticker)

    def position(self, symbol: str) -> PositionSnapshot:
        venue_symbol = self._client_symbol(symbol)
        positions = self._call(self.client.fetch_positions, [venue_symbol])
        normalized_symbol = self._exchange_symbol(venue_symbol)

        for candidate in positions:
            parsed = self._extract_position(candidate, normalized_symbol)
            if parsed is not None:
                return parsed

        return PositionSnapshot(
            symbol=normalized_symbol,
            position_amt=0.0,
            entry_price=0.0,
            leverage=self.config.leverage,
            unrealized_pnl=0.0,
        )

    def submit_order(self, request: OrderRequest) -> OrderResult:
        venue_symbol = self._client_symbol(request.symbol)
        params = self._order_params(request)
        self._ensure_leverage(venue_symbol, request.leverage)

        result = self._call(
            self.client.create_order,
            symbol=venue_symbol,
            type=self._map_order_type(request.order_type),
            side=request.side.value.lower(),
            amount=request.quantity,
            price=request.price,
            params=params,
        )

        result_symbol = self._exchange_symbol(
            result.get("symbol") or venue_symbol)
        order_id = result.get("id") or result.get("orderId") or ""
        status = result.get("status") or "UNKNOWN"
        return OrderResult(
            symbol=result_symbol,
            side=request.side,
            order_id=str(order_id),
            status=str(status),
        )

    def best_bid_ask(self, symbol: str) -> BestBidAsk:
        venue_symbol = self._client_symbol(symbol)
        order_book = self._call(
            self.client.fetch_order_book, venue_symbol, limit=5)
        bid_price, bid_qty = self._first_level(order_book.get("bids", ()))
        ask_price, ask_qty = self._first_level(order_book.get("asks", ()))
        return BestBidAsk(bid_price=bid_price, bid_qty=bid_qty, ask_price=ask_price, ask_qty=ask_qty)

    def account_balance(self, currency: str, *, bucket: str = "total") -> float:
        balances = self._call(self.client.fetch_balance)
        target_currency = currency.upper()
        bucket_key = bucket.lower()

        if isinstance(balances, Mapping):
            container = balances.get(bucket_key)
            if isinstance(container, Mapping):
                value = self._to_float(container.get(target_currency), None)
                if value is not None:
                    return value

            entry = balances.get(target_currency)
            if isinstance(entry, Mapping):
                value = self._to_float(entry.get(bucket_key), None)
                if value is not None:
                    return value
                fallback = self._to_float(entry.get("total"), None)
                if fallback is not None:
                    return fallback
                fallback = self._to_float(entry.get("free"), None)
                if fallback is not None:
                    return fallback

        return 0.0

    def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        *,
        limit: int = 100,
    ) -> Sequence[Sequence[float | int]]:
        venue_symbol = self._client_symbol(symbol)
        candles = self._call(self.client.fetch_ohlcv,
                             venue_symbol, timeframe, limit=limit)
        return candles

    def _client_options(self) -> Dict[str, Any]:
        market = getattr(self.credentials, "market", self.market_type)
        return {"defaultType": market}

    def _client_params(self) -> Dict[str, Any]:
        params: Dict[str, Any] = {
            "apiKey": self.credentials.api_key,
            "secret": self.credentials.api_secret,
            "enableRateLimit": True,
            "timeout": 10_000,
            "options": self._client_options(),
        }
        return params

    def _order_params(self, request: OrderRequest) -> Dict[str, Any]:
        params: Dict[str, Any] = {"reduceOnly": bool(request.reduce_only)}
        if request.time_in_force is not None:
            params["timeInForce"] = request.time_in_force.upper()
        if request.stop_price is not None:
            params["stopPrice"] = request.stop_price
        return params

    def _build_client(self) -> ExchangeClient:
        factory = getattr(ccxt, self.exchange_id, None)
        if factory is None:
            raise ExchangeError(
                f"Unsupported exchange id '{self.exchange_id}'")

        try:
            client = factory(self._client_params())
        except ExchangeClientError as exc:
            raise ExchangeError.from_client(exc) from exc

        if self.config.testnet and self.supports_sandbox and hasattr(client, "set_sandbox_mode"):
            client.set_sandbox_mode(True)
        return client

    def _call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        try:
            return func(*args, **kwargs)
        except ExchangeClientError as exc:
            raise ExchangeError.from_client(exc) from exc

    def _ensure_leverage(self, venue_symbol: str, leverage: int) -> None:
        if leverage <= 0 or not hasattr(self.client, "set_leverage"):
            return
        if self._leverage_cache.get(venue_symbol) == leverage:
            return
        self._call(self.client.set_leverage, leverage, venue_symbol)
        self._leverage_cache[venue_symbol] = leverage

    def _client_symbol(self, symbol: str) -> str:
        normalized = symbol.upper()
        cached = self._symbol_cache.get(normalized)
        if cached:
            return cached

        if "/" in normalized:
            venue_symbol = normalized
        else:
            if not getattr(self.client, "markets", None):
                self._call(self.client.load_markets)
            market = getattr(self.client, "markets_by_id", {}).get(normalized)
            if isinstance(market, Mapping):
                venue_symbol = market.get("symbol", normalized)
            elif isinstance(market, Sequence):
                venue_symbol = next(
                    (m.get("symbol") for m in market if isinstance(
                        m, Mapping) and m.get("symbol")),
                    normalized,
                )
            elif len(normalized) > 4:
                venue_symbol = f"{normalized[:-4]}/{normalized[-4:]}"
            else:
                venue_symbol = normalized

        self._symbol_cache[normalized] = venue_symbol
        return venue_symbol

    def minimum_order_quantity(self, symbol: str) -> float:
        venue_symbol = self._client_symbol(symbol)
        market: Mapping[str, Any] | None
        try:
            market = self.client.market(venue_symbol)
        except Exception:
            market = None

        if not isinstance(market, Mapping):
            return 0.0

        limits = market.get("limits") or {}
        amount_limits = limits.get("amount") or {}
        min_qty = amount_limits.get("min")

        if min_qty is None:
            precision = (market.get("precision") or {}).get("amount")
            if isinstance(precision, (int, float)) and precision >= 0:
                try:
                    min_qty = 10 ** (-int(precision))
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    min_qty = None

        try:
            return float(min_qty) if min_qty else 0.0
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _exchange_symbol(symbol: str) -> str:
        if not symbol:
            return ""
        if ":" in symbol:
            symbol = symbol.split(":", 1)[0]
        return symbol.replace("/", "")

    @staticmethod
    def _map_order_type(order_type: OrderType) -> str:
        mapping = {
            OrderType.MARKET: "MARKET",
            OrderType.LIMIT: "LIMIT",
            OrderType.STOP_MARKET: "STOP_MARKET",
            OrderType.TAKE_PROFIT_MARKET: "TAKE_PROFIT_MARKET",
            OrderType.STOP: "STOP",
            OrderType.TAKE_PROFIT: "TAKE_PROFIT",
        }
        return mapping.get(order_type, order_type.value.upper())

    @staticmethod
    def _first_level(levels: Iterable[Sequence[Any]]) -> Tuple[float, float]:
        iterator: Iterator[Sequence[Any]] = iter(levels)
        try:
            price, qty = next(iterator)
            return float(price), float(qty)
        except StopIteration:
            return 0.0, 0.0
        except (TypeError, ValueError):
            return 0.0, 0.0

    def _extract_position(self, position: Mapping[str, Any], expected_symbol: str) -> PositionSnapshot | None:
        info: Mapping[str, Any] = position.get("info") or {}
        symbol = info.get("symbol") or self._exchange_symbol(
            position.get("symbol", ""))
        if symbol != expected_symbol:
            return None

        entry_price = self._to_float(info.get("entryPrice"), self._to_float(
            position.get("entryPrice"), 0.0)) or 0.0
        position_amt = self._to_float(info.get("positionAmt"), self._to_float(
            position.get("contracts"), 0.0)) or 0.0
        leverage = self._to_int(info.get("leverage"), self._to_int(
            position.get("leverage"), self.config.leverage))
        unrealized = self._to_float(info.get("unRealizedProfit"), self._to_float(
            position.get("unrealizedPnl"), 0.0)) or 0.0
        return PositionSnapshot(
            symbol=expected_symbol,
            position_amt=position_amt,
            entry_price=entry_price,
            leverage=leverage,
            unrealized_pnl=unrealized,
        )

    @staticmethod
    def _to_float(value: Any, default: float | None = None) -> float | None:
        if value in (None, "", "0", "0.0"):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _to_int(value: Any, default: int) -> int:
        if value in (None, "", "0", "0.0"):
            return default
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return default


@dataclass(slots=True)
class BinanceGateway(ClientExchangeGateway):
    exchange_id: ClassVar[str] = "binanceusdm"
    market_type: ClassVar[str] = "future"

    def _client_options(self) -> Dict[str, Any]:
        options = ClientExchangeGateway._client_options(self)
        options.setdefault("adjustForTimeDifference", True)
        return options

    def _order_params(self, request: OrderRequest) -> Dict[str, Any]:
        params = ClientExchangeGateway._order_params(self, request)
        params.setdefault("positionSide", "BOTH")
        return params

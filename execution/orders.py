from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import List, Optional, Sequence, Tuple

if True:
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from execution.config import TradingConfig
from execution.strategy import TradeSignal
from execution.types import BestBidAsk, OrderRequest, OrderSide, OrderType, TargetKind, TargetSpec


@dataclass(frozen=True)
class OrderPlan:
    entry: OrderRequest
    protective: Tuple[OrderRequest, ...]


@dataclass(slots=True)
class OrderPlanner:
    config: TradingConfig

    def plan(
        self,
        signal: TradeSignal,
        quantity: float,
        reference_price: float,
        best_bid_ask: Optional[BestBidAsk],
    ) -> OrderPlan:
        entry = self._entry_order(signal, quantity, reference_price, best_bid_ask)
        protective = self._protective_orders(signal, entry, reference_price)
        return OrderPlan(entry=entry, protective=tuple[OrderRequest, ...](protective))

    def _entry_order(
        self,
        signal: TradeSignal,
        quantity: float,
        reference_price: float,
        best_bid_ask: Optional[BestBidAsk],
    ) -> OrderRequest:
        order_type = signal.order_type or OrderType.MARKET
        price = None
        time_in_force = None

        if order_type == OrderType.LIMIT:
            price = signal.limit_price
            if price is None and signal.use_bbo_counterparty and best_bid_ask is not None:
                price = best_bid_ask.ask_price if signal.side == OrderSide.BUY else best_bid_ask.bid_price
            if price is None:
                price = reference_price
            time_in_force = "GTC"

        return OrderRequest(
            symbol=self.config.symbol,
            side=signal.side,
            order_type=order_type,
            quantity=quantity,
            leverage=self.config.leverage,
            price=price,
            time_in_force=time_in_force,
            reduce_only=False,
        )

    def _protective_orders(
        self,
        signal: TradeSignal,
        base_order: OrderRequest,
        execution_price: float,
    ) -> Sequence[OrderRequest]:
        targets = signal.protective_targets
        if not targets:
            return ()

        orders: List[OrderRequest] = []
        entry_price = base_order.price or execution_price
        exit_side = OrderSide.SELL if signal.side == OrderSide.BUY else OrderSide.BUY
        quantity = max(base_order.quantity, 1e-12)
        leverage = max(base_order.leverage, 1)

        stop_price = self._resolve_target(
            targets.stop_loss,
            entry_price,
            signal.side,
            quantity=quantity,
            leverage=leverage,
            is_take_profit=False,
        )
        if stop_price is not None:
            orders.append(
                OrderRequest(
                    symbol=self.config.symbol,
                    side=exit_side,
                    order_type=OrderType.STOP_MARKET,
                    quantity=base_order.quantity,
                    leverage=base_order.leverage,
                    stop_price=stop_price,
                    reduce_only=True,
                )
            )

        take_profit_price = self._resolve_target(
            targets.take_profit,
            entry_price,
            signal.side,
            quantity=quantity,
            leverage=leverage,
            is_take_profit=True,
        )
        if take_profit_price is not None:
            orders.append(
                OrderRequest(
                    symbol=self.config.symbol,
                    side=exit_side,
                    order_type=OrderType.TAKE_PROFIT_MARKET,
                    quantity=base_order.quantity,
                    leverage=base_order.leverage,
                    stop_price=take_profit_price,
                    reduce_only=True,
                )
            )
        return orders

    @staticmethod
    def _resolve_target(
        target: Optional[TargetSpec],
        entry_price: float,
        entry_side: OrderSide,
        *,
        quantity: float,
        leverage: int,
        is_take_profit: bool,
    ) -> Optional[float]:
        if target is None:
            return None

        if target.kind == TargetKind.PRICE:
            return max(0.0, target.value)

        if target.kind == TargetKind.PERCENT:
            change = max(0.0, target.value)
            return OrderPlanner._apply_ratio(entry_price, entry_side, change, is_take_profit)

        if target.kind == TargetKind.ROI:
            roi = max(0.0, target.value)
            ratio = roi / max(leverage, 1)
            return OrderPlanner._apply_ratio(entry_price, entry_side, ratio, is_take_profit)

        if target.kind == TargetKind.PNL:
            pnl = max(0.0, target.value)
            absolute_change = pnl / quantity
            return OrderPlanner._apply_absolute(entry_price, entry_side, absolute_change, is_take_profit)

        return None

    @staticmethod
    def _apply_ratio(
        entry_price: float,
        entry_side: OrderSide,
        ratio: float,
        is_take_profit: bool,
    ) -> float:
        ratio = max(0.0, ratio)
        if entry_side == OrderSide.BUY:
            factor = 1 + ratio if is_take_profit else 1 - ratio
        else:
            factor = 1 - ratio if is_take_profit else 1 + ratio
        return max(0.0, entry_price * factor)

    @staticmethod
    def _apply_absolute(
        entry_price: float,
        entry_side: OrderSide,
        amount: float,
        is_take_profit: bool,
    ) -> float:
        amount = max(0.0, amount)
        if entry_side == OrderSide.BUY:
            price = entry_price + amount if is_take_profit else entry_price - amount
        else:
            price = entry_price - amount if is_take_profit else entry_price + amount
        return max(0.0, price)

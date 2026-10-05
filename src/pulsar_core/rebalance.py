"""Diff calculation: from a target portfolio to order drafts.

Semantics (core-engine design): compare the target portfolio against
current holdings *plus* in-flight (submitted but unfilled) quantities and
derive the buy/sell difference per symbol:

* buys are floored down to whole board lots (100 shares on A-shares) —
  a residual below one lot produces no order, it is recorded as skipped;
* sells are lot-rounded when reducing toward a non-zero target, but a
  target of zero liquidates the whole position including the odd tail
  (零股清仓: odd lots may only be sold when clearing the position);
* sells additionally respect T+1 availability — shares bought today are
  not sellable until the next trading day, so a blocked remainder simply
  waits for the next decision point (the target stays declared).

Funding checks, fee computation and final venue-side lot validation stay
at the venue/gateway by contract; this module only produces sized order
drafts for the risk chain.
"""

from __future__ import annotations

from typing import Mapping

from pydantic import Field

from pulsar_contracts import ContractModel, Side

from .account import PositionView
from .signals import TargetPortfolio

__all__ = ["OrderDraft", "SkippedLeg", "RebalanceResult", "compute_rebalance"]

#: Default A-share board lot.
DEFAULT_LOT_SIZE = 100


class OrderDraft(ContractModel):
    """A sized, pre-risk order candidate produced by the diff calculation."""

    side: Side
    symbol: str = Field(min_length=1)
    quantity: int = Field(gt=0)


class SkippedLeg(ContractModel):
    """A target leg that produced no order, with the reason recorded."""

    symbol: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class RebalanceResult(ContractModel):
    """Drafts (sorted by symbol) plus the skipped legs of one decision point."""

    drafts: tuple[OrderDraft, ...] = ()
    skipped: tuple[SkippedLeg, ...] = ()


def _floor_to_lot(quantity: int, lot_size: int) -> int:
    return (quantity // lot_size) * lot_size


def compute_rebalance(
    target: TargetPortfolio,
    *,
    positions: Mapping[str, PositionView],
    equity: float,
    prices: Mapping[str, float],
    pending: Mapping[str, int],
    lot_size: int = DEFAULT_LOT_SIZE,
) -> RebalanceResult:
    """Turn ``target`` into order drafts against holdings and in-flight qty.

    Parameters:

    * ``positions`` — current holdings keyed by symbol;
    * ``equity`` — marked equity used to size weight legs;
    * ``prices`` — last seen price per symbol (required for every leg,
      both for sizing and because intents price off the last close);
    * ``pending`` — net signed in-flight quantity per symbol (positive =
      pending buys, negative = pending sells).

    The result iterates symbols in sorted order, so identical inputs give
    identical drafts — always.
    """
    if lot_size <= 0:
        raise ValueError(f"lot_size must be positive, got {lot_size}")

    drafts: list[OrderDraft] = []
    skipped: list[SkippedLeg] = []

    for symbol in target.symbols:
        leg = target.targets[symbol]

        # -- resolve the absolute target share count -----------------------
        price = prices.get(symbol)
        if price is None or price <= 0:
            skipped.append(
                SkippedLeg(symbol=symbol, reason="no last price to size the target leg")
            )
            continue
        if leg.weight is not None:
            value = equity * leg.weight
            target_shares = int(value // price) if value > 0 else 0
        else:
            assert leg.shares is not None  # the leg validator guarantees one mode
            target_shares = leg.shares

        held = positions.get(symbol)
        current = held.quantity if held is not None else 0
        net_pending = pending.get(symbol, 0)
        effective = current + net_pending
        delta = target_shares - effective

        if delta > 0:
            quantity = _floor_to_lot(delta, lot_size)
            if quantity <= 0:
                skipped.append(
                    SkippedLeg(
                        symbol=symbol,
                        reason=f"buy difference {delta} below one lot ({lot_size})",
                    )
                )
                continue
            drafts.append(OrderDraft(side=Side.BUY, symbol=symbol, quantity=quantity))
        elif delta < 0:
            needed = -delta
            # T+1: only available shares may be sold, and pending sells have
            # already reserved part of that availability.
            available = held.available_quantity if held is not None else 0
            available -= max(0, -net_pending)
            available = max(0, available)
            if target_shares == 0:
                # liquidation may sell the odd tail (零股清仓)
                quantity = min(needed, available)
            else:
                quantity = min(_floor_to_lot(needed, lot_size), available)
            if quantity <= 0:
                skipped.append(
                    SkippedLeg(
                        symbol=symbol,
                        reason="no sellable quantity available (T+1 or in-flight)",
                    )
                )
                continue
            drafts.append(OrderDraft(side=Side.SELL, symbol=symbol, quantity=quantity))

    return RebalanceResult(drafts=tuple(drafts), skipped=tuple(skipped))

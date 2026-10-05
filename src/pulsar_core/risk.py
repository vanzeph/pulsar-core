"""RiskGate: the pre-trade rule chain at the intent exit.

Design policy (core-engine design): risk checks run as a chain over every
order draft between the diff calculation and intent emission. Any rule may
reject a draft; a rejected draft is dropped with its reason recorded —
never partially executed, never retried within the same decision point.

Structural guarantee: strategies have no path to the execution port, so the
chain is the only way an order can exist. The runtime enforces this by
construction (see ``runtime.py``); the full five-rule set from the design
(单票仓位上限、组合敞口上限、单日亏损停机、标的黑名单、流动性下限)
arrives with the risk task — this module ships the chain plus the
single-position cap as the first built-in rule.
"""

from __future__ import annotations

from typing import Iterable, Sequence

from pydantic import Field

from pulsar_contracts import ContractModel, Side, Timestamp

from .account import PositionView
from .rebalance import OrderDraft

__all__ = [
    "RiskView",
    "RejectionRecord",
    "GateOutcome",
    "RiskRule",
    "RiskGate",
    "SinglePositionCapRule",
]


class RiskView(ContractModel):
    """The account/market snapshot rules evaluate against.

    * ``equity`` / ``cash`` — the marked account state;
    * ``positions`` — holdings keyed by symbol;
    * ``last_prices`` — last seen price per symbol;
    * ``pending`` — net signed in-flight quantity per symbol;
    * ``ts`` — the decision time of the point under review.
    """

    ts: Timestamp
    equity: float
    cash: float
    positions: dict[str, PositionView] = Field(default_factory=dict)
    last_prices: dict[str, float] = Field(default_factory=dict)
    pending: dict[str, int] = Field(default_factory=dict)

    def effective_quantity(self, symbol: str) -> int:
        """Held quantity adjusted by net in-flight orders for ``symbol``."""
        held = self.positions.get(symbol)
        current = held.quantity if held is not None else 0
        return current + self.pending.get(symbol, 0)


class RejectionRecord(ContractModel):
    """One draft dropped by one rule, with the reason — always recorded."""

    ts: Timestamp
    rule: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: Side
    quantity: int = Field(gt=0)


class GateOutcome(ContractModel):
    """The chain verdict for one decision point."""

    approved: tuple[OrderDraft, ...] = ()
    rejections: tuple[RejectionRecord, ...] = ()


class RiskRule:
    """Base class / protocol anchor for one pre-trade rule.

    Implement :meth:`check` and return ``None`` to pass or a human-readable
    rejection reason. Rules must be deterministic and side-effect free; the
    chain records the verdict.
    """

    #: Stable identifier used in rejection records and manifests.
    name: str = "risk_rule"

    def check(self, draft: OrderDraft, view: RiskView) -> str | None:
        raise NotImplementedError


class RiskGate:
    """An ordered chain of rules every draft must pass, one by one.

    Evaluation is short-circuiting per draft: the first rejecting rule wins
    and the draft is dropped; later rules never see it. Approved drafts
    keep their input order.
    """

    def __init__(self, rules: Iterable[RiskRule] = ()) -> None:
        self._rules: tuple[RiskRule, ...] = tuple(rules)
        names = [rule.name for rule in self._rules]
        if len(set(names)) != len(names):
            raise ValueError(f"risk rule names must be unique, got {names}")

    @property
    def rules(self) -> tuple[RiskRule, ...]:
        return self._rules

    @property
    def rule_names(self) -> tuple[str, ...]:
        """Chain names in order (for manifests and logs)."""
        return tuple(rule.name for rule in self._rules)

    def review(self, drafts: Sequence[OrderDraft], view: RiskView) -> GateOutcome:
        """Run every draft through the chain and return the verdict."""
        approved: list[OrderDraft] = []
        rejections: list[RejectionRecord] = []
        for draft in drafts:
            rejection: RejectionRecord | None = None
            for rule in self._rules:
                reason = rule.check(draft, view)
                if reason is not None:
                    rejection = RejectionRecord(
                        ts=view.ts,
                        rule=rule.name,
                        reason=reason,
                        symbol=draft.symbol,
                        side=draft.side,
                        quantity=draft.quantity,
                    )
                    break
            if rejection is None:
                approved.append(draft)
            else:
                rejections.append(rejection)
        return GateOutcome(approved=tuple(approved), rejections=tuple(rejections))


class SinglePositionCapRule(RiskRule):
    """单票仓位上限: cap on one symbol's market value as a share of equity.

    Sells always pass (they reduce exposure). A buy is rejected when the
    post-trade market value of the symbol — holdings plus net in-flight
    plus the drafted quantity, at the last price — would exceed
    ``max_weight × equity``.
    """

    name = "single_position_cap"

    def __init__(self, max_weight: float = 1.0) -> None:
        if not 0 < max_weight <= 1.0:
            raise ValueError(f"max_weight must be in (0, 1], got {max_weight}")
        self.max_weight = float(max_weight)

    def check(self, draft: OrderDraft, view: RiskView) -> str | None:
        if draft.side is Side.SELL:
            return None
        price = view.last_prices.get(draft.symbol)
        if price is None or price <= 0:
            return f"no last price for {draft.symbol} to evaluate exposure"
        if view.equity <= 0:
            return f"non-positive equity ({view.equity}) cannot back new exposure"
        projected = view.effective_quantity(draft.symbol) + draft.quantity
        weight = (projected * price) / view.equity
        if weight > self.max_weight + 1e-9:
            return (
                f"post-trade weight {weight:.4f} of {draft.symbol} exceeds "
                f"cap {self.max_weight:.4f}"
            )
        return None

"""RiskGate: the pre-trade rule chain at the intent exit.

Design policy (core-engine design): risk checks run as a chain over every
order draft between the diff calculation and intent emission. Any rule may
reject a draft; a rejected draft is dropped with its reason recorded —
never partially executed, never retried within the same decision point.

The five rule families of the design's 风控规则域 ship here as independent,
configurable rules:

* 单票仓位上限 — :class:`SinglePositionCapRule`;
* 组合敞口上限（默认不杠杆） — :class:`PortfolioExposureCapRule`;
* 单日亏损停机 — :class:`DailyLossHaltRule`, the one stateful breaker;
* 标的黑名单（ST、退市整理期、自定义） — :class:`SymbolBlacklistRule`;
* 流动性下限（成交额阈值） — :class:`LiquidityFloorRule`.

:class:`standard_risk_chain` assembles them in a canonical order for
assembly layers; every rejection is recorded with the rule name, the
threshold that was violated and the actual observed value.

Structural guarantee: strategies have no path to the execution port, so the
chain is the only way an order can exist. The runtime enforces this by
construction (see ``runtime.py``) and defaults to the standard chain.
"""

from __future__ import annotations

from datetime import date
from typing import Iterable, Mapping, Sequence

from pydantic import Field
from pulsar_contracts import ContractModel, Instrument, InstrumentStatus, Side, Timestamp

from .account import PositionView
from .rebalance import OrderDraft

__all__ = [
    "RiskView",
    "RuleRejection",
    "RejectionRecord",
    "GateOutcome",
    "RiskRule",
    "RiskGate",
    "SinglePositionCapRule",
    "PortfolioExposureCapRule",
    "DailyLossHaltRule",
    "SymbolBlacklistRule",
    "LiquidityFloorRule",
    "standard_risk_chain",
]

#: Weight comparisons tolerate float noise up to this slack.
_WEIGHT_EPSILON = 1e-9


class RiskView(ContractModel):
    """The account/market snapshot rules evaluate against.

    * ``equity`` / ``cash`` — the marked account state;
    * ``positions`` — holdings keyed by symbol;
    * ``last_prices`` — last seen price per symbol;
    * ``last_amounts`` — last observed turnover per symbol, in CNY (the
      daily bar ``amount`` during bar replay, the cumulative day ``amount``
      of the latest snapshot in realtime mode);
    * ``pending`` — net signed in-flight quantity per symbol;
    * ``ts`` — the decision time of the point under review.
    """

    ts: Timestamp
    equity: float
    cash: float
    positions: dict[str, PositionView] = Field(default_factory=dict)
    last_prices: dict[str, float] = Field(default_factory=dict)
    last_amounts: dict[str, float] = Field(default_factory=dict)
    pending: dict[str, int] = Field(default_factory=dict)

    def effective_quantity(self, symbol: str) -> int:
        """Held quantity adjusted by net in-flight orders for ``symbol``."""
        held = self.positions.get(symbol)
        current = held.quantity if held is not None else 0
        return current + self.pending.get(symbol, 0)


class RuleRejection(ContractModel):
    """A rule's structured verdict: why, against which threshold, what value.

    ``threshold`` / ``actual`` carry the rule's configured limit and the
    observed value that violated it — numbers where the rule is numeric,
    short strings otherwise, ``None`` where not applicable.
    """

    reason: str = Field(min_length=1)
    threshold: float | str | None = None
    actual: float | str | None = None


class RejectionRecord(ContractModel):
    """One draft dropped by one rule, with the reason — always recorded.

    ``threshold`` and ``actual`` mirror the rejecting rule's structured
    verdict so audits can answer "which limit, by how much" without
    parsing prose.
    """

    ts: Timestamp
    rule: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    symbol: str = Field(min_length=1)
    side: Side
    quantity: int = Field(gt=0)
    threshold: float | str | None = None
    actual: float | str | None = None


class GateOutcome(ContractModel):
    """The chain verdict for one decision point."""

    approved: tuple[OrderDraft, ...] = ()
    rejections: tuple[RejectionRecord, ...] = ()


class RiskRule:
    """Base class / protocol anchor for one pre-trade rule.

    Implement :meth:`check` and return ``None`` to pass, a plain string for
    an unstructured rejection, or a :class:`RuleRejection` to carry the
    threshold and the observed value into the record.

    Rules must be deterministic. Stateless rules (everything except the
    daily-loss breaker) are the norm; :class:`DailyLossHaltRule` is the
    sanctioned stateful rule, and its state is a pure function of the
    sequence of views it has observed.
    """

    #: Stable identifier used in rejection records and manifests.
    name: str = "risk_rule"

    def check(self, draft: OrderDraft, view: RiskView) -> str | RuleRejection | None:
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
                verdict = rule.check(draft, view)
                if verdict is None:
                    continue
                if isinstance(verdict, RuleRejection):
                    reason, threshold, actual = (
                        verdict.reason,
                        verdict.threshold,
                        verdict.actual,
                    )
                else:
                    reason, threshold, actual = verdict, None, None
                rejection = RejectionRecord(
                    ts=view.ts,
                    rule=rule.name,
                    reason=reason,
                    symbol=draft.symbol,
                    side=draft.side,
                    quantity=draft.quantity,
                    threshold=threshold,
                    actual=actual,
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

    def check(self, draft: OrderDraft, view: RiskView) -> str | RuleRejection | None:
        if draft.side is Side.SELL:
            return None
        price = view.last_prices.get(draft.symbol)
        if price is None or price <= 0:
            return RuleRejection(
                reason=f"no last price for {draft.symbol} to evaluate exposure",
                actual=draft.symbol,
            )
        if view.equity <= 0:
            return RuleRejection(
                reason=f"non-positive equity ({view.equity}) cannot back new exposure",
                actual=view.equity,
            )
        projected = view.effective_quantity(draft.symbol) + draft.quantity
        weight = (projected * price) / view.equity
        if weight > self.max_weight + _WEIGHT_EPSILON:
            return RuleRejection(
                reason=(
                    f"post-trade weight {weight:.4f} of {draft.symbol} exceeds "
                    f"cap {self.max_weight:.4f}"
                ),
                threshold=self.max_weight,
                actual=weight,
            )
        return None


class PortfolioExposureCapRule(RiskRule):
    """组合敞口上限: cap on total position market value over equity.

    The design's default is ``1.0`` — no leverage: the gross market value
    of all holdings (plus net in-flight and the drafted buy itself, at
    last prices) may not exceed equity. Sells always pass. Values above
    ``1.0`` are accepted for assemblies that deliberately run leveraged.
    """

    name = "portfolio_exposure_cap"

    def __init__(self, max_gross_weight: float = 1.0) -> None:
        if not max_gross_weight > 0:
            raise ValueError(
                f"max_gross_weight must be positive, got {max_gross_weight}"
            )
        self.max_gross_weight = float(max_gross_weight)

    def check(self, draft: OrderDraft, view: RiskView) -> str | RuleRejection | None:
        if draft.side is Side.SELL:
            return None
        if view.equity <= 0:
            return RuleRejection(
                reason=f"non-positive equity ({view.equity}) cannot back new exposure",
                actual=view.equity,
            )
        gross = 0.0
        for symbol in sorted(set(view.positions) | {draft.symbol}):
            quantity = view.effective_quantity(symbol)
            if symbol == draft.symbol:
                quantity += draft.quantity
            if quantity <= 0:
                continue
            price = view.last_prices.get(symbol)
            if price is None or price <= 0:
                return RuleRejection(
                    reason=(
                        f"cannot mark holding {symbol} at a last price; "
                        "gross exposure is unverifiable"
                    ),
                    actual=symbol,
                )
            gross += quantity * price
        weight = gross / view.equity
        if weight > self.max_gross_weight + _WEIGHT_EPSILON:
            return RuleRejection(
                reason=(
                    f"post-trade gross exposure {weight:.4f} exceeds "
                    f"cap {self.max_gross_weight:.4f}"
                ),
                threshold=self.max_gross_weight,
                actual=weight,
            )
        return None


class DailyLossHaltRule(RiskRule):
    """单日亏损停机: intraday equity drawdown circuit breaker.

    The rule latches the marked equity of the first decision point it sees
    each trading day as the day's starting equity. When equity has drawn
    down from that reference by at least ``max_daily_loss`` (touching the
    threshold counts), the breaker trips: for the rest of the day *every*
    new intent — buys and sells alike — is rejected, and the rejection
    records carry the alert. The next trading day re-arms the breaker
    against that day's own starting equity.

    This is the one stateful rule in the chain. Its state is a pure
    function of the sequence of views reviewed, so a run remains fully
    deterministic: same views in, same halts out. The breaker only ever
    observes views carrying drafts (``check`` is the observation point),
    which is exactly the semantics of "停止接受新意图" — it trips at the
    first intent attempted after the breach.
    """

    name = "daily_loss_halt"

    def __init__(self, max_daily_loss: float = 0.05) -> None:
        if not 0 < max_daily_loss <= 1.0:
            raise ValueError(
                f"max_daily_loss must be in (0, 1], got {max_daily_loss}"
            )
        self.max_daily_loss = float(max_daily_loss)
        self._day: date | None = None
        self._day_start_equity: float | None = None
        self._halted = False
        self._trip_drawdown: float | None = None

    # -- explicit state (observability for assembly and tests) ----------------

    @property
    def halted(self) -> bool:
        """Whether the breaker is currently refusing all new intents."""
        return self._halted

    @property
    def current_day(self) -> date | None:
        """The trading day the breaker is currently tracking."""
        return self._day

    @property
    def day_start_equity(self) -> float | None:
        """The latched equity reference of the current trading day."""
        return self._day_start_equity

    @property
    def trip_drawdown(self) -> float | None:
        """The drawdown that tripped the breaker, or ``None`` while armed."""
        return self._trip_drawdown

    # -- evaluation -------------------------------------------------------------

    def check(self, draft: OrderDraft, view: RiskView) -> str | RuleRejection | None:
        self._observe(view)
        if self._halted:
            drawdown = self._trip_drawdown
            assert drawdown is not None  # halted implies a trip record (or sentinel)
            return RuleRejection(
                reason=(
                    f"daily loss halt active: equity drew down {drawdown:.4f} "
                    f">= limit {self.max_daily_loss:.4f} on {self._day}; "
                    "no new intents until the next trading day"
                ),
                threshold=self.max_daily_loss,
                actual=drawdown,
            )
        return None

    def _observe(self, view: RiskView) -> None:
        day = view.ts.date()
        if day != self._day:
            self._day = day
            self._day_start_equity = view.equity
            self._halted = False
            self._trip_drawdown = None
        start = self._day_start_equity
        assert start is not None  # the rollover above latches it for this day
        if start <= 0:
            # A day starting at non-positive equity cannot back any intent.
            self._halted = True
            self._trip_drawdown = 1.0
            return
        drawdown = (start - view.equity) / start
        if drawdown >= self.max_daily_loss and not self._halted:
            self._halted = True
            self._trip_drawdown = drawdown


class SymbolBlacklistRule(RiskRule):
    """标的黑名单: ST, delisting arrangement, custom exclusions.

    A buy is rejected when its symbol is on the custom exclusion list, or
    when the injected instrument snapshot says the name is ST, already
    delisted, or inside its delisting arrangement period (退市整理期:
    within ``delisting_window_days`` before ``delist_date``, or the date
    has already passed while status still says listed). Sells always pass
    — a name that turns unsafe must remain exitable.

    ``instruments`` is an immutable snapshot supplied by the assembly
    layer from the data port; this package never fetches anything itself.
    Symbols without instrument knowledge are judged on the custom list
    alone (fail-open: the blacklist excludes known names, it does not
    gate the whole universe).
    """

    name = "symbol_blacklist"

    def __init__(
        self,
        *,
        symbols: Iterable[str] = (),
        instruments: Mapping[str, Instrument] | None = None,
        delisting_window_days: int = 30,
    ) -> None:
        if delisting_window_days < 0:
            raise ValueError(
                f"delisting_window_days must be >= 0, got {delisting_window_days}"
            )
        self._excluded = frozenset(symbols)
        self._instruments: dict[str, Instrument] = (
            dict(instruments) if instruments is not None else {}
        )
        self.delisting_window_days = int(delisting_window_days)

    @property
    def excluded_symbols(self) -> frozenset[str]:
        """The custom exclusion list (ST/delisting checks are separate)."""
        return self._excluded

    def _blacklist_reason(self, symbol: str, as_of: date) -> str | None:
        """Why ``symbol`` is unbuyable on ``as_of``, or ``None`` if it is not."""
        if symbol in self._excluded:
            return f"{symbol} is on the custom exclusion list"
        info = self._instruments.get(symbol)
        if info is None:
            return None
        if info.is_st:
            return f"{symbol} is flagged ST"
        if info.status is InstrumentStatus.DELISTED:
            return f"{symbol} is delisted"
        if info.delist_date is not None:
            days_to_delist = (info.delist_date - as_of).days
            if days_to_delist < 0:
                return f"{symbol} has passed its delist date {info.delist_date}"
            if days_to_delist <= self.delisting_window_days:
                return (
                    f"{symbol} is in its delisting arrangement period "
                    f"(delists {info.delist_date})"
                )
        return None

    def check(self, draft: OrderDraft, view: RiskView) -> str | RuleRejection | None:
        if draft.side is Side.SELL:
            return None
        why = self._blacklist_reason(draft.symbol, view.ts.date())
        if why is not None:
            return RuleRejection(
                reason=f"buy of {draft.symbol} refused: {why}",
                threshold="tradeable (not ST / not delisting / not excluded)",
                actual=why,
            )
        return None


class LiquidityFloorRule(RiskRule):
    """流动性下限: no new positions in names below a turnover floor.

    A buy is rejected unless the symbol's last observed turnover
    (``RiskView.last_amounts``, CNY) is at least ``min_amount``; a symbol
    with no turnover observation at all is rejected too — unverifiable
    liquidity is not liquidity. Sells always pass (exits stay possible
    exactly when liquidity dries up).
    """

    name = "liquidity_floor"

    def __init__(self, min_amount: float = 1_000_000.0) -> None:
        if not min_amount >= 0:
            raise ValueError(f"min_amount must be >= 0, got {min_amount}")
        self.min_amount = float(min_amount)

    def check(self, draft: OrderDraft, view: RiskView) -> str | RuleRejection | None:
        if draft.side is Side.SELL:
            return None
        amount = view.last_amounts.get(draft.symbol)
        if amount is None:
            return RuleRejection(
                reason=(
                    f"no turnover observation for {draft.symbol}; "
                    "liquidity floor is unverifiable"
                ),
                threshold=self.min_amount,
            )
        if amount < self.min_amount:
            return RuleRejection(
                reason=(
                    f"turnover {amount:,.0f} CNY of {draft.symbol} is below "
                    f"the floor {self.min_amount:,.0f} CNY"
                ),
                threshold=self.min_amount,
                actual=amount,
            )
        return None


def standard_risk_chain(
    *,
    single_position_cap: float = 1.0,
    max_gross_exposure: float = 1.0,
    max_daily_loss: float = 0.05,
    min_turnover: float = 1_000_000.0,
    excluded_symbols: Iterable[str] = (),
    instruments: Mapping[str, Instrument] | None = None,
    delisting_window_days: int = 30,
) -> tuple[RiskRule, ...]:
    """The design's five pre-trade rules in canonical order.

    Order is policy, not accident: the stateful daily-loss breaker runs
    first so it observes every view and refuses everything once tripped;
    the name-based checks follow; the exposure arithmetic closes the
    chain. The runtime defaults to this chain — assemblies pass their own
    :class:`RiskGate` to override parameters, order or membership.
    """

    return (
        DailyLossHaltRule(max_daily_loss=max_daily_loss),
        SymbolBlacklistRule(
            symbols=excluded_symbols,
            instruments=instruments,
            delisting_window_days=delisting_window_days,
        ),
        LiquidityFloorRule(min_amount=min_turnover),
        SinglePositionCapRule(max_weight=single_position_cap),
        PortfolioExposureCapRule(max_gross_weight=max_gross_exposure),
    )

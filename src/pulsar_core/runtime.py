"""StrategyRuntime: the intent pipeline wired onto the kernel loop.

One runtime drives one strategy over one run. Per market event it executes
the pipeline defined by the core-engine design::

    on_bar / on_tick            (strategy declares target tendencies)
        -> Signal list          (this decision point's declarations)
        -> PortfolioBuilder     (signals -> TargetPortfolio)
        -> compute_rebalance    (target vs holdings + in-flight -> drafts)
        -> RiskGate.review      (rule chain; rejected drafts are recorded)
        -> OrderIntent          (idempotency key = run id + sequence)
        -> ExecutionPort.submit (the ONLY call into the venue, ever)

Structural guarantee ("风控兜底：策略无法绕过"): strategies receive contexts
that carry no ordering surface — no execution port, no risk gate, no
engine reference, only reads and target declaration. ``port.submit`` is
called from exactly one place in this package (``_emit``, after the gate)
and from nowhere else; a static test enforces this at the source level.

The runtime also owns the decision-side account (updated from fill events
flowing back through the bus) and the per-symbol bar history the context
indicators read. Intent pricing uses a limit at the symbol's last seen
price — a deliberate Research-mode simplification; venues decide actual
fills, and live/paper assembly may override pricing policy at the venue.
"""

from __future__ import annotations

from collections import deque
from datetime import date, datetime
from typing import Any

from pulsar_contracts import (
    ContractModel,
    ExecutionEventType,
    ExecutionPort,
    IdempotencyKey,
    OrderId,
    OrderIntent,
    PriceMode,
    Side,
)

from .account import PositionView, TradingAccount
from .bus import EventBus
from .errors import PulsarCoreError
from .events import Event, EventKind, SessionPhase
from .lifecycle import LifecycleRecord
from .manifest import RunManifest
from .rebalance import DEFAULT_LOT_SIZE, OrderDraft, SkippedLeg, compute_rebalance
from .risk import RejectionRecord, RiskGate, RiskView, standard_risk_chain
from .signals import PortfolioBuilder, PassThroughBuilder, Signal
from .strategy import (
    BaseContext,
    BarContext,
    FillContext,
    StrategyBase,
    TickContext,
    _Declarations,
)

__all__ = ["StrategyRuntime", "Submission"]

#: Default number of recent bars kept per symbol for context indicators.
DEFAULT_HISTORY_DEPTH = 1000


class Submission(ContractModel):
    """One emitted intent paired with the order id its venue returned."""

    intent: OrderIntent
    order_id: OrderId


class _PendingOrder:
    """An intent awaiting its fill (tracked from submit time, not fill time)."""

    __slots__ = ("symbol", "side", "quantity", "remaining")

    def __init__(self, symbol: str, side: Side, quantity: int) -> None:
        self.symbol = symbol
        self.side = side
        self.quantity = quantity
        self.remaining = quantity


class StrategyRuntime:
    """Drives one strategy run: hooks in, intents out, fills back."""

    def __init__(
        self,
        *,
        bus: EventBus,
        port: ExecutionPort,
        strategy: StrategyBase,
        builder: PortfolioBuilder | None = None,
        gate: RiskGate | None = None,
        initial_cash: float = 1_000_000.0,
        lot_size: int = DEFAULT_LOT_SIZE,
        history_depth: int = DEFAULT_HISTORY_DEPTH,
    ) -> None:
        if lot_size <= 0:
            raise ValueError(f"lot_size must be positive, got {lot_size}")
        if history_depth <= 0:
            raise ValueError(f"history_depth must be positive, got {history_depth}")
        self._bus = bus
        self._port = port
        self._strategy = strategy
        self._builder = builder if builder is not None else PassThroughBuilder()
        # Default chain: the design's five pre-trade rules (daily-loss halt,
        # blacklist, liquidity floor, single-position cap, gross exposure cap)
        # at their standard parameters — no leverage, no halts below 5% daily
        # drawdown, 1M CNY turnover floor. Assembly overrides with its own
        # chain built from ``standard_risk_chain`` or raw rules.
        self._gate = gate if gate is not None else RiskGate(standard_risk_chain())
        self._lot_size = lot_size

        self.account = TradingAccount(cash=initial_cash)
        self._state: dict[str, Any] = {}
        self._history: dict[str, deque[Any]] = {}
        self._history_depth = history_depth
        self._last_amounts: dict[str, float] = {}
        self._pending_orders: dict[OrderId, _PendingOrder] = {}
        self._seq = 0
        self._run_id: str | None = None
        self._manifest: RunManifest | None = None
        self._current_day: date | None = None
        self._retire_record: LifecycleRecord | None = None

        #: Everything the run produced, in deterministic order.
        self.submissions: list[Submission] = []
        self.rejections: list[RejectionRecord] = []
        self.skipped: list[SkippedLeg] = []

        # wiring: the runtime listens on the loop and mirrors venue events
        # back onto it so fills dispatch in the same deterministic order.
        bus.subscribe(EventKind.SESSION, self._on_session)
        bus.subscribe(EventKind.MARKET, self._on_market)
        bus.subscribe(EventKind.EXECUTION, self._on_execution)
        port.on_event(self._on_port_event)

    # -- run binding ----------------------------------------------------------

    def bind_manifest(self, manifest: RunManifest) -> None:
        """Receive the run's manifest; intents key off its ``run_id``.

        Wire this as ``ReplaySession(..., on_manifest=runtime.bind_manifest)``
        so idempotency keys identify the run the venue is executing. The
        manifest reference is also kept for the lifecycle audit overlay:
        a mid-run :meth:`retire` stamps its record there.
        """
        self._run_id = manifest.run_id
        self._manifest = manifest

    @property
    def run_id(self) -> str:
        """The bound run id (fails loudly when the runtime was never wired)."""
        if self._run_id is None:
            raise PulsarCoreError(
                "run id not bound: pass on_manifest=runtime.bind_manifest to "
                "the session before running"
            )
        return self._run_id

    @property
    def strategy(self) -> StrategyBase:
        return self._strategy

    @property
    def gate(self) -> RiskGate:
        """The chain guarding this run's intent exit (read-only access)."""
        return self._gate

    @property
    def state(self) -> dict[str, Any]:
        """The strategy's explicit state dict (archive with the run)."""
        return self._state

    @property
    def intents(self) -> tuple[OrderIntent, ...]:
        """Every emitted intent, in emission order."""
        return tuple(submission.intent for submission in self.submissions)

    # -- lifecycle (下线) -------------------------------------------------------

    @property
    def retired(self) -> bool:
        """Whether this session received a retire and stopped deciding."""
        return self._retire_record is not None

    @property
    def retire_record(self) -> LifecycleRecord | None:
        """The audit record of the retire this session received, if any."""
        return self._retire_record

    def retire(
        self,
        *,
        reason: str,
        operator: str = "",
        ts: datetime | None = None,
        from_status: str = "active",
        to_status: str = "retired",
    ) -> LifecycleRecord:
        """Take this session offline: no new order intents, ever again.

        Core-engine design (模型配置生命周期): a running session that
        receives a 下线 immediately stops producing new order intents.
        The checkpoint sits at the pipeline entry — declarations are
        dropped before the diff calculation, so nothing reaches the risk
        gate or the venue — and ``_emit`` carries a second, defensive
        backstop. Existing positions are deliberately *not* force-sold:
        they follow the strategy's own exit rules.

        The action is archived into the bound run's manifest (lifecycle
        audit overlay: status change, reason, timestamp), so the archived
        document answers "why did this run stop trading". ``ts`` defaults
        to the kernel clock's now — deterministic under replay, real time
        under a realtime clock. Retiring twice is idempotent: the first
        record stays authoritative and is returned.
        """
        if self._retire_record is not None:
            return self._retire_record
        record = LifecycleRecord(
            action="retire",
            from_status=from_status,
            to_status=to_status,
            reason=reason,
            operator=operator,
            ts=ts if ts is not None else self._bus.now,
        )
        self._retire_record = record
        if self._manifest is not None:
            self._manifest.record_lifecycle(record)
        return record

    # -- event handlers ---------------------------------------------------------

    def _on_session(self, event: Event) -> None:
        phase = event.session_phase
        assert phase is not None
        ctx = BaseContext(
            now=event.ts, state=self._state, portfolio=self.account.snapshot()
        )
        if phase is SessionPhase.STARTED:
            self._strategy.on_start(ctx)
        elif phase is SessionPhase.FINISHED:
            self._strategy.on_stop(ctx)

    def _on_market(self, event: Event) -> None:
        bar = event.bar
        snapshot = event.snapshot
        if bar is not None:
            self._rollover_if_new_day(bar.ts.date())
            self.account.mark_price(bar.symbol, bar.close)
            self._last_amounts[bar.symbol] = bar.amount
            history = self._history.setdefault(
                bar.symbol, deque(maxlen=self._history_depth)
            )
            history.append(bar)
            collector = _Declarations()
            ctx = BarContext(
                bar=bar,
                history=tuple(history),
                now=event.ts,
                state=self._state,
                portfolio=self.account.snapshot(),
            )
            ctx.attach_collector(collector)
            self._strategy.on_bar(ctx)
        elif snapshot is not None:
            self._rollover_if_new_day(snapshot.ts.date())
            self.account.mark_price(snapshot.symbol, snapshot.last_price)
            self._last_amounts[snapshot.symbol] = snapshot.amount
            collector = _Declarations()
            tick_ctx = TickContext(
                snapshot=snapshot,
                now=event.ts,
                state=self._state,
                portfolio=self.account.snapshot(),
            )
            tick_ctx.attach_collector(collector)
            self._strategy.on_tick(tick_ctx)
        else:  # pragma: no cover - the event envelope enforces a payload
            return
        self._run_pipeline(collector.take(), now=event.ts)

    def _on_execution(self, event: Event) -> None:
        execution = event.execution
        assert execution is not None
        pending = self._pending_orders.get(execution.order_id)
        fill = execution.fill
        if fill is not None:
            self.account.apply_fill(fill)
            if pending is not None:
                pending.remaining -= fill.quantity
                if pending.remaining <= 0:
                    del self._pending_orders[execution.order_id]
            ctx = FillContext(
                fill=fill,
                now=event.ts,
                state=self._state,
                portfolio=self.account.snapshot(),
            )
            self._strategy.on_fill(ctx)
        elif execution.event_type in (
            ExecutionEventType.REJECTED,
            ExecutionEventType.ERROR,
            ExecutionEventType.CANCELLED,
        ):
            # the quantity is no longer in flight; the next decision point
            # re-derives the difference against live holdings.
            if pending is not None:
                del self._pending_orders[execution.order_id]

    def _on_port_event(self, execution: Any) -> None:
        """Venue callback: mirror the execution event onto the kernel loop."""
        self._bus.publish(Event.of_execution(execution))

    # -- pipeline ----------------------------------------------------------------

    def _run_pipeline(self, signals: list[Signal], *, now: datetime) -> None:
        if self._retire_record is not None:
            # 下线 checkpoint: a retired session never produces new order
            # intents — declarations are dropped before any diff, risk or
            # emission happens. Existing positions are left to the
            # strategy's own exit rules (no force-selling here).
            return
        if not signals:
            return
        target = self._builder.build(signals)
        positions: dict[str, PositionView] = {
            view.symbol: view for view in self.account.positions()
        }
        result = compute_rebalance(
            target,
            positions=positions,
            equity=self.account.equity,
            prices=self.account.prices,
            pending=self._net_pending(),
            lot_size=self._lot_size,
        )
        self.skipped.extend(result.skipped)
        if not result.drafts:
            return

        view = RiskView(
            ts=now,
            equity=self.account.equity,
            cash=self.account.cash,
            positions=positions,
            last_prices=self.account.prices,
            last_amounts=dict(self._last_amounts),
            pending=self._net_pending(),
        )
        outcome = self._gate.review(result.drafts, view)
        self.rejections.extend(outcome.rejections)
        for draft in outcome.approved:
            self._emit(draft)

    def _emit(self, draft: OrderDraft) -> None:
        """Turn one gate-approved draft into an intent and submit it.

        The single ``port.submit`` call site of this package: nothing else
        may talk to the venue, and this path always runs after the gate.
        """
        if self._retire_record is not None:  # defensive backstop of the
            # retire checkpoint in _run_pipeline — a retired session emits
            # nothing, and a draft reaching this point is recorded, not sent
            self.skipped.append(
                SkippedLeg(
                    symbol=draft.symbol,
                    reason=(
                        "experiment retired: session stopped producing new "
                        f"order intents ({self._retire_record.reason})"
                    ),
                )
            )
            return
        if self._run_id is None:
            raise PulsarCoreError(
                "run id not bound: pass on_manifest=runtime.bind_manifest to "
                "the session before running"
            )
        price = self.account.last_price(draft.symbol)
        if price is None or price <= 0:  # defensive: drafts are priced legs
            self.skipped.append(
                SkippedLeg(symbol=draft.symbol, reason="no last price at emission")
            )
            return
        self._seq += 1
        intent = OrderIntent(
            idempotency_key=IdempotencyKey(run_id=self._run_id, seq=self._seq),
            side=draft.side,
            symbol=draft.symbol,
            quantity=draft.quantity,
            price_mode=PriceMode.LIMIT,
            limit_price=price,
        )
        order_id = self._port.submit(intent)
        self.submissions.append(Submission(intent=intent, order_id=order_id))
        self._pending_orders[order_id] = _PendingOrder(
            symbol=draft.symbol, side=draft.side, quantity=draft.quantity
        )

    # -- internals -----------------------------------------------------------------

    def _rollover_if_new_day(self, day: date) -> None:
        if day != self._current_day:
            self.account.on_new_trading_day(day)
            self._current_day = day

    def _net_pending(self) -> dict[str, int]:
        """Net signed in-flight quantity per symbol (buys +, sells -)."""
        net: dict[str, int] = {}
        for pending in self._pending_orders.values():
            sign = 1 if pending.side is Side.BUY else -1
            net[pending.symbol] = net.get(pending.symbol, 0) + sign * pending.remaining
        return net

"""The strategy framework: lifecycle hooks and typed context objects.

A strategy is a plugin that expresses *what it wants* — never how to get
it. The contract has three legs (core-engine design):

* **Lifecycle**: ``on_start`` / ``on_bar`` / ``on_tick`` / ``on_fill`` /
  ``on_stop`` hooks, invoked by the runtime in event order.
* **Typed contexts**: strategies read market data and the account book
  only through the context handed to each hook, and declare target
  positions through ``target_weight`` / ``target_shares``. Contexts carry
  no ordering surface of any kind — there is no method that places,
  sizes or amends an order, and no reference to the execution port, the
  risk chain or the engine itself. The risk exit is therefore
  structurally impossible to bypass.
* **Stateless-first**: cross-bar state lives explicitly in ``ctx.state``
  (one dict per run, archived with the run). Anything not stored there is
  expected to be recomputed from the event stream, so the same inputs
  always yield the same outputs.

Parameters are declared on the class (``params = Params(fast=5,
slow=20)``) and bound per run from configuration, so parameter sweeps
never touch strategy code; see :mod:`pulsar_core.params`.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Any, Iterable, Mapping, Sequence

from pulsar_contracts import Bar, ContractModel, Fill, Snapshot

from .account import PortfolioView
from .errors import PulsarCoreError
from .params import BoundParams, Params
from .signals import Signal

__all__ = [
    "IndicatorValue",
    "BaseContext",
    "FillContext",
    "BarContext",
    "TickContext",
    "StrategyBase",
    "FIELD_NAMES",
]

#: Bar fields the indicator helpers accept.
FIELD_NAMES = ("open", "high", "low", "close", "volume", "amount")


class IndicatorValue(ContractModel):
    """A rolling indicator reading: current value and the previous one.

    Either may be ``None`` during warm-up (not enough history yet);
    :meth:`BarContext.cross_up` / :meth:`BarContext.cross_down` treat any
    ``None`` as "no statement" (``False``), so strategies need no explicit
    warm-up branches.
    """

    value: float | None = None
    previous: float | None = None


class _Declarations:
    """Append-only collector for the targets declared at one decision point.

    The runtime creates one collector per market event, hands it to the
    context, and takes the declarations when the hook returns. Declaring
    two different targets for the same symbol within one decision point
    is ambiguous and rejected; repeating an identical declaration is a
    no-op.
    """

    def __init__(self) -> None:
        self._signals: list[Signal] = []

    def declare_weight(self, symbol: str, weight: float) -> None:
        self._record(Signal(symbol=symbol, weight=weight))

    def declare_shares(self, symbol: str, shares: int) -> None:
        self._record(Signal(symbol=symbol, shares=shares))

    def _record(self, signal: Signal) -> None:
        for existing in self._signals:
            if existing.symbol == signal.symbol:
                if existing == signal:
                    return  # identical re-declaration is idempotent
                raise PulsarCoreError(
                    f"conflicting targets declared for {signal.symbol!r} "
                    f"in one decision point"
                )
        self._signals.append(signal)

    def take(self) -> list[Signal]:
        signals = self._signals
        self._signals = []
        return signals


class BaseContext:
    """Reads shared by every hook: time, explicit state, the account book.

    ``state`` is the run's single explicit state dict — the one place
    cross-bar memory is allowed to live. ``portfolio`` is an immutable
    snapshot; strategies cannot mutate the account through it.
    """

    def __init__(
        self,
        *,
        now: datetime,
        state: dict[str, Any],
        portfolio: PortfolioView,
    ) -> None:
        self._now = now
        self._state = state
        self._portfolio = portfolio

    @property
    def now(self) -> datetime:
        """Decision time (kernel clock — backtest and realtime alike)."""
        return self._now

    @property
    def state(self) -> dict[str, Any]:
        """The run's explicit state dict (kept JSON-native by convention)."""
        return self._state

    @property
    def portfolio(self) -> PortfolioView:
        """Immutable account snapshot: cash, equity, positions."""
        return self._portfolio

    @property
    def cash(self) -> float:
        return self._portfolio.cash

    @property
    def equity(self) -> float:
        return self._portfolio.equity

    def position_quantity(self, symbol: str) -> int:
        """Held quantity of ``symbol`` (0 when nothing is held)."""
        view = self._portfolio.position(symbol)
        return view.quantity if view is not None else 0


class FillContext(BaseContext):
    """Context of ``on_fill``: the confirmed fill plus the shared reads."""

    def __init__(self, *, fill: Fill, **base: Any) -> None:
        super().__init__(**base)
        self._fill = fill

    @property
    def fill(self) -> Fill:
        return self._fill


class _MarketContext(BaseContext):
    """Shared surface of bar/tick contexts: target declaration."""

    _declarations: _Declarations | None = None  # injected by the runtime

    def attach_collector(self, declarations: _Declarations) -> None:
        """Wire the decision point's declaration collector (runtime only)."""
        self._declarations = declarations

    def target_weight(self, symbol: str, weight: float) -> None:
        """Declare a target: allocate ``weight`` of equity to ``symbol``.

        This states a desired end state; the engine computes the
        difference, rounds it, risk-checks it and turns it into an order
        intent. It does **not** place an order.
        """
        collector = self._require_collector()
        collector.declare_weight(symbol, weight)

    def target_shares(self, symbol: str, shares: int) -> None:
        """Declare an absolute target share count for ``symbol``."""
        collector = self._require_collector()
        collector.declare_shares(symbol, shares)

    def _require_collector(self) -> _Declarations:
        if self._declarations is None:
            raise PulsarCoreError(
                "target declaration is only available while the runtime is "
                "dispatching a market event"
            )
        return self._declarations


class BarContext(_MarketContext):
    """Context of ``on_bar``: bar reads, indicator helpers, target declaration.

    Indicator helpers operate on the current symbol's bar history — the
    current bar included — capped at the runtime's history depth.
    """

    def __init__(self, *, bar: Bar, history: Sequence[Bar], **base: Any) -> None:
        super().__init__(**base)
        self._bar = bar
        self._history: deque[Bar] = deque(history)

    @property
    def bar(self) -> Bar:
        """The bar that triggered this decision point."""
        return self._bar

    @property
    def symbol(self) -> str:
        symbol: str = self._bar.symbol
        return symbol

    # -- history / indicators -------------------------------------------------

    def bars(self, n: int) -> tuple[Bar, ...]:
        """The most recent ``n`` bars of the current symbol, oldest first."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        return tuple(list(self._history)[-n:])

    def field_series(self, field: str, n: int) -> tuple[float, ...]:
        """The most recent ``n`` values of ``field`` ("close", "volume", ...)."""
        if field not in FIELD_NAMES:
            raise ValueError(
                f"unknown bar field {field!r}; expected one of {FIELD_NAMES}"
            )
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        window = self.bars(n + 1)[1:]
        return tuple(getattr(bar, field) for bar in window)

    def sma(self, field: str, n: int) -> IndicatorValue:
        """Simple moving average of ``field`` over ``n`` bars.

        Returns an :class:`IndicatorValue` whose ``value`` is the mean of
        the last ``n`` observations and whose ``previous`` is the same mean
        over the window one bar earlier; either is ``None`` during
        warm-up.
        """
        if field not in FIELD_NAMES:
            raise ValueError(
                f"unknown bar field {field!r}; expected one of {FIELD_NAMES}"
            )
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        window = list(self._history)
        if len(window) < n:
            return IndicatorValue(value=None, previous=None)
        current = _mean(getattr(bar, field) for bar in window[-n:])
        previous: float | None = None
        if len(window) >= n + 1:
            previous = _mean(getattr(bar, field) for bar in window[-n - 1 : -1])
        return IndicatorValue(value=current, previous=previous)

    # -- cross helpers ----------------------------------------------------------

    def cross_up(
        self, fast: IndicatorValue | None, slow: IndicatorValue | None
    ) -> bool:
        """Whether ``fast`` crossed strictly above ``slow`` on this bar."""
        return _crossed(fast, slow, direction="up")

    def cross_down(
        self, fast: IndicatorValue | None, slow: IndicatorValue | None
    ) -> bool:
        """Whether ``fast`` crossed strictly below ``slow`` on this bar."""
        return _crossed(fast, slow, direction="down")


class TickContext(_MarketContext):
    """Context of ``on_tick``: the realtime snapshot plus the shared reads."""

    def __init__(self, *, snapshot: Snapshot, **base: Any) -> None:
        super().__init__(**base)
        self._snapshot = snapshot

    @property
    def snapshot(self) -> Snapshot:
        return self._snapshot

    @property
    def symbol(self) -> str:
        symbol: str = self._snapshot.symbol
        return symbol


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    return sum(materialized) / len(materialized)


def _crossed(
    fast: IndicatorValue | None,
    slow: IndicatorValue | None,
    *,
    direction: str,
) -> bool:
    if fast is None or slow is None:
        return False
    if fast.value is None or slow.value is None:
        return False
    if fast.previous is None or slow.previous is None:
        return False
    if direction == "up":
        return fast.value > slow.value and fast.previous <= slow.previous
    return fast.value < slow.value and fast.previous >= slow.previous


class StrategyBase:
    """Base class of every strategy plugin.

    Subclasses declare parameters and override the hooks they need. Hooks
    do nothing by default; the runtime calls them in event order and
    propagates any exception (a failing strategy aborts the run, per the
    failure-behavior policy).
    """

    #: Parameter declaration; override in subclasses.
    params: Params = Params()

    def __init__(self, params: Mapping[str, Any] | None = None) -> None:
        declaration = type(self).params
        if not isinstance(declaration, Params):
            raise PulsarCoreError(
                f"{type(self).__name__}.params must be a Params declaration"
            )
        # The class attribute carries the declaration; the instance
        # attribute shadows it with the bound (effective) parameters, so
        # strategies read ``self.params.fast`` after configuration overrides.
        self.params = declaration.bind(params)  # type: ignore[assignment]

    # -- lifecycle hooks --------------------------------------------------------

    def on_start(self, ctx: BaseContext) -> None:
        """Called once when the run starts, before any market event."""

    def on_bar(self, ctx: BarContext) -> None:
        """Called for every bar event; declare targets through ``ctx``."""

    def on_tick(self, ctx: TickContext) -> None:
        """Called for every realtime snapshot event (Paper/Live feeds)."""

    def on_fill(self, ctx: FillContext) -> None:
        """Called for every confirmed fill flowing back from the venue."""

    def on_stop(self, ctx: BaseContext) -> None:
        """Called once when the run finishes, after the last event."""

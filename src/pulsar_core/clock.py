"""Time-source abstraction shared by backtest and realtime runs.

The kernel loop is mode-agnostic: it asks its :class:`Clock` for the current
time and, in backtests, advances virtual time event by event. Everything
downstream reads ``clock.now()`` only — a replayed past and the live present
are indistinguishable, which is what lets Research, Paper and Live share one
loop and one set of consumers.
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from pulsar_contracts import SHANGHAI_TZ

from .errors import ClockWentBackwardsError

__all__ = ["Clock", "BacktestClock", "RealtimeClock"]


def _shanghai(value: datetime) -> datetime:
    """Return ``value`` as an aware datetime in Asia/Shanghai."""
    if value.tzinfo is None:
        return value.replace(tzinfo=SHANGHAI_TZ)
    return value.astimezone(SHANGHAI_TZ)


@runtime_checkable
class Clock(Protocol):
    """The only time source visible to the kernel and its consumers."""

    @property
    def label(self) -> str:
        """``"backtest"`` or ``"realtime"`` — for logs and manifests only."""
        ...

    def now(self) -> datetime:
        """Current time as an aware Asia/Shanghai datetime."""
        ...

    def advance_to(self, ts: datetime) -> None:
        """Move time forward to ``ts``; never backwards.

        Backtest clocks jump their virtual time to the timestamp of the
        event being dispatched; realtime clocks treat this as a no-op
        because wall time advances on its own.
        """
        ...


class BacktestClock:
    """Virtual clock driven by the dispatch loop of a replay session.

    Time only moves when the bus dispatches an event: :meth:`advance_to`
    jumps forward to the event timestamp and refuses to move backwards,
    making the replayed timeline monotonic and the run reproducible.
    """

    def __init__(self, start: datetime) -> None:
        self._now = _shanghai(start)

    @property
    def label(self) -> str:
        return "backtest"

    def now(self) -> datetime:
        return self._now

    def advance_to(self, ts: datetime) -> None:
        target = _shanghai(ts)
        if target < self._now:
            raise ClockWentBackwardsError(
                f"backtest clock cannot move backwards: {target.isoformat()} < "
                f"{self._now.isoformat()}"
            )
        self._now = target

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"BacktestClock(now={self._now.isoformat()!r})"


class RealtimeClock:
    """Wall-clock time for Paper/Live runs.

    :meth:`advance_to` is a no-op: the loop calls it on every dispatch just
    like in backtests, but real time advances by itself.
    """

    @property
    def label(self) -> str:
        return "realtime"

    def now(self) -> datetime:
        return datetime.now(SHANGHAI_TZ)

    def advance_to(self, ts: datetime) -> None:
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"RealtimeClock(now={self.now().isoformat()!r})"

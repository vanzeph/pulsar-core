"""Error types raised by the pulsar-core kernel.

Failure behavior is part of the kernel contract (core-engine design):

* data gaps relative to the trading calendar abort the run and are reported —
  never silently skipped;
* a handler exception terminates the run while the events produced so far
  stay inspectable on the bus journal;
* backtest time never flows backwards.
"""

from __future__ import annotations

from datetime import date

__all__ = [
    "PulsarCoreError",
    "DataGapError",
    "ClockWentBackwardsError",
]


class PulsarCoreError(Exception):
    """Base class of every error raised by pulsar-core."""


class DataGapError(PulsarCoreError):
    """A replay session found trading days without bars.

    Raised when the fetched data does not cover a trading day of the
    calendar the session replays against. The run aborts with the missing
    days attached; it never skips them silently.
    """

    def __init__(self, missing_days: list[date]) -> None:
        self.missing_days: tuple[date, ...] = tuple(missing_days)
        count = len(self.missing_days)
        preview = ", ".join(day.isoformat() for day in self.missing_days[:5])
        if count > 5:
            preview += ", ..."
        super().__init__(f"data gap: {count} trading day(s) without bars ({preview})")


class ClockWentBackwardsError(PulsarCoreError):
    """The backtest clock was asked to move to a timestamp before its current time."""

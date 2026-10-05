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
    "ExperimentConfigError",
    "LifecycleError",
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


class ExperimentConfigError(PulsarCoreError):
    """An experiment TOML document is malformed or fails validation.

    Raised by the experiment loader for unknown sections or keys, missing
    required values, type mismatches, unregistered names (factor / model /
    preprocess step / portfolio method / universe) and sweep axes that do
    not address the template — a broken config must fail loudly at load
    time instead of half-running later.
    """


class LifecycleError(PulsarCoreError):
    """An experiment lifecycle rule was violated (模型配置生命周期).

    Raised when an assembly attempts to run an experiment in a mode its
    status does not admit (paper/live with a non-active config, anything
    with a retired one), when a status transition is illegal (only
    candidate->active and active->retired exist), when activation lacks
    explicit human confirmation, or when a retire/activate call is
    malformed (empty reason or operator, non-sha commit).

    Carries the offending ``status`` and/or ``mode`` when known; the
    message always states status, mode and the reason together so the
    failure is self-explaining in logs.
    """

    def __init__(
        self,
        message: str,
        *,
        status: "str | None" = None,
        mode: "str | None" = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.mode = mode

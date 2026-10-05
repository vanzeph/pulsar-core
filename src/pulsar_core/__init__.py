"""pulsar-core: the deterministic engine core of the Pulsar quant system.

This package implements the single-threaded deterministic event kernel, the
Clock abstraction that lets backtest and realtime runs share one loop, the
bar-level historical replay session, and the RunManifest reproducibility
mechanism.

Dependency policy (architecture baseline): pulsar-core depends only on
pulsar-contracts plus basic libraries. It must never import a data-source or
broker SDK, a network client, or any sibling pulsar package — a static
import-purity test enforces this.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bus import EventBus, Handler, canonical_event_json
from .clock import BacktestClock, Clock, RealtimeClock
from .errors import ClockWentBackwardsError, DataGapError, PulsarCoreError
from .events import Event, EventKind, SessionPhase, TimerPayload
from .manifest import RunManifest, bars_watermark, code_version, load_manifest
from .session import ReplaySession, RunResult

try:
    __version__ = version("pulsar-core")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # kernel loop
    "EventBus",
    "Handler",
    "canonical_event_json",
    # time sources
    "Clock",
    "BacktestClock",
    "RealtimeClock",
    # events
    "Event",
    "EventKind",
    "SessionPhase",
    "TimerPayload",
    # reproducibility
    "RunManifest",
    "bars_watermark",
    "code_version",
    "load_manifest",
    # replay session
    "ReplaySession",
    "RunResult",
    # errors
    "PulsarCoreError",
    "DataGapError",
    "ClockWentBackwardsError",
]

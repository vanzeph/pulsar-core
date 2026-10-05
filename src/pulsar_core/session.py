"""Bar-level historical replay session (Research mode skeleton).

Assembles one deterministic run on top of the shared kernel loop:

1. read the trading calendar and the bars from the ``MarketDataPort`` — the
   only data touchpoint of the core (ports come from pulsar-contracts;
   implementations live in other packages);
2. validate completeness against the calendar: a trading day without a
   single bar aborts the run with :class:`DataGapError` — never silently
   skipped;
3. build the :class:`~pulsar_core.manifest.RunManifest` (config snapshot +
   data watermarks + code version + seed);
4. replay day by day: publish each trading day's bars onto the bus and
   dispatch them — together with everything handlers publish back — inside
   the same single-threaded loop, while ``BacktestClock`` advances event by
   event.

The strategy framework, portfolio construction, risk gate and performance
accounting arrive in later tasks; this skeleton exposes the raw loop through
``subscribe``/``subscribe_all`` so those pieces plug in without touching the
kernel.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from pandas import DataFrame
from pydantic import Field

from pulsar_contracts import (
    AdjustMode,
    Bar,
    ContractModel,
    Freq,
    MarketDataPort,
    SHANGHAI_TZ,
)

from .bus import EventBus
from .clock import BacktestClock
from .errors import DataGapError, PulsarCoreError
from .events import Event, EventKind
from .manifest import RunManifest, bars_watermark, code_version

__all__ = ["ReplaySession", "RunResult"]

#: Canonical bar columns produced by ``MarketDataPort.fetch_bars``.
_BAR_COLUMNS = (
    "symbol",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjust_factor",
    "quality",
)

#: Required bar columns (``quality`` and ``adjust_factor`` may be defaulted).
_REQUIRED_BAR_COLUMNS = frozenset(_BAR_COLUMNS) - {"quality", "adjust_factor"}


class RunResult(ContractModel):
    """Outcome of one replay run — everything is deterministic.

    A rerun of the same manifest on the same code version must produce an
    equal ``RunResult`` (equality here is field-by-field on JSON-native
    values, i.e. bit-identical for floats produced by identical operations).
    """

    manifest: RunManifest
    trading_days: int = Field(ge=0)
    bar_events: int = Field(ge=0)
    processed_events: int = Field(ge=0)
    journal_digest: str = Field(min_length=1)

    @property
    def run_id(self) -> str:
        return self.manifest.run_id


def _day_start(day: date) -> datetime:
    """Left-closed bar timestamp convention: ``00:00`` Asia/Shanghai."""
    return datetime.combine(day, time(), tzinfo=SHANGHAI_TZ)


def _bars_from_frame(frame: DataFrame, freq: Freq) -> list[Bar]:
    """Convert a canonical ``fetch_bars`` frame into validated ``Bar`` objects.

    The port's canonical column set carries no ``freq`` column — the
    frequency is a query parameter, so it is injected here. Row order of the
    incoming frame is explicitly not relied upon (the port contract allows
    any order): rows are sorted by ``(ts, symbol)`` and duplicate
    ``(symbol, ts)`` rows are rejected as ambiguous data.
    """
    if frame is None or len(frame) == 0:
        return []
    missing = sorted(_REQUIRED_BAR_COLUMNS - set(frame.columns))
    if missing:
        raise PulsarCoreError(f"bar frame misses canonical columns: {missing}")
    ordered = frame.sort_values(by=["ts", "symbol"], kind="stable")
    columns = [column for column in _BAR_COLUMNS if column in ordered.columns]
    bars: list[Bar] = []
    seen: set[tuple[str, str]] = set()
    for raw in ordered[columns].to_dict(orient="records"):
        record: dict[str, Any] = {str(key): value for key, value in raw.items()}
        for optional in ("quality", "adjust_factor"):
            value = record.get(optional)
            if value is None or (isinstance(value, float) and value != value):
                record.pop(optional, None)  # NaN / missing -> model default
        bar = Bar(freq=freq, **record)
        key = (bar.symbol, bar.ts.isoformat())
        if key in seen:
            raise PulsarCoreError(
                f"ambiguous bar data: duplicate (symbol, ts) row {key}"
            )
        seen.add(key)
        bars.append(bar)
    return bars


class ReplaySession:
    """Bar-level historical replay over the deterministic kernel loop.

    Parameters mirror what a run configuration carries: the port (injected
    implementation of ``MarketDataPort``), the replay window, the bar
    frequency and adjustment mode, the random seed, an optional config
    snapshot and code version for the manifest, and optionally a
    pre-built bus (tests and future assembly layers may wire their own).
    """

    def __init__(
        self,
        *,
        port: MarketDataPort,
        symbols: list[str],
        start: date,
        end: date,
        freq: Freq = Freq.DAILY,
        adjust: AdjustMode = AdjustMode.FORWARD,
        seed: int = 0,
        config: dict[str, Any] | None = None,
        code_version: str | None = None,
        bus: EventBus | None = None,
    ) -> None:
        if start > end:
            raise ValueError(f"start {start} must not precede end {end}")
        if not symbols:
            raise ValueError("symbols must not be empty")
        self._port = port
        self._symbols = sorted(symbols)
        self._start = start
        self._end = end
        self._freq = freq
        self._adjust = adjust
        self._seed = seed
        self._config: dict[str, Any] = dict(config) if config else {}
        self._code_version = code_version
        self._bus = bus if bus is not None else EventBus(
            BacktestClock(_day_start(start))
        )
        #: Deterministic per-run RNG. Everything inside a run that needs
        #: randomness draws from here — never from the global ``random``
        #: module — so reruns replay the same draw sequence.
        self.rng = random.Random(seed)
        self._ran = False

    # -- wiring -------------------------------------------------------------

    @property
    def bus(self) -> EventBus:
        return self._bus

    @property
    def symbols(self) -> list[str]:
        return list(self._symbols)

    def subscribe(self, kind: EventKind, handler: Callable[[Event], None]) -> None:
        """Subscribe a handler to one event kind on the session's bus."""
        self._bus.subscribe(kind, handler)

    def subscribe_all(self, handler: Callable[[Event], None]) -> None:
        """Subscribe a wildcard handler to every event on the session's bus."""
        self._bus.subscribe_all(handler)

    # -- execution ----------------------------------------------------------

    def run(self) -> RunResult:
        """Fetch, validate, build the manifest and replay the window."""
        if self._ran:
            raise PulsarCoreError("ReplaySession is single-use; build a new one")
        self._ran = True

        trading_days = self._port.calendar(self._start, self._end)
        if not trading_days:
            raise PulsarCoreError(
                f"trading calendar is empty for [{self._start}, {self._end}]"
            )

        frame = self._port.fetch_bars(
            self._symbols, self._start, self._end, self._freq, self._adjust
        )
        bars = _bars_from_frame(frame, self._freq)

        bars_by_day: dict[date, list[Bar]] = {}
        for bar in bars:
            bars_by_day.setdefault(bar.ts.date(), []).append(bar)

        missing_days = [day for day in trading_days if not bars_by_day.get(day)]
        if missing_days:
            raise DataGapError(missing_days)

        manifest = RunManifest.build(
            mode="research",
            seed=self._seed,
            config=self._snapshot_config(),
            code_version=self._code_version,
            data_watermarks=bars_watermark(bars),
        )

        self._replay(trading_days, bars_by_day)

        return RunResult(
            manifest=manifest,
            trading_days=len(trading_days),
            bar_events=sum(len(day_bars) for day_bars in bars_by_day.values()),
            processed_events=len(self._bus.journal),
            journal_digest=self._bus.journal_digest,
        )

    # -- internals ----------------------------------------------------------

    def _snapshot_config(self) -> dict[str, Any]:
        """Session inputs merged over the user config for the manifest."""
        snapshot: dict[str, Any] = {
            "session": {
                "kind": "bar_replay",
                "symbols": list(self._symbols),
                "start": self._start,
                "end": self._end,
                "freq": self._freq.value,
                "adjust": self._adjust.value,
            }
        }
        snapshot.update(self._config)
        return snapshot

    def _replay(self, trading_days: list[date], bars_by_day: dict[date, list[Bar]]) -> None:
        """Drive the bus: one dispatch window per trading day.

        Each window ``[day, next_trading_day)`` processes the day's bars plus
        every follow-up event handlers produced within the day; events
        scheduled into a future window stay queued and dispatch in order
        when that window opens. After the last day the loop is fully
        drained so nothing scheduled during the run is lost.
        """
        first_day = trading_days[0]
        self._bus.publish(Event.session_started(_day_start(first_day)))

        for index, day in enumerate(trading_days):
            for bar in bars_by_day.get(day, []):
                self._bus.publish(Event.of_bar(bar))
            window_end = (
                _day_start(trading_days[index + 1])
                if index + 1 < len(trading_days)
                else _day_start(day + timedelta(days=1))
            )
            self._bus.run(until=window_end)

        journal_ts = [event.ts for event in self._bus.journal]
        finished_ts = max(journal_ts) if journal_ts else _day_start(first_day)
        self._bus.publish(Event.session_finished(finished_ts))
        self._bus.run()

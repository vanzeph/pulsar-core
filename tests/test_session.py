"""ReplaySession: calendar-driven replay, gap abort, input validation."""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta

import pandas
import pytest
from pulsar_contracts import Freq, SHANGHAI_TZ

from conftest import FakeMarketDataPort

from pulsar_core import (
    DataGapError,
    EventKind,
    PulsarCoreError,
    ReplaySession,
    SessionPhase,
)

SYMBOLS = ["600000", "000001"]
CODE_VERSION = "test-code-version"


def make_session(port, start, end, **overrides) -> ReplaySession:
    kwargs = dict(
        port=port,
        symbols=SYMBOLS,
        start=start,
        end=end,
        seed=99,
        code_version=CODE_VERSION,
    )
    kwargs.update(overrides)
    return ReplaySession(**kwargs)


def test_happy_path_replays_every_trading_day_in_order(window) -> None:
    start, end = window
    port = FakeMarketDataPort()
    session = make_session(port, start, end)
    bars: list = []
    session.subscribe(EventKind.MARKET, bars.append)
    result = session.run()

    expected_days = port.calendar(start, end)
    expected_bars = port.bars(SYMBOLS, start, end)
    assert result.trading_days == len(expected_days) == 10
    assert result.bar_events == len(expected_bars) == 20
    assert result.processed_events == 22  # 20 bars + started + finished
    assert [event.bar for event in bars] == expected_bars  # (ts, symbol) order

    journal = session.bus.journal
    assert journal[0].session_phase is SessionPhase.STARTED
    assert journal[-1].session_phase is SessionPhase.FINISHED
    assert journal[-1].ts == expected_bars[-1].ts
    # within one day bars dispatch in (ts, symbol) order
    first_day_bars = [
        event.bar for event in bars if event.bar.ts.date() == expected_days[0]
    ]
    assert [bar.symbol for bar in first_day_bars] == ["000001", "600000"]


def test_manifest_snapshot_contains_session_inputs(window) -> None:
    start, end = window
    result = make_session(FakeMarketDataPort(), start, end).run()
    config = result.manifest.config
    assert config["session"] == {
        "kind": "bar_replay",
        "symbols": sorted(SYMBOLS),
        "start": start.isoformat(),  # manifest config is JSON-native
        "end": end.isoformat(),
        "freq": Freq.DAILY.value,
        "adjust": "forward",
    }
    assert result.manifest.mode == "research"
    assert result.manifest.seed == 99
    assert result.manifest.code_version == CODE_VERSION
    assert set(result.manifest.data_watermarks) == {
        "bars/1d/000001",
        "bars/1d/600000",
    }
    assert result.manifest.run_id == result.run_id


def test_backtest_clock_follows_the_replay(window) -> None:
    start, end = window
    session = make_session(FakeMarketDataPort(), start, end)
    session.run()
    expected_bars = FakeMarketDataPort().bars(SYMBOLS, start, end)
    assert session.bus.now == expected_bars[-1].ts
    assert session.bus.clock.label == "backtest"


def test_data_gap_aborts_and_reports_missing_days(window) -> None:
    start, end = window
    gap_days = [date(2026, 6, 3), date(2026, 6, 4)]
    port = FakeMarketDataPort(gap_dates=gap_days)
    session = make_session(port, start, end)
    with pytest.raises(DataGapError) as excinfo:
        session.run()
    assert sorted(excinfo.value.missing_days) == gap_days
    assert "2026-06-03" in str(excinfo.value)
    # aborted before any dispatch: nothing was silently replayed
    assert session.bus.journal == ()


def test_empty_calendar_is_an_error() -> None:
    class EmptyCalendarPort(FakeMarketDataPort):
        def calendar(self, start, end):
            return []

    session = make_session(EmptyCalendarPort(), date(2026, 6, 1), date(2026, 6, 5))
    with pytest.raises(PulsarCoreError, match="calendar is empty"):
        session.run()


def test_bad_windows_and_symbols_are_rejected(window) -> None:
    start, end = window
    port = FakeMarketDataPort()
    with pytest.raises(ValueError, match="must not precede"):
        make_session(port, end, start)
    with pytest.raises(ValueError, match="symbols"):
        make_session(port, start, end, symbols=[])


def test_session_is_single_use(window) -> None:
    start, end = window
    session = make_session(FakeMarketDataPort(), start, end)
    session.run()
    with pytest.raises(PulsarCoreError, match="single-use"):
        session.run()


def test_duplicate_bar_rows_are_ambiguous_data(window) -> None:
    start, end = window

    class DuplicatePort(FakeMarketDataPort):
        def fetch_bars(self, symbols, start, end, freq, adjust):
            frame = super().fetch_bars(symbols, start, end, freq, adjust)
            return pandas.concat([frame, frame.iloc[[0]]])

    session = make_session(DuplicatePort(), start, end)
    with pytest.raises(PulsarCoreError, match="duplicate"):
        session.run()


def test_missing_canonical_column_is_rejected(window) -> None:
    start, end = window

    class NoClosePort(FakeMarketDataPort):
        def fetch_bars(self, symbols, start, end, freq, adjust):
            frame = super().fetch_bars(symbols, start, end, freq, adjust)
            return frame.drop(columns=["close"])

    session = make_session(NoClosePort(), start, end)
    with pytest.raises(PulsarCoreError, match="canonical columns"):
        session.run()


def test_timer_scheduled_for_next_day_dispatches_in_that_window(window) -> None:
    start, end = window
    port = FakeMarketDataPort()
    session = make_session(port, start, end)
    fired: list = []

    def on_bar(event):
        if event.bar.ts.date() == start and event.bar.symbol == "600000":
            next_day = start + timedelta(days=1)  # 2026-06-02, a trading day
            when = datetime.combine(next_day, time(), tzinfo=SHANGHAI_TZ)
            session.bus.schedule(when, "eod_hook", {"close": event.bar.close})

    def on_timer(event):
        fired.append(event)

    session.subscribe(EventKind.MARKET, on_bar)
    session.subscribe(EventKind.TIMER, on_timer)
    session.run()

    assert len(fired) == 1
    timer = fired[0]
    assert timer.ts == datetime(2026, 6, 2, 0, 0, tzinfo=SHANGHAI_TZ)
    # The timer was published during day 1 (earlier publish seq): it
    # dispatches before day 2's bars despite the equal timestamp.
    day2_events = [
        event for event in session.bus.journal if event.ts.date() == date(2026, 6, 2)
    ]
    assert day2_events[0].kind is EventKind.TIMER
    assert day2_events[0].timer.name == "eod_hook"
    assert day2_events[1].kind is EventKind.MARKET


def test_rng_is_seeded_from_the_manifest_seed(window) -> None:
    start, end = window
    session = make_session(FakeMarketDataPort(), start, end, seed=1234)
    reference = random.Random(1234)
    assert [session.rng.random() for _ in range(3)] == [
        reference.random() for _ in range(3)
    ]

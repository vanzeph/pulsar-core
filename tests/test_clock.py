"""Clock abstraction: backtest virtual time vs realtime wall time."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from pulsar_contracts import SHANGHAI_TZ

from pulsar_core import BacktestClock, Clock, ClockWentBackwardsError, RealtimeClock


def test_backtest_clock_starts_normalized_to_shanghai() -> None:
    naive = datetime(2026, 6, 1, 9, 30)
    clock = BacktestClock(naive)
    assert clock.now() == datetime(2026, 6, 1, 9, 30, tzinfo=SHANGHAI_TZ)
    assert clock.now().tzinfo is SHANGHAI_TZ
    assert clock.label == "backtest"


def test_backtest_clock_converts_foreign_tz() -> None:
    utc = datetime(2026, 6, 1, 1, 0, tzinfo=timezone.utc)
    clock = BacktestClock(utc)
    assert clock.now() == datetime(2026, 6, 1, 9, 0, tzinfo=SHANGHAI_TZ)


def test_backtest_clock_advances_forward_only() -> None:
    clock = BacktestClock(datetime(2026, 6, 1, tzinfo=SHANGHAI_TZ))
    clock.advance_to(datetime(2026, 6, 2, 15, tzinfo=SHANGHAI_TZ))
    assert clock.now() == datetime(2026, 6, 2, 15, tzinfo=SHANGHAI_TZ)
    clock.advance_to(datetime(2026, 6, 2, 15, tzinfo=SHANGHAI_TZ))  # idempotent
    try:
        clock.advance_to(datetime(2026, 6, 2, 14, tzinfo=SHANGHAI_TZ))
    except ClockWentBackwardsError:
        pass
    else:  # pragma: no cover - the assertion is the exception
        raise AssertionError("expected ClockWentBackwardsError")


def test_realtime_clock_reads_wall_time_in_shanghai() -> None:
    clock = RealtimeClock()
    before = datetime.now(SHANGHAI_TZ) - timedelta(seconds=1)
    now = clock.now()
    after = datetime.now(SHANGHAI_TZ) + timedelta(seconds=1)
    assert before <= now <= after
    assert now.tzinfo is SHANGHAI_TZ
    assert clock.label == "realtime"
    clock.advance_to(now + timedelta(days=1))  # no-op, must not raise
    assert clock.now() <= after


def test_both_clocks_satisfy_the_protocol() -> None:
    backtest = BacktestClock(datetime(2026, 6, 1, tzinfo=SHANGHAI_TZ))
    realtime = RealtimeClock()
    assert isinstance(backtest, Clock)
    assert isinstance(realtime, Clock)

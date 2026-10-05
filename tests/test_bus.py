"""EventBus: ordering, windows, journaling and failure behavior."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from pulsar_contracts import SHANGHAI_TZ

from pulsar_core import BacktestClock, EventBus, EventKind, RealtimeClock
from pulsar_core.bus import canonical_event_json


def ts(day: int, hour: int = 0) -> datetime:
    return datetime(2026, 6, day, hour, tzinfo=SHANGHAI_TZ)


def make_bus() -> tuple[EventBus, list]:
    bus = EventBus(BacktestClock(ts(1)))
    seen: list = []
    return bus, seen


def test_events_dispatch_in_timestamp_order_regardless_of_publish_order() -> None:
    bus, seen = make_bus()
    bus.subscribe_all(seen.append)
    for when, name in [(ts(3), "third"), (ts(1), "first"), (ts(2), "second")]:
        bus.schedule(when, name)
    assert bus.run() == 3
    assert [event.timer.name for event in seen] == ["first", "second", "third"]


def test_same_timestamp_events_dispatch_in_publish_order() -> None:
    bus, seen = make_bus()
    bus.subscribe_all(seen.append)
    for name in ["a", "b", "c"]:
        bus.schedule(ts(1, 9), name)
    bus.run()
    assert [event.timer.name for event in seen] == ["a", "b", "c"]


def test_run_until_keeps_future_events_queued() -> None:
    bus, _ = make_bus()
    bus.schedule(ts(1), "today")
    bus.schedule(ts(2), "tomorrow")
    processed = bus.run(until=ts(2))
    assert processed == 1
    assert bus.pending == 1
    assert bus.run() == 1  # the deferred event dispatches in the next pass


def test_until_bound_is_exclusive() -> None:
    bus, _ = make_bus()
    bus.schedule(ts(1, 10), "at-boundary")
    assert bus.run(until=ts(1, 10)) == 0
    assert bus.pending == 1


def test_handler_can_publish_during_dispatch() -> None:
    bus, seen = make_bus()
    bus.subscribe_all(seen.append)

    def chain(event):  # publish follow-up work from inside a handler
        if event.timer.name == "kick-off":
            bus.schedule(event.ts + timedelta(hours=1), "follow-up")

    bus.subscribe(EventKind.TIMER, chain)
    bus.schedule(ts(1), "kick-off")
    assert bus.run() == 2
    assert [event.timer.name for event in seen] == ["kick-off", "follow-up"]


def test_kind_handlers_run_before_wildcards_in_subscription_order() -> None:
    bus, seen = make_bus()
    bus.subscribe_all(lambda event: seen.append("wild-1"))
    bus.subscribe(EventKind.TIMER, lambda event: seen.append("kind-1"))
    bus.subscribe(EventKind.TIMER, lambda event: seen.append("kind-2"))
    bus.subscribe_all(lambda event: seen.append("wild-2"))
    bus.schedule(ts(1), "x")
    bus.run()
    assert seen == ["kind-1", "kind-2", "wild-1", "wild-2"]


def test_handler_exception_propagates_and_preserves_journal() -> None:
    bus, _ = make_bus()
    seen: list = []

    def boom(event):
        if event.timer.name == "boom":
            raise RuntimeError("strategy failed")

    bus.subscribe(EventKind.TIMER, boom)
    bus.subscribe_all(seen.append)
    bus.schedule(ts(1), "ok-1")
    bus.schedule(ts(2), "boom")
    bus.schedule(ts(3), "never")
    with pytest.raises(RuntimeError, match="strategy failed"):
        bus.run()
    # Failure behavior: the run terminates, dispatched events stay inspectable,
    # undispatched events stay queued. The exception fires in the kind
    # handler, so the wildcard recorder never sees the failing event.
    assert [event.timer.name for event in bus.journal] == ["ok-1", "boom"]
    assert bus.pending == 1
    assert [event.timer.name for event in seen] == ["ok-1"]


def test_backtest_clock_advances_with_dispatch() -> None:
    bus = EventBus(BacktestClock(ts(1)))
    bus.schedule(ts(5, 15), "later")
    bus.run()
    assert bus.now == ts(5, 15)


def test_poll_dispatches_only_events_due_now() -> None:
    bus = EventBus(RealtimeClock())
    bus.schedule(ts(1), "past")  # 2026-06-01 already lies in the past
    future_when = datetime.now(SHANGHAI_TZ) + timedelta(hours=1)
    bus.schedule(future_when, "future")
    assert bus.poll() == 1
    assert [event.timer.name for event in bus.journal] == ["past"]
    assert bus.pending == 1  # the future event stays queued ...
    assert bus.run(until=future_when + timedelta(seconds=1)) == 1  # ... and dispatches once due
    assert [event.timer.name for event in bus.journal] == ["past", "future"]


def test_journal_digest_stable_and_sensitive() -> None:
    bus_a, _ = make_bus()
    bus_a.schedule(ts(1), "one")
    bus_a.schedule(ts(2), "two")
    bus_a.run()

    bus_b, _ = make_bus()
    bus_b.schedule(ts(1), "one")
    bus_b.schedule(ts(2), "two")
    bus_b.run()
    assert bus_a.journal_digest == bus_b.journal_digest

    bus_c, _ = make_bus()
    bus_c.schedule(ts(1), "one")
    bus_c.schedule(ts(2), "TWO")  # different payload
    bus_c.run()
    assert bus_a.journal_digest != bus_c.journal_digest


def test_canonical_event_json_is_sorted_and_compact() -> None:
    bus, _ = make_bus()
    event = bus.schedule(ts(1), "x", {"b": 1, "a": 2})
    text = canonical_event_json(event)
    assert text.index('"a"') < text.index('"b"')
    assert ", " not in text
    assert '"kind":"timer"' in text

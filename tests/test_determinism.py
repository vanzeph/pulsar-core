"""Determinism acceptance: same inputs, two runs, bit-identical results.

The core-engine promise under test: rerunning the same manifest inputs on
the same code version reproduces the run exactly — event by event, digest
by digest, and in every float the consumer computed along the way. A
proto-strategy records arithmetic over closes, draws from the seeded run
RNG and schedules follow-up timers, so the check exercises the whole loop,
not just bar publication.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from pulsar_contracts import SHANGHAI_TZ

from conftest import FakeMarketDataPort

from pulsar_core import EventKind, ReplaySession

SYMBOLS = ["600000", "000001"]
CODE_VERSION = "test-code-version"
START, END = date(2026, 6, 1), date(2026, 6, 12)


class ProtoStrategy:
    """Loop consumer standing in for the future strategy framework.

    Reads bars, accumulates float arithmetic, draws from the seeded RNG,
    schedules next-day timers and reacts to them — everything a run may do
    that could leak nondeterminism.
    """

    def __init__(self, session: ReplaySession) -> None:
        self.session = session
        self.trace: list[tuple] = []
        self.close_sum = 0.0
        session.subscribe(EventKind.MARKET, self._on_market)
        session.subscribe(EventKind.TIMER, self._on_timer)

    def _on_market(self, event) -> None:
        bar = event.bar
        self.close_sum += bar.close * self.session.rng.random()
        self.trace.append(
            (
                "bar",
                bar.symbol,
                bar.ts.isoformat(),
                bar.close,
                round(self.close_sum, 12),
            )
        )
        if bar.symbol == "600000":
            when = datetime.combine(
                bar.ts.date() + timedelta(days=1), time(), tzinfo=SHANGHAI_TZ
            )
            self.session.bus.schedule(when, "eod", {"sum": round(self.close_sum, 12)})

    def _on_timer(self, event) -> None:
        self.trace.append(
            ("timer", event.timer.name, event.ts.isoformat(), event.timer.data["sum"])
        )


def run_once(seed: int = 42):
    session = ReplaySession(
        port=FakeMarketDataPort(),
        symbols=SYMBOLS,
        start=START,
        end=END,
        seed=seed,
        code_version=CODE_VERSION,
    )
    strategy = ProtoStrategy(session)
    result = session.run()
    return result, strategy.trace, session.bus.journal


def test_two_runs_of_the_same_inputs_are_bit_identical() -> None:
    result_a, trace_a, journal_a = run_once()
    result_b, trace_b, journal_b = run_once()

    # 1. run results equal field by field (manifest included)
    assert result_a == result_b
    assert result_a.run_id == result_b.run_id
    # 2. manifests serialize to the identical document
    assert result_a.manifest.to_json() == result_b.manifest.to_json()
    # 3. journals identical event by event (dispatch order and payloads)
    assert journal_a == journal_b
    # 4. the digest — the bit-level identity check — matches
    assert result_a.journal_digest == result_b.journal_digest
    # 5. consumer-side arithmetic (floats included) reproduced exactly
    assert trace_a == trace_b


def test_manifest_round_trip_preserves_identity(tmp_path) -> None:
    result_a, _, _ = run_once()
    path = result_a.manifest.write(tmp_path / "run_a.json")

    from pulsar_core import load_manifest

    loaded = load_manifest(path)
    result_b, _, _ = run_once()
    assert loaded == result_b.manifest
    assert loaded.to_json() == result_b.manifest.to_json()
    assert loaded.matches_code(CODE_VERSION)


def test_different_seed_changes_the_run_id_and_the_journal() -> None:
    result_a, _, _ = run_once(seed=42)
    result_b, _, _ = run_once(seed=43)
    # guard against a vacuous equality test above
    assert result_a.run_id != result_b.run_id
    assert result_a.journal_digest != result_b.journal_digest

"""Run artifacts: events.parquet contract, replay recompute, run directory.

The replay test is the reproducibility acceptance of this area: write the
journal to events.parquet, read it back, replay the read-back stream and
assert the recomputed equity curve equals the original — an archived run
must recompute bit-identically from the artifact alone.

pyarrow is imported *inside* the tests that inspect parquet directly, and
this module is named to sort after ``test_import_purity.py``: that test
asserts on global ``sys.modules`` after a plain ``import pulsar_core``,
and pyarrow transitively loads ``socket``. The library keeps pyarrow out
of its import path; tests that trigger archive writing (and thereby load
pyarrow) must run after the purity check.
"""

from __future__ import annotations

import pytest

from conftest import FillingExecutionPort, ScriptedClosesPort, down_up_down
from test_performance import INITIAL_CASH, known_stream

from pulsar_core import (
    EVENTS_FILENAME,
    EVENTS_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    BacktestClock,
    EventBus,
    Params,
    PulsarCoreError,
    ReplaySession,
    RiskGate,
    SinglePositionCapRule,
    StrategyBase,
    StrategyRuntime,
    build_metrics_report,
    compute_equity_curve,
    load_manifest,
    load_metrics_report,
    read_event_archive,
    write_event_archive,
    write_run_artifacts,
)
from pulsar_core.session import _day_start

KNOWN_STREAM = known_stream()


class RoundTrip(StrategyBase):
    """Momentum entry on a 0.5% up-move, exit on a 0.5% down-move."""

    params = Params()

    def on_bar(self, ctx) -> None:
        window = ctx.bars(10)
        if len(window) < 2:
            return
        previous = window[-2].close
        if ctx.bar.close > previous * 1.005:
            ctx.target_weight(ctx.symbol, 0.5)
        elif ctx.bar.close < previous * 0.995:
            ctx.target_weight(ctx.symbol, 0.0)


def run_session(directory):
    closes = down_up_down(6, 8, 6)
    port = ScriptedClosesPort({"600000": closes})
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    runtime = StrategyRuntime(
        bus=bus,
        port=venue,
        strategy=RoundTrip(),
        gate=RiskGate((SinglePositionCapRule(1.0),)),
        initial_cash=INITIAL_CASH,
    )
    session = ReplaySession(
        port=port,
        symbols=["600000"],
        start=start,
        end=end,
        seed=11,
        config={"strategy": {"name": "round_trip"}},
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    result = session.run()
    artifacts = write_run_artifacts(
        result,
        events=bus.journal,
        initial_cash=INITIAL_CASH,
        directory=directory,
    )
    return result, bus, artifacts


class TestEventArchiveContract:
    def test_schema_version_column_and_row_count(self, tmp_path) -> None:
        import pyarrow.parquet as pq

        events = KNOWN_STREAM
        path = write_event_archive(
            events, tmp_path / EVENTS_FILENAME, run_id="arch-run"
        )
        table = pq.read_table(path)
        assert table.num_rows == len(events)
        assert "schema_version" in table.schema.names
        assert set(table.column("schema_version").to_pylist()) == {EVENTS_SCHEMA_VERSION}
        # identity + query columns of the contract
        for column in (
            "run_id",
            "seq",
            "ts",
            "kind",
            "event_type",
            "symbol",
            "side",
            "price",
            "quantity",
            "commission",
            "stamp_duty",
            "transfer_fee",
            "payload",
        ):
            assert column in table.schema.names
        assert table.column("run_id").to_pylist() == ["arch-run"] * len(events)
        assert table.column("seq").to_pylist() == list(range(1, len(events) + 1))
        # dispatch order is preserved (kinds in journal order)
        kinds = [event.kind.value for event in events]
        assert table.column("kind").to_pylist() == kinds
        # fill rows carry the denormalized execution columns
        fill_rows = [
            (index, event.execution.fill)
            for index, event in enumerate(events)
            if event.execution is not None and event.execution.fill is not None
        ]
        assert fill_rows  # the known stream does trade
        for index, fill in fill_rows:
            assert table.column("symbol")[index].as_py() == fill.symbol
            assert table.column("side")[index].as_py() == fill.side.value
            assert table.column("price")[index].as_py() == fill.price
            assert table.column("quantity")[index].as_py() == fill.quantity
            assert table.column("commission")[index].as_py() == fill.commission
            assert table.column("stamp_duty")[index].as_py() == fill.stamp_duty
            assert table.column("transfer_fee")[index].as_py() == fill.transfer_fee

    def test_empty_journal_writes_a_schema_only_archive(self, tmp_path) -> None:
        import pyarrow.parquet as pq

        path = write_event_archive([], tmp_path / EVENTS_FILENAME, run_id="empty")
        table = pq.read_table(path)
        assert table.num_rows == 0
        assert "schema_version" in table.schema.names
        assert read_event_archive(path) == ()

    def test_unknown_schema_version_is_rejected_loudly(self, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = write_event_archive(
            KNOWN_STREAM, tmp_path / EVENTS_FILENAME, run_id="future-run"
        )
        table = pq.read_table(path)
        bumped = table.set_column(
            table.schema.get_field_index("schema_version"),
            "schema_version",
            pa.array([EVENTS_SCHEMA_VERSION + 1] * table.num_rows, type=pa.int64()),
        )
        pq.write_table(bumped, path)
        with pytest.raises(PulsarCoreError, match="not supported"):
            read_event_archive(path)

    def test_archive_missing_contract_columns_is_rejected(self, tmp_path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "broken.parquet"
        pq.write_table(pa.table({"seq": pa.array([1], type=pa.int64())}), path)
        with pytest.raises(PulsarCoreError, match="misses the 'schema_version' column"):
            read_event_archive(path)

    def test_events_round_trip_field_by_field(self, tmp_path) -> None:
        events = KNOWN_STREAM
        path = write_event_archive(
            events, tmp_path / EVENTS_FILENAME, run_id="roundtrip"
        )
        assert read_event_archive(path) == tuple(events)


class TestReplayFromArchive:
    def test_written_events_replay_the_same_equity_curve(self, tmp_path) -> None:
        """写 events -> 读回 -> 重算 -> 相等 (the replay acceptance)."""
        events = KNOWN_STREAM
        path = write_event_archive(events, tmp_path / EVENTS_FILENAME, run_id="replay")
        read_back = read_event_archive(path)

        original_curve = compute_equity_curve(events, initial_cash=INITIAL_CASH)
        recomputed_curve = compute_equity_curve(read_back, initial_cash=INITIAL_CASH)
        assert recomputed_curve == original_curve

        original_report = build_metrics_report(
            events, initial_cash=INITIAL_CASH, run_id="replay"
        )
        recomputed_report = build_metrics_report(
            read_back, initial_cash=INITIAL_CASH, run_id="replay"
        )
        assert recomputed_report == original_report
        assert recomputed_report.to_json() == original_report.to_json()


class TestRunDirectory:
    def test_artifacts_land_side_by_side_and_load_back(self, tmp_path) -> None:
        directory = tmp_path / "run-dir"
        result, bus, artifacts = run_session(directory)

        assert sorted(path.name for path in directory.iterdir()) == sorted(
            [MANIFEST_FILENAME, EVENTS_FILENAME, METRICS_FILENAME]
        )
        assert artifacts.run_id == result.run_id

        manifest = load_manifest(directory / MANIFEST_FILENAME)
        assert manifest == result.manifest
        assert manifest.run_id == result.run_id

        report = load_metrics_report(directory / METRICS_FILENAME)
        assert report.run_id == result.run_id
        assert report.journal_digest == result.journal_digest
        assert report.initial_cash == INITIAL_CASH
        assert report.metrics.trading_days == result.trading_days
        assert len(report.equity_curve) == result.trading_days + 1  # + start point

        events = read_event_archive(directory / EVENTS_FILENAME)
        assert len(events) == result.processed_events
        assert events == bus.journal

    def test_archived_run_recomputes_its_report_bit_identically(self, tmp_path) -> None:
        result, _, artifacts = run_session(tmp_path / "recompute")
        events = read_event_archive(artifacts.events_path)
        recomputed = build_metrics_report(
            events,
            initial_cash=INITIAL_CASH,
            run_id=result.run_id,
            journal_digest=result.journal_digest,
        )
        assert recomputed == artifacts.report
        assert recomputed.to_json() == artifacts.report.to_json()

    def test_two_identical_runs_produce_identical_artifacts(self, tmp_path) -> None:
        import pyarrow.parquet as pq

        first_result, _, first = run_session(tmp_path / "run-a")
        second_result, _, second = run_session(tmp_path / "run-b")

        assert first_result.run_id == second_result.run_id
        assert first.report.to_json() == second.report.to_json()
        # same journal -> same archived content (compared as tables, not bytes)
        assert read_event_archive(first.events_path) == read_event_archive(
            second.events_path
        )
        assert pq.read_table(first.events_path).equals(
            pq.read_table(second.events_path)
        )

    def test_report_metrics_reflect_the_venue_fills(self, tmp_path) -> None:
        _, _, artifacts = run_session(tmp_path / "metrics")
        report = artifacts.report
        # the filling venue charges zero fees and fills at the limit price,
        # which equals the reference mark -> zero total fee attribution
        assert report.fee_attribution.total == pytest.approx(0.0)
        assert report.fee_attribution.fills >= 1
        assert report.metrics.turnover_value > 0.0
        # entering during the rally and riding it must lift NAV above 1
        navs = [point.nav for point in report.equity_curve]
        assert max(navs) > 1.0
        assert report.start is not None and report.end is not None

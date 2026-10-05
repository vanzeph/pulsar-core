"""Run artifact persistence: the ``events.parquet`` archive and the run dir.

The visualization layer depends on three stable data contracts only —
RunManifest (JSON), the event archive (``events.parquet``) and the metrics
report (JSON) — and never imports pulsar code. This module produces two of
them and lays out the run directory all three live in:

    <run dir>/run_manifest.json     reproducibility record (RunManifest)
    <run dir>/events.parquet        full event journal, schema-versioned
    <run dir>/metrics_report.json   equity curve + metrics + fee attribution

The parquet schema is the contract surface: every row is one dispatched
event with a ``schema_version`` column (readers reject versions they do
not understand), identity columns (``run_id``, ``seq``, ``ts``, ``kind``),
denormalized query columns for DuckDB-side filtering (``event_type``,
``symbol``, ``side``, ``price``, ``quantity``, the three fee columns) and
a ``payload`` column carrying the event's canonical JSON — the lossless
source read_event_archive() replays from, so an archived run recomputes
bit-identically. Archiving is deliberately opt-in ("事件流可选全量落盘"):
callers that need replay/attribution write the archive, minimal runs skip
it.

Purity note: pyarrow (the parquet engine) transitively imports ``socket``,
so it is imported *inside* the archive functions, never at module import
time — a plain ``import pulsar_core`` must stay free of the network stack
(the runtime import-purity test asserts exactly that). Writing an archive
is local disk I/O only; the engine never opens a network session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pydantic import Field

from pulsar_contracts import ContractModel

from .bus import canonical_event_json
from .errors import PulsarCoreError
from .events import Event, EventKind, SessionPhase
from .performance import MetricsReport, build_metrics_report
from .session import RunResult

__all__ = [
    "EVENTS_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "EVENTS_FILENAME",
    "METRICS_FILENAME",
    "RunArtifacts",
    "write_event_archive",
    "read_event_archive",
    "write_run_artifacts",
]

#: Version of the events.parquet schema. Bump on any column change; readers
#: reject archives whose schema_version they do not implement.
EVENTS_SCHEMA_VERSION = 1

#: Canonical file names of the run-directory contract.
MANIFEST_FILENAME = "run_manifest.json"
EVENTS_FILENAME = "events.parquet"
METRICS_FILENAME = "metrics_report.json"

#: Event archive columns as ``(name, arrow type, nullable)`` triples. Kept
#: pyarrow-free at module level (see the purity note above); the arrow
#: schema is materialized inside the writer.
_EVENT_COLUMNS: tuple[tuple[str, str, bool], ...] = (
    ("schema_version", "int64", False),
    ("run_id", "string", False),
    ("seq", "int64", False),
    ("ts", "string", False),
    ("kind", "string", False),
    ("event_type", "string", False),
    ("symbol", "string", True),
    ("side", "string", True),
    ("price", "float64", True),
    ("quantity", "int64", True),
    ("commission", "float64", True),
    ("stamp_duty", "float64", True),
    ("transfer_fee", "float64", True),
    ("payload", "string", False),
)

_EVENT_COLUMN_NAMES = tuple(name for name, _, _ in _EVENT_COLUMNS)


class RunArtifacts(ContractModel):
    """Paths and payload of one run's persisted artifacts."""

    run_id: str = Field(min_length=1)
    directory: str
    manifest_path: str
    events_path: str
    metrics_path: str
    report: MetricsReport


def _event_type(event: Event) -> str:
    """Refined event classifier used as the ``event_type`` column."""
    if event.kind is EventKind.SESSION:
        assert event.session_phase is not None  # envelope invariant
        return str(event.session_phase.value)
    if event.kind is EventKind.MARKET:
        return "bar" if event.bar is not None else "snapshot"
    if event.kind is EventKind.EXECUTION:
        assert event.execution is not None  # envelope invariant
        return str(event.execution.event_type.value)
    assert event.timer is not None  # envelope invariant
    return str(event.timer.name)


def write_event_archive(
    events: Iterable[Event], path: str | Path, *, run_id: str
) -> Path:
    """Persist the full event journal as a schema-versioned parquet file.

    ``events`` is the dispatched journal in dispatch order (typically
    ``bus.journal``); row order is preserved so ``seq`` and row index
    identify each event exactly. Parent directories are created.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    journal = list(events)
    columns: dict[str, list[object]] = {name: [] for name in _EVENT_COLUMN_NAMES}
    for seq, event in enumerate(journal, start=1):
        fill = event.execution.fill if event.execution is not None else None
        columns["schema_version"].append(EVENTS_SCHEMA_VERSION)
        columns["run_id"].append(run_id)
        columns["seq"].append(seq)
        columns["ts"].append(event.ts.isoformat())
        columns["kind"].append(event.kind.value)
        columns["event_type"].append(_event_type(event))
        columns["symbol"].append(
            event.bar.symbol
            if event.bar is not None
            else event.snapshot.symbol
            if event.snapshot is not None
            else fill.symbol
            if fill is not None
            else None
        )
        columns["side"].append(fill.side.value if fill is not None else None)
        columns["price"].append(
            event.bar.close
            if event.bar is not None
            else event.snapshot.last_price
            if event.snapshot is not None
            else fill.price
            if fill is not None
            else None
        )
        columns["quantity"].append(fill.quantity if fill is not None else None)
        columns["commission"].append(fill.commission if fill is not None else None)
        columns["stamp_duty"].append(fill.stamp_duty if fill is not None else None)
        columns["transfer_fee"].append(fill.transfer_fee if fill is not None else None)
        columns["payload"].append(canonical_event_json(event))

    arrow_types = {"int64": pa.int64(), "string": pa.string(), "float64": pa.float64()}
    schema = pa.schema(
        pa.field(name, arrow_types[arrow_type], nullable=nullable)
        for name, arrow_type, nullable in _EVENT_COLUMNS
    )
    table = pa.table(
        {field.name: pa.array(columns[field.name], type=field.type) for field in schema},
        schema=schema,
    )
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target, compression="snappy")
    return target


def read_event_archive(path: str | Path) -> tuple[Event, ...]:
    """Load an event archive back into validated kernel events.

    Reconstruction replays the ``payload`` canonical JSON, so the returned
    events compare equal (field by field) to the journal that was written.
    Fails loudly on unknown schema versions or malformed payloads.
    """
    import pyarrow.parquet as pq

    table = pq.read_table(Path(path))
    names = set(table.schema.names)
    for required in ("schema_version", "payload"):
        if required not in names:
            raise PulsarCoreError(
                f"event archive misses the '{required}' column (schema: {sorted(names)})"
            )
    versions = {
        version
        for version in table.column("schema_version").to_pylist()
        if version != EVENTS_SCHEMA_VERSION
    }
    if versions:
        raise PulsarCoreError(
            f"event archive schema version(s) {sorted(versions)} not supported; "
            f"this reader implements {EVENTS_SCHEMA_VERSION}"
        )
    events: list[Event] = []
    for payload in table.column("payload").to_pylist():
        events.append(Event.model_validate_json(payload))
    return tuple(events)


def write_run_artifacts(
    result: RunResult,
    *,
    events: Iterable[Event],
    initial_cash: float,
    directory: str | Path,
) -> RunArtifacts:
    """Persist the complete run directory: manifest, archive, metrics report.

    ``events`` is the run's dispatched journal (``session.bus.journal``) —
    the same stream ``result`` was derived from; ``initial_cash`` is the
    run's starting equity. The three files land side by side and the
    report's ``journal_digest`` ties it to ``result``.
    """
    journal = list(events)
    out = Path(directory)
    manifest_path = result.manifest.write(out / MANIFEST_FILENAME)
    events_path = write_event_archive(
        journal, out / EVENTS_FILENAME, run_id=result.run_id
    )
    report = build_metrics_report(
        journal,
        initial_cash=initial_cash,
        run_id=result.run_id,
        journal_digest=result.journal_digest,
    )
    metrics_path = report.write(out / METRICS_FILENAME)
    return RunArtifacts(
        run_id=result.run_id,
        directory=str(out),
        manifest_path=str(manifest_path),
        events_path=str(events_path),
        metrics_path=str(metrics_path),
        report=report,
    )

"""RunManifest: deterministic derivation, persistence and watermarks."""

from __future__ import annotations

import json
import math
from datetime import date, datetime

import pytest
from pulsar_contracts import Bar, Freq

from pulsar_core import (
    RunManifest,
    bars_watermark,
    code_version,
    load_manifest,
)

BASE_INPUTS = {
    "mode": "research",
    "seed": 42,
    "config": {"portfolio": {"top_n": 30}, "start": date(2026, 6, 1)},
    "code_version": "abc123commit",
    "data_watermarks": {"bars/1d/600000": "2026-06-30T00:00:00+08:00"},
}


def build(**overrides) -> RunManifest:
    inputs = dict(BASE_INPUTS)
    inputs.update(overrides)
    return RunManifest.build(**inputs)


def test_same_inputs_rebuild_the_same_manifest() -> None:
    first = build()
    second = build()
    assert first == second
    assert first.run_id == second.run_id
    assert first.to_json() == second.to_json()


def test_run_id_reacts_to_every_identity_field() -> None:
    baseline = build().run_id
    assert build(seed=43).run_id != baseline
    assert build(mode="paper").run_id != baseline
    assert build(code_version="def456").run_id != baseline
    assert (
        build(data_watermarks={"bars/1d/600000": "2026-07-01T00:00:00+08:00"}).run_id
        != baseline
    )
    assert build(config={"portfolio": {"top_n": 31}}).run_id != baseline


def test_unknown_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown run mode"):
        build(mode="imagination")


def test_non_json_native_config_is_rejected_loudly() -> None:
    with pytest.raises(ValueError, match="non-JSON-native"):
        build(config={"bad": object()})


def test_nan_config_is_rejected_for_determinism() -> None:
    with pytest.raises(ValueError):
        build(config={"bad": math.nan})


def test_identity_json_excludes_run_id_and_is_stable() -> None:
    assert build().identity_json() == build().identity_json()
    identity = json.loads(build().identity_json())
    assert "run_id" not in identity
    assert identity["config"]["start"] == "2026-06-01"  # dates encode as ISO


def test_write_and_load_round_trip(tmp_path) -> None:
    manifest = build()
    path = manifest.write(tmp_path / "nested" / "run_manifest.json")
    loaded = load_manifest(path)
    assert loaded == manifest
    assert loaded.to_json() == manifest.to_json()
    assert loaded.matches_code("abc123commit")
    assert not loaded.matches_code("other")


def test_load_rejects_malformed_documents(tmp_path) -> None:
    path = tmp_path / "broken.json"
    path.write_text('{"run_id": "short"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_manifest(path)


def test_watermark_keys_sorted_and_latest_ts_wins() -> None:
    bars = [
        _bar("600000", "2026-06-01"),
        _bar("000001", "2026-06-01"),
        _bar("600000", "2026-06-03"),
        _bar("600000", "2026-06-02"),
    ]
    marks = bars_watermark(bars)
    assert list(marks) == ["bars/1d/000001", "bars/1d/600000"]
    assert marks["bars/1d/600000"] == "2026-06-03T00:00:00+08:00"


def test_code_version_env_override(monkeypatch) -> None:
    monkeypatch.setenv("PULSAR_CODE_VERSION", "deadbeef")
    assert code_version() == "deadbeef"


def _bar(symbol: str, day: str) -> Bar:
    return Bar(
        symbol=symbol,
        ts=datetime.fromisoformat(f"{day}T00:00:00+08:00"),
        freq=Freq.DAILY,
        open=10.0,
        high=11.0,
        low=9.5,
        close=10.5,
        volume=1000.0,
        amount=10500.0,
    )

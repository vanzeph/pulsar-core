"""Training-sample assembly from a :class:`FactorHistoryView`.

ML modelers consume the *preprocessed* factor panel the pipeline already
serves: at every evaluation date ``d`` the features are the symbol's
preprocessed factor row and the label is its ``horizon``-bar forward
return, z-scored across the symbols of that date (a cross-sectional
target keeps the model from learning market-level drift). Everything is
assembled through the history view — which the engine clips at the
requesting cross-section's date — so training sees exactly the past, no
look-ahead, the same discipline the IC-weighted modeler follows.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Sequence

from ..errors import PulsarCoreError

__all__ = [
    "FlatSample",
    "SequenceSample",
    "evaluation_dates",
    "flat_samples",
    "sequence_samples",
    "zscore_values",
]

#: One MLP training row: feature vector + label.
FlatSample = tuple[list[float], float]

#: One LSTM training row: ``window``-long feature sequence + label.
SequenceSample = tuple[list[list[float]], float]


def evaluation_dates(
    common_dates: Sequence[date], *, lookback: int, horizon: int
) -> list[date]:
    """The trailing ``lookback`` dates that have ``horizon`` bars after them.

    Mirrors the IC-weighted modeler's candidate window: the last
    ``lookback + horizon`` common dates, minus the final ``horizon`` —
    each remaining date provably has its forward window inside history.
    """
    if lookback < 1 or horizon < 1:
        raise PulsarCoreError("lookback and horizon must be >= 1")
    trailing = list(common_dates[-(lookback + horizon) :])
    return trailing[:lookback]


def _panel(
    history: Any,
    *,
    factors: Sequence[str],
    dates: Sequence[date],
) -> dict[date, dict[str, dict[str, "float | None"]]]:
    """Preprocessed columns per date, fetched once per (date, factor)."""
    panel: dict[date, dict[str, dict[str, "float | None"]]] = {}
    for day in dates:
        columns: dict[str, dict[str, "float | None"]] = {}
        for factor in factors:
            columns[factor] = dict(history.preprocessed_values(factor, day))
        panel[day] = columns
    return panel


def _complete_features(
    columns: dict[str, dict[str, "float | None"]],
    factors: Sequence[str],
) -> dict[str, list[float]]:
    """Feature rows for symbols with a non-missing value in every factor."""
    symbols: set[str] = set()
    for column in columns.values():
        symbols.update(column)
    rows: dict[str, list[float]] = {}
    for symbol in sorted(symbols):
        features: list[float] = []
        complete = True
        for factor in factors:
            value = columns[factor].get(symbol)
            if value is None:
                complete = False
                break
            features.append(float(value))
        if complete:
            rows[symbol] = features
    return rows


def zscore_values(values: list[float]) -> list[float]:
    """Zero-mean / unit-population-std mapping; a constant list maps to 0.0."""
    if not values:
        return []
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    if variance <= 0.0:
        return [0.0] * len(values)
    std = variance**0.5
    return [(value - mean) / std for value in values]


def _labels(
    history: Any, day: date, horizon: int, rows: dict[str, list[float]]
) -> dict[str, float]:
    """Forward returns for ``day``'s symbols, z-scored across the date."""
    returns = dict(history.forward_returns(day, horizon))
    shared = sorted(set(rows) & set(returns))
    if not shared:
        return {}
    scored = zscore_values([returns[symbol] for symbol in shared])
    return dict(zip(shared, scored))


def flat_samples(
    history: Any,
    *,
    factors: Sequence[str],
    lookback: int,
    horizon: int,
) -> tuple[list[FlatSample], tuple[date, date] | None]:
    """MLP training rows over the trailing evaluation dates.

    Returns ``(samples, window)`` where ``window`` is the first/last
    evaluation date actually used (``None`` when nothing was usable —
    young histories, handled by the scorer's documented fallback).
    """
    dates = evaluation_dates(history.common_dates(), lookback=lookback, horizon=horizon)
    panel = _panel(history, factors=factors, dates=dates)
    samples: list[FlatSample] = []
    used: list[date] = []
    for day in dates:
        rows = _complete_features(panel[day], factors)
        if not rows:
            continue
        labels = _labels(history, day, horizon, rows)
        if not labels:
            continue
        used.append(day)
        for symbol in sorted(labels):
            samples.append((rows[symbol], labels[symbol]))
    window = (used[0], used[-1]) if used else None
    return samples, window


def sequence_samples(
    history: Any,
    *,
    factors: Sequence[str],
    lookback: int,
    horizon: int,
    window: int,
) -> tuple[list[SequenceSample], tuple[date, date] | None]:
    """LSTM training rows: rolling ``window``-long feature sequences.

    For every evaluation date ``d``, a symbol contributes a sample when it
    has complete feature rows on the ``window`` consecutive panel dates
    ending at ``d`` — missing values drop the sample, never imputed.
    """
    if window < 1:
        raise PulsarCoreError("window must be >= 1")
    common = list(history.common_dates())
    dates = evaluation_dates(common, lookback=lookback, horizon=horizon)
    span = common[-(lookback + horizon + window - 1) :] if dates else []
    panel = _panel(history, factors=factors, dates=span)
    complete_by_date = {day: _complete_features(panel[day], factors) for day in span}
    samples: list[SequenceSample] = []
    used: list[date] = []
    for day in dates:
        end = span.index(day) + 1
        window_dates = span[max(0, end - window) : end]
        if len(window_dates) < window:
            continue
        rows = complete_by_date[day]
        labels = _labels(history, day, horizon, rows)
        if not labels:
            continue
        # symbols complete across the whole window (superset of today's rows)
        candidates: set[str] = set(rows)
        for past in window_dates[:-1]:
            candidates &= set(complete_by_date[past])
        if not candidates:
            continue
        used.append(day)
        for symbol in sorted(set(labels) & candidates):
            sequence = [complete_by_date[past][symbol] for past in window_dates]
            samples.append((sequence, labels[symbol]))
    window_range = (used[0], used[-1]) if used else None
    return samples, window_range

"""Cross-sectional preprocessing: winsorize, zscore, missing handling.

Design (core-engine design): factor readings are preprocessed
cross-sectionally — each step transforms one factor's column
``{symbol: value}`` at one decision date — and steps compose in the order
the experiment config lists them::

    preprocess = ["winsorize", "zscore"]

Semantics:

* **winsorize** clips present values to the ``[quantile, 1 - quantile]``
  cross-sectional quantiles (linear interpolation, deterministic);
* **zscore** standardizes to zero mean / unit population std; a constant
  column maps to all ``0.0`` (no division by zero);
* **fillna** replaces ``None`` with the cross-sectional median (default)
  or zero — configure it explicitly if missing values should be filled;
* steps are order-sensitive and exactly the composition in the config
  runs — no hidden defaults beyond an empty pipeline.

``None`` (insufficient factor history) passes through winsorize/zscore
untouched; whether it is filled depends solely on a configured
``fillna``. Symbols whose values are still ``None`` at model time are
dropped from scoring by the modelers (documented in
:mod:`pulsar_core.modelers`).
"""

from __future__ import annotations

from typing import Any, Mapping

from .errors import PulsarCoreError
from .params import BoundParams, Params
from .registry import Registry

__all__ = [
    "Column",
    "PreprocessStep",
    "WinsorizeStep",
    "ZscoreStep",
    "FillNaStep",
    "PREPROCESS_REGISTRY",
    "build_step",
]

#: One factor's cross-section at one decision date; ``None`` = missing.
Column = dict[str, "float | None"]


class PreprocessStep:
    """Base class of one cross-sectional transform over a factor column.

    Subclasses declare their parameters with the strategy-parameter
    machinery (``params = Params(...)``) so TOML overrides are validated
    exactly like strategy parameters: unknown names and type mismatches
    fail loudly. The bound set is stored as :attr:`bound`; subclasses
    narrow their declared scalars into typed attributes in ``__init__``.
    """

    name: str = "preprocess_step"
    params: Params = Params()

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        self.bound: BoundParams = type(self).params.bind(overrides)

    def apply(self, column: Mapping[str, "float | None"]) -> Column:
        raise NotImplementedError


def _number(bound: BoundParams, key: str, step: str) -> float:
    raw = bound[key]
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise PulsarCoreError(f"{step} {key} must be a number, got {raw!r}")
    return float(raw)


class WinsorizeStep(PreprocessStep):
    """Clip present values to cross-sectional ``[q, 1-q]`` quantiles."""

    name = "winsorize"
    params = Params(quantile=0.025)

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__(overrides)
        self.quantile = _number(self.bound, "quantile", "winsorize")
        if not 0.0 < self.quantile < 0.5:
            raise PulsarCoreError(
                f"winsorize quantile must lie in (0, 0.5), got {self.quantile}"
            )

    def apply(self, column: Mapping[str, "float | None"]) -> Column:
        present = sorted(value for value in column.values() if value is not None)
        if not present:
            return dict(column)
        lower = _quantile(present, self.quantile)
        upper = _quantile(present, 1.0 - self.quantile)
        return {
            symbol: None if value is None else min(max(value, lower), upper)
            for symbol, value in column.items()
        }


class ZscoreStep(PreprocessStep):
    """Standardize present values to zero mean / unit population std."""

    name = "zscore"
    params = Params()

    def apply(self, column: Mapping[str, "float | None"]) -> Column:
        present = [value for value in column.values() if value is not None]
        if not present:
            return dict(column)
        count = len(present)
        mean = sum(present) / count
        variance = sum((value - mean) ** 2 for value in present) / count
        std = float(variance) ** 0.5
        if std <= 1e-15:
            # constant column: every reading is exactly average
            return {
                symbol: None if value is None else 0.0
                for symbol, value in column.items()
            }
        return {
            symbol: None if value is None else (value - mean) / std
            for symbol, value in column.items()
        }


class FillNaStep(PreprocessStep):
    """Replace ``None`` with the cross-sectional median (or zero)."""

    name = "fillna"
    params = Params(method="median")

    def __init__(self, overrides: Mapping[str, Any] | None = None) -> None:
        super().__init__(overrides)
        raw = self.bound["method"]
        if not isinstance(raw, str) or raw not in ("median", "zero"):
            raise PulsarCoreError(
                f"fillna method must be 'median' or 'zero', got {raw!r}"
            )
        self.method = raw

    def apply(self, column: Mapping[str, "float | None"]) -> Column:
        if self.method == "zero":
            fill = 0.0
        else:
            present = sorted(value for value in column.values() if value is not None)
            if not present:
                raise PulsarCoreError(
                    "fillna(median) cannot fill an all-missing factor column; "
                    "fix the data or drop the factor"
                )
            fill = _quantile(present, 0.5)
        return {
            symbol: fill if value is None else value
            for symbol, value in column.items()
        }


#: The default registry of preprocess steps an experiment can compose.
PREPROCESS_REGISTRY: Registry[type[PreprocessStep]] = Registry(
    kind="preprocess step", name_of=lambda step: step.name
)
PREPROCESS_REGISTRY.register(WinsorizeStep)
PREPROCESS_REGISTRY.register(ZscoreStep)
PREPROCESS_REGISTRY.register(FillNaStep)


def build_step(name: str, overrides: Mapping[str, Any] | None = None) -> PreprocessStep:
    """Instantiate the registered step ``name`` with validated overrides."""
    step_type = PREPROCESS_REGISTRY.resolve(name)
    return step_type(overrides)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolation quantile over pre-sorted values (numpy style)."""
    if not sorted_values:
        raise PulsarCoreError("quantile of an empty sample is undefined")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    fraction = position - lower_index
    return float(
        sorted_values[lower_index]
        + fraction * (sorted_values[upper_index] - sorted_values[lower_index])
    )

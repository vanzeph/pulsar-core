"""The factor library: registered, bar-sequence factor definitions.

Layered-configuration design (core-engine design, 因子库与实验配置): factors
are *code plus registration*. A :class:`FactorDefinition` bundles the
stable registry name, human metadata, the cross-sectional ``direction``
and the ``compute`` function over one symbol's bar sequence; experiments
then select factor subsets by name from pure TOML.

Computation contract:

* a factor sees exactly one symbol's bars, oldest first, *including* the
  decision-date bar (values at date ``t`` use bars with ``ts <= t``);
* ``compute`` returns ``None`` when history is insufficient
  (``len(bars) < min_bars``) — insufficient history is *missing data*, not
  an error; the preprocess pipeline decides how missing values are
  handled (see :mod:`pulsar_core.preprocess`);
* the raw value is oriented by ``direction`` downstream: the oriented
  value ``raw * direction`` always means "higher = more preferred", so
  modelers and portfolio construction never need per-factor sign logic.

Built-ins are bar-only by construction: the ``Bar`` contract carries
prices/volume/turnover and no fundamentals, so value-class factors that
need earnings (a real ``ep_ttm``) require a fundamentals channel that does
not exist yet in the ports; they will register through this same surface
once such a port is designed. The built-ins below cover the momentum,
volatility, reversal and intraday-range classes:

==================  =====================================================
name                oriented definition (higher = more preferred)
==================  =====================================================
``momentum_20``     trailing 20-bar close-to-close return
``momentum_60``     trailing 60-bar close-to-close return
``volatility_20``   minus the population std of the last 20 one-bar returns
``reversal_5``      minus the trailing 5-bar return (short-term reversal)
``range_20``        minus the mean intraday high-low range fraction
==================  =====================================================
"""

from __future__ import annotations

from math import sqrt
from typing import Callable, Sequence

from pulsar_contracts import Bar

from .registry import Registry

__all__ = [
    "FactorDefinition",
    "FACTOR_REGISTRY",
    "factor_value",
    "register_factor",
    "momentum_factor",
    "volatility_factor",
    "reversal_factor",
    "range_factor",
]

#: What one factor's compute function receives and returns: the symbol's
#: bars (oldest first, current bar included) and the raw reading, or
#: ``None`` when there is not enough history.
FactorCompute = Callable[[Sequence[Bar]], "float | None"]


class FactorDefinition:
    """One registered factor: name, metadata, orientation, compute.

    ``direction`` is ``+1`` when a higher raw reading is better and ``-1``
    when a lower one is; the pipeline multiplies it in right after
    :meth:`compute` so every downstream consumer sees "higher = better".
    """

    def __init__(
        self,
        *,
        name: str,
        label: str,
        compute: FactorCompute,
        direction: int = 1,
        min_bars: int,
    ) -> None:
        if not name:
            raise ValueError("factor name must be non-empty")
        if direction not in (1, -1):
            raise ValueError(f"direction must be +1 or -1, got {direction}")
        if min_bars < 1:
            raise ValueError(f"min_bars must be >= 1, got {min_bars}")
        self.name = name
        self.label = label
        self.compute = compute
        self.direction = direction
        self.min_bars = min_bars

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"FactorDefinition(name={self.name!r}, direction={self.direction:+d}, "
            f"min_bars={self.min_bars})"
        )


#: The default process-wide factor registry experiments resolve names in.
FACTOR_REGISTRY: Registry[FactorDefinition] = Registry(
    kind="factor", name_of=lambda factor: factor.name
)


def register_factor(factor: FactorDefinition) -> None:
    """Convenience wrapper around :meth:`FACTOR_REGISTRY.register`."""
    FACTOR_REGISTRY.register(factor)


def factor_value(factor: FactorDefinition | str, bars: Sequence[Bar]) -> float | None:
    """Compute one factor's *oriented* value over ``bars``.

    Resolves a registered name when a string is given, runs the compute
    function and multiplies by the factor's direction, so the returned
    number always means "higher = more preferred" (``None`` when the
    history is shorter than the factor's ``min_bars``).
    """
    definition = FACTOR_REGISTRY.resolve(factor) if isinstance(factor, str) else factor
    raw = definition.compute(bars)
    if raw is None:
        return None
    return raw * definition.direction


# -- built-in families -----------------------------------------------------------


def _close_series(bars: Sequence[Bar]) -> list[float]:
    return [float(bar.close) for bar in bars]


def _momentum_compute(window: int) -> FactorCompute:
    def compute(bars: Sequence[Bar]) -> float | None:
        if len(bars) < window + 1:
            return None
        closes = _close_series(bars)
        return float(closes[-1] / closes[-window - 1] - 1.0)

    return compute


def _volatility_compute(window: int) -> FactorCompute:
    def compute(bars: Sequence[Bar]) -> float | None:
        if len(bars) < window + 1:
            return None
        closes = _close_series(bars)
        returns = [
            closes[i] / closes[i - 1] - 1.0
            for i in range(len(closes) - window, len(closes))
        ]
        mean = float(sum(returns)) / window
        variance = float(sum((value - mean) ** 2 for value in returns)) / window
        return sqrt(variance)

    return compute


def _reversal_compute(window: int) -> FactorCompute:
    def compute(bars: Sequence[Bar]) -> float | None:
        if len(bars) < window + 1:
            return None
        closes = _close_series(bars)
        return float(closes[-1] / closes[-window - 1] - 1.0)

    return compute


def _range_compute(window: int) -> FactorCompute:
    def compute(bars: Sequence[Bar]) -> float | None:
        if len(bars) < window:
            return None
        window_bars = bars[-window:]
        fractions = [
            float((bar.high - bar.low) / bar.close) for bar in window_bars
        ]
        return sum(fractions) / window

    return compute


def momentum_factor(window: int = 20) -> FactorDefinition:
    """Trailing ``window``-bar close-to-close return (higher is better)."""
    return FactorDefinition(
        name=f"momentum_{window}",
        label=f"{window}-bar close-to-close momentum",
        compute=_momentum_compute(window),
        direction=1,
        min_bars=window + 1,
    )


def volatility_factor(window: int = 20) -> FactorDefinition:
    """Population std of the last ``window`` one-bar returns (lower is better)."""
    return FactorDefinition(
        name=f"volatility_{window}",
        label=f"{window}-bar daily-return volatility (population std)",
        compute=_volatility_compute(window),
        direction=-1,
        min_bars=window + 1,
    )


def reversal_factor(window: int = 5) -> FactorDefinition:
    """Short-term reversal: minus the trailing ``window``-bar return."""
    return FactorDefinition(
        name=f"reversal_{window}",
        label=f"{window}-bar short-term reversal",
        compute=_reversal_compute(window),
        direction=-1,
        min_bars=window + 1,
    )


def range_factor(window: int = 20) -> FactorDefinition:
    """Mean intraday high-low range fraction (lower is better)."""
    return FactorDefinition(
        name=f"range_{window}",
        label=f"{window}-bar mean intraday high-low range fraction",
        compute=_range_compute(window),
        direction=-1,
        min_bars=window,
    )


FACTOR_REGISTRY.register(momentum_factor(20))
FACTOR_REGISTRY.register(momentum_factor(60))
FACTOR_REGISTRY.register(volatility_factor(20))
FACTOR_REGISTRY.register(reversal_factor(5))
FACTOR_REGISTRY.register(range_factor(20))

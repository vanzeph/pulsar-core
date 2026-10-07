"""Modelers (打分模型): combine preprocessed factor columns into scores.

A modeler is *code plus registration* (core-engine design): a
:class:`ModelScorer` receives one decision date's
:class:`CrossSection` — the preprocessed, orientation-aligned factor
columns — plus a :class:`FactorHistoryView` for modelers that estimate
their own weights from history, and returns one score per symbol.
Higher score always means "more preferred"; the portfolio constructor
turns scores into target weights.

Symbols with a ``None`` (still-missing) value in *any* configured factor
are dropped from scoring: a modeler never guesses around holes. Combine
the ``fillna`` preprocess step with these modelers when missing values
should be imputed instead.

Built-ins:

* ``equal_weight`` — mean of the factor columns;
* ``linear_score`` — fixed linear weights from the experiment config
  (``params.weights``, one weight per configured factor);
* ``ic_weighted`` (design-doc alias ``linear_ic``) — factor weights from
  the trailing mean Spearman rank IC between each factor and forward
  returns, estimated over ``lookback`` past evaluation points with
  ``horizon``-bar forward returns; non-positive mean ICs get zero
  weight, and the weights fall back to equal when fewer than
  ``min_points`` evaluation points exist (young histories).
"""

from __future__ import annotations

from datetime import date
from typing import Any, Callable, Mapping, Sequence
import math

from .errors import PulsarCoreError
from .registry import Registry

__all__ = [
    "CrossSection",
    "FactorHistoryView",
    "ModelScorer",
    "ModelDefinition",
    "MODEL_REGISTRY",
    "register_model",
    "EqualWeightScorer",
    "LinearScoreScorer",
    "IcWeightedScorer",
    "spearman_ic",
]

#: Minimum number of symbols for one IC evaluation point to count.
MIN_IC_SYMBOLS = 3


class CrossSection:
    """One decision date's preprocessed factor columns.

    ``values`` maps factor name → ``{symbol: value}``; values are the
    oriented (higher = better), preprocessed readings and may be
    ``None`` (missing).
    """

    def __init__(self, *, as_of: date, values: Mapping[str, Mapping[str, "float | None"]]) -> None:
        self.as_of = as_of
        self.values: dict[str, dict[str, "float | None"]] = {
            factor: dict(column) for factor, column in values.items()
        }

    @property
    def factors(self) -> tuple[str, ...]:
        """Factor names in sorted order (deterministic iteration)."""
        return tuple(sorted(self.values))

    def complete_rows(self) -> dict[str, dict[str, float]]:
        """Rows (symbols) with a non-missing value in every factor."""
        symbols: set[str] = set()
        for column in self.values.values():
            symbols.update(column)
        rows: dict[str, dict[str, float]] = {}
        for symbol in sorted(symbols):
            row: dict[str, float] = {}
            complete = True
            for factor in self.factors:
                value = self.values[factor].get(symbol)
                if value is None:
                    complete = False
                    break
                row[factor] = value
            if complete:
                rows[symbol] = row
        return rows


class FactorHistoryView:
    """Read-only historical factor/return access for weight estimation.

    Implemented by the factor pipeline over the run's bar history; all
    methods are clipped to the requesting cross-section's date so no
    look-ahead can leak into weight estimation.
    """

    def factor_names(self) -> tuple[str, ...]:
        """The factor names this view serves, sorted."""
        raise NotImplementedError

    def common_dates(self) -> tuple[date, ...]:
        """Sorted trading dates shared by the run's history window."""
        raise NotImplementedError

    def factor_values(self, factor: str, as_of: date) -> Mapping[str, "float | None"]:
        """The factor's oriented raw values computed at ``as_of``."""
        raise NotImplementedError

    def preprocessed_values(
        self, factor: str, as_of: date
    ) -> Mapping[str, "float | None"]:
        """The factor's values as the configured preprocess chain leaves them.

        The ML modelers (``mlp_torch`` / ``lstm_torch``) train and score on
        the same preprocessed panel the cross-sections carry; views that
        serve no preprocessing fall back to the raw oriented values.
        """
        return self.factor_values(factor, as_of)

    def forward_returns(self, from_date: date, horizon: int) -> Mapping[str, float]:
        """``horizon``-bar forward close-to-close returns from ``from_date``."""
        raise NotImplementedError


class ModelScorer:
    """Base class of one registered scoring model."""

    #: Extra bars of history this model needs beyond the factors' own
    #: ``min_bars`` (IC estimation needs a lookback of observations).
    warmup_bars: int = 0

    name: str = "model"

    def score(
        self, section: CrossSection, history: FactorHistoryView
    ) -> dict[str, float]:
        raise NotImplementedError


class ModelDefinition:
    """Registry entry: how to build one scorer from config parameters."""

    def __init__(
        self,
        *,
        name: str,
        create: Callable[[Mapping[str, Any], Sequence[str]], ModelScorer],
        description: str = "",
    ) -> None:
        self.name = name
        self.create = create
        self.description = description


# -- equal weight ----------------------------------------------------------------


class EqualWeightScorer(ModelScorer):
    """Mean of the factor columns — the baseline model."""

    name = "equal_weight"

    def score(
        self, section: CrossSection, history: FactorHistoryView
    ) -> dict[str, float]:
        rows = section.complete_rows()
        if not rows:
            return {}
        count = len(section.factors)
        return {
            symbol: sum(row[factor] for factor in section.factors) / count
            for symbol, row in rows.items()
        }


def _create_equal_weight(
    params: Mapping[str, Any], factor_names: Sequence[str]
) -> ModelScorer:
    if params:
        raise PulsarCoreError(
            f"model 'equal_weight' takes no parameters, got {sorted(params)}"
        )
    return EqualWeightScorer()


# -- linear score ----------------------------------------------------------------


class LinearScoreScorer(ModelScorer):
    """Fixed linear combination ``sum(w_f * x_f)`` from the experiment config."""

    name = "linear_score"

    def __init__(self, weights: Mapping[str, float]) -> None:
        self.weights: dict[str, float] = dict(weights)

    def score(
        self, section: CrossSection, history: FactorHistoryView
    ) -> dict[str, float]:
        rows = section.complete_rows()
        return {
            symbol: sum(self.weights[factor] * row[factor] for factor in section.factors)
            for symbol, row in rows.items()
        }


def _create_linear_score(
    params: Mapping[str, Any], factor_names: Sequence[str]
) -> ModelScorer:
    configured = set(factor_names)
    unknown_params = sorted(set(params) - {"weights"})
    if unknown_params:
        raise PulsarCoreError(
            f"model 'linear_score' takes only 'weights', got {unknown_params}"
        )
    raw_weights = params.get("weights")
    if not isinstance(raw_weights, Mapping):
        raise PulsarCoreError(
            "model 'linear_score' requires params.weights = { factor = weight }"
        )
    weights: dict[str, float] = {}
    for factor, weight in raw_weights.items():
        if not isinstance(factor, str):
            raise PulsarCoreError(f"weights keys must be factor names, got {factor!r}")
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise PulsarCoreError(
                f"weight for {factor!r} must be a number, got {weight!r}"
            )
        weights[factor] = float(weight)
    missing = sorted(configured - set(weights))
    if missing:
        raise PulsarCoreError(
            f"model 'linear_score' needs a weight for every configured factor; "
            f"missing: {missing}"
        )
    extra = sorted(set(weights) - configured)
    if extra:
        raise PulsarCoreError(
            f"model 'linear_score' weights name unconfigured factors: {extra}"
        )
    return LinearScoreScorer(weights)


# -- IC weighted -----------------------------------------------------------------


class IcWeightedScorer(ModelScorer):
    """Factor weights from trailing mean Spearman rank IC vs forward returns.

    At each decision date the scorer estimates, over the last
    ``lookback`` evaluation points, each factor's mean rank IC against
    ``horizon``-bar forward returns; weights are the positive parts
    renormalized to sum to one. Fewer than ``min_points`` usable points
    (young history) falls back to equal weights — deterministic and
    documented, never an error mid-run.
    """

    name = "ic_weighted"

    def __init__(self, *, lookback: int = 60, horizon: int = 5, min_points: int = 10) -> None:
        if lookback < 1 or horizon < 1 or min_points < 1:
            raise PulsarCoreError(
                "ic_weighted parameters lookback/horizon/min_points must be >= 1"
            )
        self.lookback = lookback
        self.horizon = horizon
        self.min_points = min_points
        # lookback observations + horizon forward bars + one slack bar
        self.warmup_bars = lookback + horizon + 1

    def factor_weights(self, history: FactorHistoryView) -> dict[str, float]:
        """Estimated weights per factor at the cross-section's date.

        Exposed for analysis surfaces; :meth:`score` uses the same
        estimate, so the applied weights are always inspectable.
        """
        factors = list(history.factor_names())
        section_dates = history.common_dates()
        # evaluation dates must leave room for `horizon` forward bars
        candidates = section_dates[-(self.lookback + self.horizon) :]
        observations: dict[str, list[float]] = {}
        for evaluation_date in candidates:
            returns = history.forward_returns(evaluation_date, self.horizon)
            if len(returns) < MIN_IC_SYMBOLS:
                continue
            for factor in factors:
                values = history.factor_values(factor, evaluation_date)
                ic = spearman_ic(values, returns)
                if ic is not None:
                    observations.setdefault(factor, []).append(ic)
        every_factor_observed = len(observations) == len(factors) and factors != []
        points = min((len(values) for values in observations.values()), default=0)
        if not every_factor_observed or points < self.min_points:
            return _equal_weights(factors)
        positives = {
            factor: max(sum(values) / len(values), 0.0)
            for factor, values in observations.items()
        }
        total = sum(positives.values())
        if total <= 0.0:
            return _equal_weights(factors)
        return {factor: value / total for factor, value in positives.items()}

    def score(
        self, section: CrossSection, history: FactorHistoryView
    ) -> dict[str, float]:
        rows = section.complete_rows()
        if not rows:
            return {}
        weights = self.factor_weights(history)
        factors = section.factors
        effective = {factor: weights.get(factor, 0.0) for factor in factors}
        return {
            symbol: sum(effective[factor] * row[factor] for factor in factors)
            for symbol, row in rows.items()
        }


def _equal_weights(factors: Sequence[str]) -> dict[str, float]:
    count = len(factors)
    return {factor: 1.0 / count for factor in factors} if count else {}


def _create_ic_weighted(
    params: Mapping[str, Any], factor_names: Sequence[str]
) -> ModelScorer:
    allowed = {"lookback", "horizon", "min_points"}
    unknown = sorted(set(params) - allowed)
    if unknown:
        raise PulsarCoreError(
            f"model 'ic_weighted' parameters are {sorted(allowed)}, got {unknown}"
        )
    resolved: dict[str, int] = {
        "lookback": 60,
        "horizon": 5,
        "min_points": 10,
    }
    for key, value in params.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise PulsarCoreError(
                f"ic_weighted parameter {key!r} must be an integer, got {value!r}"
            )
        resolved[key] = value
    return IcWeightedScorer(**resolved)


# -- statistics helpers -------------------------------------------------------------


def _average_ranks(values: Sequence[float]) -> list[float]:
    """Ranks (1-based, average for ties) of ``values``."""
    count = len(values)
    order = sorted(range(count), key=lambda index: values[index])
    ranks = [0.0] * count
    start = 0
    while start < count:
        end = start
        while end + 1 < count and values[order[end + 1]] == values[order[start]]:
            end += 1
        average = (start + end) / 2.0 + 1.0
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def _pearson(x: Sequence[float], y: Sequence[float]) -> float | None:
    """Pearson correlation; ``None`` when either side has zero variance."""
    count = len(x)
    if count != len(y) or count == 0:
        return None
    mean_x = sum(x) / count
    mean_y = sum(y) / count
    var_x = sum((value - mean_x) ** 2 for value in x) / count
    var_y = sum((value - mean_y) ** 2 for value in y) / count
    if var_x <= 0.0 or var_y <= 0.0:
        return None
    covariance = sum(
        (x[i] - mean_x) * (y[i] - mean_y) for i in range(count)
    ) / count
    return float(covariance / (math.sqrt(var_x) * math.sqrt(var_y)))


def spearman_ic(x: Mapping[str, "float | None"], y: Mapping[str, float]) -> float | None:
    """Spearman rank IC of two symbol-keyed series over shared symbols.

    Any missing (``None``) value on the factor side disqualifies the
    point (callers decide how to treat missing data); fewer than two
    shared symbols carries no information either (``None``).
    """
    symbols = sorted(set(x) & set(y))
    if len(symbols) < 2:
        return None
    xs: list[float] = []
    ys: list[float] = []
    for symbol in symbols:
        value = x[symbol]
        if value is None:
            return None
        xs.append(value)
        ys.append(y[symbol])
    return _pearson(_average_ranks(xs), _average_ranks(ys))


#: The default modeler registry experiments resolve ``[model] type`` in.
MODEL_REGISTRY: Registry[ModelDefinition] = Registry(
    kind="model", name_of=lambda model: model.name
)


def register_model(definition: ModelDefinition) -> None:
    """Register ``definition`` in :data:`MODEL_REGISTRY`.

    The external registration hook of the 自定义代码组装 contract: custom
    modelers materialized by ``pulsar-app`` from the unified store register
    through this surface (mirroring
    :func:`~pulsar_core.factors.register_factor`).
    """
    MODEL_REGISTRY.register(definition)


MODEL_REGISTRY.register(
    ModelDefinition(
        name="equal_weight",
        create=_create_equal_weight,
        description="mean of the configured factor columns",
    )
)
MODEL_REGISTRY.register(
    ModelDefinition(
        name="linear_score",
        create=_create_linear_score,
        description="fixed linear weights over the configured factors",
    )
)
MODEL_REGISTRY.register(
    ModelDefinition(
        name="ic_weighted",
        create=_create_ic_weighted,
        description="trailing mean rank IC weighted factor combination",
    ),
    aliases=("linear_ic",),
)

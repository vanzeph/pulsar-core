"""Modeler tests: equal weight, linear score, IC weighting (hand cases).

The IC scorer is exercised against a scripted FactorHistoryView whose
rank relationships are known by construction, so the estimated weights
are hand-derivable: a perfectly rank-predictive factor gets all weight,
an anti-predictive one gets none.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from pulsar_core import (
    CrossSection,
    EqualWeightScorer,
    FactorHistoryView,
    IcWeightedScorer,
    MODEL_REGISTRY,
    spearman_ic,
)
from pulsar_core.errors import PulsarCoreError


class ScriptedHistory(FactorHistoryView):
    """A history view with scripted dates, factor values and returns."""

    def __init__(
        self,
        *,
        factors: tuple[str, ...],
        values: dict[str, list[dict[str, float]]],
        returns: list[dict[str, float]],
        start: date = date(2026, 1, 5),
    ) -> None:
        self._factors = factors
        self._values = values
        self._returns = returns
        self._dates = [start + timedelta(days=index) for index in range(len(returns))]

    def factor_names(self) -> tuple[str, ...]:
        return self._factors

    def common_dates(self) -> tuple[date, ...]:
        return tuple(self._dates)

    def factor_values(self, factor: str, as_of: date) -> dict[str, "float | None"]:
        index = self._dates.index(as_of)
        return dict(self._values[factor][index])

    def forward_returns(self, from_date: date, horizon: int) -> dict[str, float]:
        if horizon != 1:
            return {}
        index = self._dates.index(from_date)
        return dict(self._returns[index])


class TestEqualWeight:
    def test_mean_of_factor_columns(self) -> None:
        section = CrossSection(
            as_of=date(2026, 6, 1),
            values={"f1": {"a": 1.0, "b": 2.0}, "f2": {"a": 3.0, "b": 4.0}},
        )
        scores = EqualWeightScorer().score(section, ScriptedHistory(factors=("f1", "f2"), values={}, returns=[]))
        assert scores == {"a": pytest.approx(2.0), "b": pytest.approx(3.0)}

    def test_symbols_missing_any_factor_are_dropped(self) -> None:
        section = CrossSection(
            as_of=date(2026, 6, 1),
            values={"f1": {"a": 1.0, "c": 5.0}, "f2": {"a": 3.0}},
        )
        scores = EqualWeightScorer().score(section, ScriptedHistory(factors=("f1", "f2"), values={}, returns=[]))
        assert set(scores) == {"a"}


class TestLinearScore:
    def _create(self, weights: dict[str, float], factors: tuple[str, ...]):
        return MODEL_REGISTRY.resolve("linear_score").create({"weights": weights}, factors)

    def test_fixed_linear_combination(self) -> None:
        scorer = self._create({"f1": 2.0, "f2": -1.0}, ("f1", "f2"))
        section = CrossSection(
            as_of=date(2026, 6, 1),
            values={"f1": {"a": 1.0, "b": 0.0}, "f2": {"a": 3.0, "b": 0.5}},
        )
        scores = scorer.score(section, ScriptedHistory(factors=("f1", "f2"), values={}, returns=[]))
        assert scores["a"] == pytest.approx(2.0 * 1.0 - 1.0 * 3.0)
        assert scores["b"] == pytest.approx(2.0 * 0.0 - 1.0 * 0.5)

    def test_weight_keys_must_match_configured_factors(self) -> None:
        with pytest.raises(PulsarCoreError, match="missing"):
            self._create({"f1": 1.0}, ("f1", "f2"))
        with pytest.raises(PulsarCoreError, match="unconfigured"):
            self._create({"f1": 1.0, "f2": 1.0, "f3": 1.0}, ("f1", "f2"))

    def test_weights_must_be_numbers(self) -> None:
        with pytest.raises(PulsarCoreError, match="must be a number"):
            self._create({"f1": "high"}, ("f1",))

    def test_integer_weights_accepted(self) -> None:
        scorer = self._create({"f1": 3, "f2": 1}, ("f1", "f2"))
        section = CrossSection(
            as_of=date(2026, 6, 1), values={"f1": {"a": 1.0}, "f2": {"a": 1.0}},
        )
        assert scorer.score(section, ScriptedHistory(factors=("f1", "f2"), values={}, returns=[]))["a"] == pytest.approx(4.0)


class TestIcWeighted:
    def _history(self, points: int) -> ScriptedHistory:
        # three symbols; 'good' rank-predicts returns perfectly, 'bad'
        # anti-predicts them, over `points` evaluation dates.
        good: list[dict[str, float]] = []
        bad: list[dict[str, float]] = []
        returns: list[dict[str, float]] = []
        for step in range(points):
            base = 1.0 + step * 0.1
            good.append({"a": base, "b": base + 1.0, "c": base + 2.0})
            bad.append({"a": base + 2.0, "b": base + 1.0, "c": base})
            returns.append({"a": 0.01, "b": 0.02, "c": 0.03})
        return ScriptedHistory(
            factors=("good", "bad"),
            values={"good": good, "bad": bad},
            returns=returns,
        )

    def test_predictive_factor_takes_all_weight(self) -> None:
        scorer = IcWeightedScorer(lookback=12, horizon=1, min_points=5)
        weights = scorer.factor_weights(self._history(10))
        assert weights["good"] == pytest.approx(1.0)
        assert weights["bad"] == pytest.approx(0.0)

    def test_young_history_falls_back_to_equal_weights(self) -> None:
        scorer = IcWeightedScorer(lookback=12, horizon=1, min_points=5)
        weights = scorer.factor_weights(self._history(3))
        assert weights == {"good": pytest.approx(0.5), "bad": pytest.approx(0.5)}

    def test_score_uses_estimated_weights(self) -> None:
        scorer = IcWeightedScorer(lookback=12, horizon=1, min_points=5)
        history = self._history(10)
        section = CrossSection(
            as_of=history.common_dates()[-1],
            values={"good": {"a": 1.0, "b": 2.0}, "bad": {"a": 9.0, "b": 9.0}},
        )
        scores = scorer.score(section, history)
        # bad has zero weight: the score is exactly the good column
        assert scores == {"a": pytest.approx(1.0), "b": pytest.approx(2.0)}

    def test_all_nonpositive_ics_fall_back_to_equal(self) -> None:
        # two perfectly anti-predictive factors
        values: dict[str, list[dict[str, float]]] = {"x": [], "y": []}
        returns: list[dict[str, float]] = []
        for step in range(10):
            base = 1.0 + step * 0.1
            values["x"].append({"a": base + 2.0, "b": base + 1.0, "c": base})
            values["y"].append({"a": base + 2.0, "b": base + 1.0, "c": base})
            returns.append({"a": 0.03, "b": 0.02, "c": 0.01})
        scorer = IcWeightedScorer(lookback=12, horizon=1, min_points=5)
        history = ScriptedHistory(factors=("x", "y"), values=values, returns=returns)
        assert scorer.factor_weights(history) == {"x": pytest.approx(0.5), "y": pytest.approx(0.5)}

    def test_param_validation(self) -> None:
        with pytest.raises(PulsarCoreError, match="must be an integer"):
            MODEL_REGISTRY.resolve("ic_weighted").create({"lookback": 1.5}, ("f1",))
        with pytest.raises(PulsarCoreError, match="parameters are"):
            MODEL_REGISTRY.resolve("ic_weighted").create({"n_estimators": 10}, ("f1",))


class TestSpearmanIC:
    def test_perfect_monotone_relationship(self) -> None:
        assert spearman_ic({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 10.0, "b": 20.0, "c": 30.0}) == pytest.approx(1.0)

    def test_anti_monotone_relationship(self) -> None:
        assert spearman_ic({"a": 3.0, "b": 2.0, "c": 1.0}, {"a": 10.0, "b": 20.0, "c": 30.0}) == pytest.approx(-1.0)

    def test_nonlinear_monotone_still_one(self) -> None:
        assert spearman_ic({"a": 1.0, "b": 8.0, "c": 1000.0}, {"a": 0.1, "b": 0.4, "c": 2.0}) == pytest.approx(1.0)

    def test_ties_use_average_ranks(self) -> None:
        # x ranks (with tie): 1.5, 1.5, 3 ; y ranks: 1, 2, 3
        ic = spearman_ic({"a": 1.0, "b": 1.0, "c": 2.0}, {"a": 1.0, "b": 2.0, "c": 3.0})
        expected = 0.8660254037844387  # hand-derived: corr([1.5,1.5,3],[1,2,3])
        assert ic == pytest.approx(expected)

    def test_constant_series_carries_no_information(self) -> None:
        assert spearman_ic({"a": 1.0, "b": 1.0, "c": 1.0}, {"a": 1.0, "b": 2.0, "c": 3.0}) is None

    def test_missing_value_or_too_few_symbols(self) -> None:
        assert spearman_ic({"a": None, "b": 2.0}, {"a": 1.0, "b": 2.0}) is None
        assert spearman_ic({"a": 1.0}, {"a": 1.0}) is None


class TestRegistry:
    def test_design_doc_alias_resolves(self) -> None:
        from pulsar_core.modelers import IcWeightedScorer as Concrete

        definition = MODEL_REGISTRY.resolve("linear_ic")
        assert definition is MODEL_REGISTRY.resolve("ic_weighted")
        scorer = definition.create({"lookback": 500}, ("f1",))
        assert isinstance(scorer, Concrete)
        assert scorer.lookback == 500
        assert "linear_ic" not in MODEL_REGISTRY.names()  # alias is not canonical

    def test_equal_weight_takes_no_parameters(self) -> None:
        with pytest.raises(PulsarCoreError, match="takes no parameters"):
            MODEL_REGISTRY.resolve("equal_weight").create({"lookback": 5}, ("f1",))

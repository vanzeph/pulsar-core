"""Preprocess pipeline tests: hand-computed winsorize / zscore / fillna.

Quantile references use the documented linear-interpolation rule
(numpy style) over pre-sorted values.
"""

from __future__ import annotations

import pytest

from pulsar_core import FillNaStep, WinsorizeStep, ZscoreStep, build_step
from pulsar_core.errors import PulsarCoreError


class TestWinsorize:
    def test_clips_to_quantiles_hand_calculation(self) -> None:
        # values 1..5, q = 0.2: lower at position 0.2*(5-1)=0.8 -> 1.8,
        # upper at position 3.2 -> 4.2
        step = WinsorizeStep({"quantile": 0.2})
        column = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0}
        assert step.apply(column) == pytest.approx(
            {"a": 1.8, "b": 2.0, "c": 3.0, "d": 4.0, "e": 4.2}, rel=1e-12
        )

    def test_default_quantile_on_one_to_ten(self) -> None:
        # q = 0.025 over 1..10: lower at 0.225 -> 1.225, upper at 8.775 -> 9.775
        step = WinsorizeStep()
        column = {str(v): float(v) for v in range(1, 11)}
        out = step.apply(column)
        assert out["1"] == pytest.approx(1.225)
        assert out["10"] == pytest.approx(9.775)
        assert out["5"] == 5.0

    def test_none_passes_through(self) -> None:
        step = WinsorizeStep({"quantile": 0.2})
        out = step.apply({"a": 1.0, "b": 5.0, "c": None})
        assert out["c"] is None
        assert out["a"] == pytest.approx(1.8)

    def test_all_missing_column_is_untouched(self) -> None:
        step = WinsorizeStep()
        assert step.apply({"a": None}) == {"a": None}

    def test_quantile_bounds_validated(self) -> None:
        with pytest.raises(PulsarCoreError, match="must lie in"):
            WinsorizeStep({"quantile": 0.6})
        with pytest.raises(PulsarCoreError, match="expects float"):
            WinsorizeStep({"quantile": "high"})


class TestZscore:
    def test_standardizes_hand_calculation(self) -> None:
        # values 1, 2, 3: mean 2, population std sqrt(2/3)
        step = ZscoreStep()
        std = (2.0 / 3.0) ** 0.5
        out = step.apply({"a": 1.0, "b": 2.0, "c": 3.0})
        assert out["a"] == pytest.approx(-1.0 / std)
        assert out["b"] == pytest.approx(0.0, abs=1e-12)
        assert out["c"] == pytest.approx(1.0 / std)
        mean = sum(out.values()) / len(out)
        assert mean == pytest.approx(0.0, abs=1e-12)

    def test_constant_column_maps_to_zero(self) -> None:
        out = ZscoreStep().apply({"a": 7.0, "b": 7.0, "c": 7.0})
        assert out == {"a": 0.0, "b": 0.0, "c": 0.0}

    def test_none_passes_through(self) -> None:
        out = ZscoreStep().apply({"a": 1.0, "b": None, "c": 3.0})
        assert out["b"] is None


class TestFillNa:
    def test_median_fill_hand_calculation(self) -> None:
        out = FillNaStep().apply({"a": 1.0, "b": None, "c": 3.0})
        assert out == {"a": 1.0, "b": 2.0, "c": 3.0}

    def test_median_uses_interpolated_median_of_evens(self) -> None:
        out = FillNaStep().apply({"a": 1.0, "b": None, "c": 2.0, "d": 10.0, "e": 12.0})
        assert out["b"] == pytest.approx(6.0)

    def test_zero_fill(self) -> None:
        out = FillNaStep({"method": "zero"}).apply({"a": None, "b": 5.0})
        assert out == {"a": 0.0, "b": 5.0}

    def test_all_missing_median_raises(self) -> None:
        with pytest.raises(PulsarCoreError, match="all-missing"):
            FillNaStep().apply({"a": None, "b": None})

    def test_method_validated(self) -> None:
        with pytest.raises(PulsarCoreError, match="median' or 'zero'"):
            FillNaStep({"method": "mean"})


class TestPipelineComposition:
    def test_steps_compose_in_config_order(self) -> None:
        column = {"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 100.0}
        winsorize = WinsorizeStep({"quantile": 0.2})
        zscore = ZscoreStep()
        composed = zscore.apply(winsorize.apply(column))
        # hand-derivation: sorted [1, 2, 3, 4, 100]; lower quantile at
        # position 0.2*4 = 0.8 -> 1.8; upper at 3.2 -> 4 + 0.2*(100-4) = 23.2
        manual = {"a": 1.8, "b": 2.0, "c": 3.0, "d": 4.0, "e": 23.2}
        values = list(manual.values())
        mean = sum(values) / len(values)
        std = (sum((v - mean) ** 2 for v in values) / len(values)) ** 0.5
        for symbol, value in manual.items():
            assert composed[symbol] == pytest.approx((value - mean) / std)

    def test_fillna_before_zscore_imputes_average_reading(self) -> None:
        pipeline = [FillNaStep(), ZscoreStep()]
        column = {"a": 1.0, "b": None, "c": 3.0}
        for step in pipeline:
            column = step.apply(column)
        assert column["b"] == pytest.approx(0.0, abs=1e-12)

    def test_build_step_resolves_registry_and_params(self) -> None:
        step = build_step("winsorize", {"quantile": 0.05})
        assert isinstance(step, WinsorizeStep)
        assert step.quantile == pytest.approx(0.05)

    def test_build_step_unknown_name_and_param(self) -> None:
        from pulsar_core import PREPROCESS_REGISTRY

        with pytest.raises(PulsarCoreError, match="unknown preprocess step"):
            build_step("quantile")
        assert "winsorize" in PREPROCESS_REGISTRY.names()
        with pytest.raises(PulsarCoreError, match="unknown parameter"):
            build_step("winsorize", {"n_std": 3})

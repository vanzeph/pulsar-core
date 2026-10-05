"""Signals, target portfolios and the built-in builders."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from pulsar_core import (
    EqualWeightBuilder,
    PassThroughBuilder,
    PulsarCoreError,
    Signal,
    TargetLeg,
    TargetPortfolio,
)


class TestSignal:
    def test_weight_signal(self) -> None:
        signal = Signal(symbol="600000", weight=0.5)
        assert signal.weight == 0.5
        assert signal.shares is None
        assert not signal.is_bare

    def test_shares_signal(self) -> None:
        signal = Signal(symbol="600000", shares=300)
        assert signal.is_bare is False

    def test_bare_signal_declares_interest_only(self) -> None:
        assert Signal(symbol="600000").is_bare

    def test_weight_and_shares_are_mutually_exclusive(self) -> None:
        with pytest.raises(ValidationError, match="both weight and shares"):
            Signal(symbol="600000", weight=0.5, shares=100)

    @pytest.mark.parametrize("field", ["weight", "shares"])
    def test_negative_targets_are_rejected(self, field: str) -> None:
        value = -0.1 if field == "weight" else -100
        with pytest.raises(ValidationError):
            Signal(symbol="600000", **{field: value})

    def test_zero_targets_are_explicit_exits(self) -> None:
        assert Signal(symbol="600000", weight=0.0).weight == 0.0
        assert Signal(symbol="600000", shares=0).shares == 0


class TestTargetPortfolio:
    def test_legs_carry_exactly_one_sizing(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            TargetLeg(symbol="600000")
        with pytest.raises(ValidationError, match="exactly one"):
            TargetLeg(symbol="600000", weight=1.0, shares=100)

    def test_map_keys_must_match_leg_symbols(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            TargetPortfolio(targets={"600000": TargetLeg(symbol="000001", weight=1.0)})

    def test_symbols_iterate_sorted(self) -> None:
        portfolio = TargetPortfolio(
            targets={
                "600000": TargetLeg(symbol="600000", weight=1.0),
                "000001": TargetLeg(symbol="000001", weight=0.5),
            }
        )
        assert portfolio.symbols == ("000001", "600000")
        assert len(portfolio) == 2


class TestPassThroughBuilder:
    def test_sized_signals_pass_verbatim(self) -> None:
        portfolio = PassThroughBuilder().build(
            [Signal(symbol="600000", weight=0.5), Signal(symbol="000001", shares=200)]
        )
        assert portfolio.targets["600000"].weight == 0.5
        assert portfolio.targets["000001"].shares == 200

    def test_bare_signals_cannot_be_sized(self) -> None:
        with pytest.raises(PulsarCoreError, match="bare signal"):
            PassThroughBuilder().build([Signal(symbol="600000")])

    def test_conflicting_duplicate_targets_are_rejected(self) -> None:
        with pytest.raises(PulsarCoreError, match="conflicting"):
            PassThroughBuilder().build(
                [Signal(symbol="600000", weight=0.5), Signal(symbol="600000", weight=1.0)]
            )


class TestEqualWeightBuilder:
    def test_bare_signals_share_equity_equally(self) -> None:
        portfolio = EqualWeightBuilder().build(
            [Signal(symbol="A"), Signal(symbol="B"), Signal(symbol="C")]
        )
        weights = {leg.weight for leg in portfolio.targets.values()}
        assert weights == {1.0 / 3}

    def test_explicit_signals_pass_through_untouched(self) -> None:
        portfolio = EqualWeightBuilder().build(
            [Signal(symbol="A", weight=0.2), Signal(symbol="B", shares=500)]
        )
        assert portfolio.targets["A"].weight == 0.2
        assert portfolio.targets["B"].shares == 500

    def test_mixed_declaration_sizes_only_the_bare_part(self) -> None:
        portfolio = EqualWeightBuilder().build(
            [Signal(symbol="A"), Signal(symbol="B"), Signal(symbol="P", weight=0.5)]
        )
        assert portfolio.targets["A"].weight == pytest.approx(0.5)
        assert portfolio.targets["B"].weight == pytest.approx(0.5)
        assert portfolio.targets["P"].weight == 0.5

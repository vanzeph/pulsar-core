"""Factor library tests: registry surface and hand-computed factor values.

Every numerical assertion is derived by hand from the formulas documented
in ``pulsar_core.factors`` on small bar samples.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Sequence

import pytest
from pulsar_contracts import Bar, Freq, SHANGHAI_TZ

from pulsar_core import (
    FACTOR_REGISTRY,
    FactorDefinition,
    factor_value,
    momentum_factor,
    range_factor,
    register_factor,
    reversal_factor,
    volatility_factor,
)
from pulsar_core.errors import PulsarCoreError


def make_bars(
    closes: Sequence[float], *, symbol: str = "600000", start: date = date(2026, 1, 5)
) -> list[Bar]:
    """Bars built from scripted closes; high/low straddle open and close."""
    bars: list[Bar] = []
    previous = closes[0]
    for index, close in enumerate(closes):
        day = start + timedelta(days=index)
        bars.append(
            Bar(
                symbol=symbol,
                freq=Freq.DAILY,
                ts=datetime.combine(day, time(), tzinfo=SHANGHAI_TZ),
                open=previous,
                high=max(previous, close) * 1.01,
                low=min(previous, close) * 0.99,
                close=close,
                volume=1_000.0,
                amount=close * 1_000.0,
            )
        )
        previous = close
    return bars


class TestRegistry:
    def test_builtin_factors_are_registered(self) -> None:
        for name in ("momentum_20", "momentum_60", "volatility_20", "reversal_5", "range_20"):
            assert name in FACTOR_REGISTRY
        assert "momentum_20" in FACTOR_REGISTRY.names()

    def test_unknown_factor_lists_registered_names(self) -> None:
        with pytest.raises(PulsarCoreError, match="unknown factor 'ep_ttm'"):
            FACTOR_REGISTRY.resolve("ep_ttm")

    def test_register_custom_window_factor(self) -> None:
        register_factor(momentum_factor(10))
        assert "momentum_10" in FACTOR_REGISTRY
        closes = [100.0 + i for i in range(15)]
        expected = closes[-1] / closes[-11] - 1.0
        assert factor_value("momentum_10", make_bars(closes)) == pytest.approx(expected)

    def test_duplicate_registration_rejected(self) -> None:
        with pytest.raises(PulsarCoreError, match="already registered"):
            register_factor(momentum_factor(20))

    def test_definition_validation(self) -> None:
        with pytest.raises(ValueError, match="direction"):
            FactorDefinition(
                name="x",
                label="x",
                compute=lambda bars: 1.0,
                direction=0,
                min_bars=1,
            )


class TestHandComputedValues:
    def test_momentum_20_matches_hand_calculation(self) -> None:
        closes = [100.0 + i for i in range(25)]  # c_0 .. c_24
        expected = closes[-1] / closes[-21] - 1.0  # c_24 / c_4 - 1
        assert expected == pytest.approx(124.0 / 104.0 - 1.0)
        assert factor_value("momentum_20", make_bars(closes)) == pytest.approx(expected)

    def test_momentum_60_matches_hand_calculation(self) -> None:
        closes = [50.0 * 1.01**i for i in range(61)]
        expected = closes[-1] / closes[-61] - 1.0
        assert factor_value("momentum_60", make_bars(closes)) == pytest.approx(expected)

    def test_volatility_20_matches_hand_calculation(self) -> None:
        closes = [100.0 * 1.005**i for i in range(21)]
        returns = [
            closes[i] / closes[i - 1] - 1.0 for i in range(1, len(closes))
        ]
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / len(returns)
        std = variance**0.5
        # direction = -1 (low volatility preferred): oriented = -std
        assert factor_value("volatility_20", make_bars(closes)) == pytest.approx(-std)

    def test_reversal_5_orientation(self) -> None:
        closes = [100.0, 101.0, 102.0, 103.0, 104.0, 110.0]
        raw_return = closes[-1] / closes[-6] - 1.0
        # direction = -1: a high trailing return reads as "less preferred"
        assert factor_value("reversal_5", make_bars(closes)) == pytest.approx(-raw_return)

    def test_range_20_matches_hand_calculation(self) -> None:
        closes = [100.0 + (i % 3) for i in range(25)]
        bars = make_bars(closes)
        fractions = [
            (bar.high - bar.low) / bar.close for bar in bars[-20:]
        ]
        expected = -sum(fractions) / len(fractions)  # direction = -1
        assert factor_value("range_20", bars) == pytest.approx(expected)

    def test_insufficient_history_is_missing_not_error(self) -> None:
        assert factor_value("momentum_20", make_bars([100.0] * 20)) is None
        assert factor_value("volatility_20", make_bars([100.0] * 20)) is None
        assert factor_value("reversal_5", make_bars([100.0] * 5)) is None
        assert factor_value("range_20", make_bars([100.0] * 19)) is None

    def test_values_use_bars_up_to_and_including_current(self) -> None:
        closes = [100.0 + i for i in range(25)]
        bars = make_bars(closes)
        full = factor_value("momentum_20", bars)
        truncated = factor_value("momentum_20", bars[:-1])
        assert full == pytest.approx(closes[-1] / closes[-21] - 1.0)
        assert truncated == pytest.approx(closes[-2] / closes[-22] - 1.0)
        assert full != truncated

    def test_constructors_produce_consistent_metadata(self) -> None:
        assert momentum_factor(20).min_bars == 21
        assert volatility_factor(20).direction == -1
        assert reversal_factor(5).min_bars == 6
        assert range_factor(20).min_bars == 20
        assert momentum_factor(20).direction == 1

"""StrategyBase hooks, contexts, indicators and explicit state."""

from __future__ import annotations

import random
from datetime import date, datetime, time

import pytest
from pulsar_contracts import Bar, Freq, SHANGHAI_TZ

from pulsar_core import (
    BaseContext,
    BarContext,
    FillContext,
    IndicatorValue,
    Params,
    PortfolioView,
    PulsarCoreError,
    StrategyBase,
    TickContext,
)
from pulsar_core.strategy import _Declarations


def make_bar(symbol: str, day: date, close: float) -> Bar:
    stamp = datetime.combine(day, time(), tzinfo=SHANGHAI_TZ)
    return Bar(
        symbol=symbol,
        ts=stamp,
        freq=Freq.DAILY,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=1_000.0,
        amount=close * 1_000.0,
    )


def make_ctx(
    closes: list[float],
    symbol: str = "600000",
    *,
    collector: _Declarations | None = None,
) -> BarContext:
    base = date(2026, 6, 1)
    bars = [
        make_bar(symbol, date.fromordinal(base.toordinal() + index), close)
        for index, close in enumerate(closes)
    ]
    ctx = BarContext(
        bar=bars[-1],
        history=bars,
        now=bars[-1].ts,
        state={},
        portfolio=PortfolioView(cash=1_000.0, equity=1_000.0, positions=()),
    )
    if collector is not None:
        ctx.attach_collector(collector)
    return ctx


class TestIndicators:
    def test_sma_matches_manual_computation(self) -> None:
        closes = [float(n) for n in range(1, 26)]  # 1..25
        ctx = make_ctx(closes)
        reading = ctx.sma("close", 5)
        assert reading.value == pytest.approx(sum(range(21, 26)) / 5)  # mean(21..25)
        assert reading.previous == pytest.approx(sum(range(20, 25)) / 5)

    def test_sma_warmup_returns_none_none(self) -> None:
        ctx = make_ctx([10.0, 11.0, 12.0])
        reading = ctx.sma("close", 5)
        assert reading == IndicatorValue(value=None, previous=None)

    def test_sma_exactly_n_bars_has_no_previous(self) -> None:
        ctx = make_ctx([10.0, 11.0, 12.0])
        reading = ctx.sma("close", 3)
        assert reading.value == pytest.approx(11.0)
        assert reading.previous is None

    def test_cross_up_and_cross_down_detect_the_flip(self) -> None:
        # rise (fast above slow), fall (death cross), rise (golden cross)
        closes = [40.0, 42.0, 44.0, 46.0, 48.0, 50.0, 45.0, 41.0, 39.0, 38.0, 40.0, 43.0, 47.0, 52.0]
        crossed_up_at: list[int] = []
        crossed_down_at: list[int] = []
        for upto in range(4, len(closes) + 1):
            ctx = make_ctx(closes[:upto])
            fast, slow = ctx.sma("close", 2), ctx.sma("close", 4)
            if ctx.cross_up(fast, slow):
                crossed_up_at.append(upto)
            if ctx.cross_down(fast, slow):
                crossed_down_at.append(upto)
        assert len(crossed_up_at) == 1  # exactly one golden cross
        assert len(crossed_down_at) == 1  # and one death cross earlier
        assert crossed_down_at[0] < crossed_up_at[0]

    def test_cross_with_none_inputs_is_false(self) -> None:
        ctx = make_ctx([10.0, 11.0])
        assert ctx.cross_up(IndicatorValue(value=None, previous=None), None) is False
        assert ctx.cross_down(None, IndicatorValue(value=1.0)) is False

    def test_field_series_and_bars_window(self) -> None:
        ctx = make_ctx([5.0, 6.0, 7.0, 8.0])
        assert ctx.field_series("close", 2) == (7.0, 8.0)
        assert [bar.close for bar in ctx.bars(2)] == [7.0, 8.0]
        assert ctx.symbol == "600000"
        assert ctx.bar.close == 8.0

    def test_unknown_field_and_bad_n_are_rejected(self) -> None:
        ctx = make_ctx([5.0, 6.0])
        with pytest.raises(ValueError, match="unknown bar field"):
            ctx.sma("settle", 2)
        with pytest.raises(ValueError, match="n must be positive"):
            ctx.sma("close", 0)
        with pytest.raises(ValueError, match="n must be positive"):
            ctx.bars(0)


class TestContexts:
    def test_base_context_exposes_reads(self) -> None:
        ctx = BaseContext(
            now=datetime(2026, 6, 1, tzinfo=SHANGHAI_TZ),
            state={"n": 1},
            portfolio=PortfolioView(cash=100.0, equity=150.0, positions=()),
        )
        assert ctx.cash == 100.0
        assert ctx.equity == 150.0
        assert ctx.state == {"n": 1}
        assert ctx.position_quantity("600000") == 0

    def test_declaration_outside_dispatch_raises(self) -> None:
        ctx = make_ctx([10.0])
        with pytest.raises(PulsarCoreError, match="only available"):
            ctx.target_weight("600000", 1.0)

    def test_conflicting_declarations_in_one_point_are_rejected(self) -> None:
        collector = _Declarations()
        ctx = make_ctx([10.0], collector=collector)
        ctx.target_weight("600000", 1.0)
        with pytest.raises(PulsarCoreError, match="conflicting"):
            ctx.target_shares("600000", 100)

    def test_identical_redeclaration_is_idempotent(self) -> None:
        collector = _Declarations()
        ctx = make_ctx([10.0], collector=collector)
        ctx.target_weight("600000", 1.0)
        ctx.target_weight("600000", 1.0)
        assert len(collector.take()) == 1


class TestStrategyBase:
    def test_params_bind_from_configuration(self) -> None:
        class Demo(StrategyBase):
            params = Params(fast=5, slow=20)

        assert Demo().params.fast == 5
        assert Demo({"fast": 3}).params.fast == 3
        assert Demo({"fast": 3}).params.slow == 20
        with pytest.raises(PulsarCoreError):
            Demo({"nope": 1})

    def test_default_hooks_are_noops(self) -> None:
        strategy = StrategyBase()
        ctx = BaseContext(
            now=datetime(2026, 6, 1, tzinfo=SHANGHAI_TZ),
            state={},
            portfolio=PortfolioView(cash=0.0, equity=0.0),
        )
        strategy.on_start(ctx)
        strategy.on_bar(make_ctx([1.0]))  # type: ignore[arg-type]
        strategy.on_stop(ctx)

    def test_non_params_declaration_is_rejected(self) -> None:
        class Broken(StrategyBase):
            params = {"fast": 5}  # type: ignore[assignment]

        with pytest.raises(PulsarCoreError, match="must be a Params"):
            Broken()


class TestDeterministicIndicators:
    def test_sma_is_a_pure_function_of_history(self) -> None:
        rng = random.Random(11)
        closes = [round(100 * (1 + rng.uniform(-0.02, 0.02)), 4) for _ in range(50)]
        first = make_ctx(closes).sma("close", 10)
        second = make_ctx(list(closes)).sma("close", 10)
        assert first == second


def test_tick_context_declarations(tmp_path=None) -> None:  # noqa: ARG001
    from pulsar_contracts import QuoteLevel, Snapshot

    snapshot = Snapshot(
        symbol="600000",
        ts=datetime(2026, 6, 1, 1, 0, tzinfo=SHANGHAI_TZ),
        seq=1,
        last_price=10.0,
        volume=100.0,
        amount=1_000.0,
        bids=(QuoteLevel(price=9.9, volume=10),),
        asks=(QuoteLevel(price=10.1, volume=10),),
    )
    collector = _Declarations()
    ctx = TickContext(
        snapshot=snapshot,
        now=snapshot.ts,
        state={},
        portfolio=PortfolioView(cash=1.0, equity=1.0),
    )
    ctx.attach_collector(collector)
    assert ctx.symbol == "600000"
    ctx.target_weight("600000", 0.5)
    signals = collector.take()
    assert signals[0].symbol == "600000"
    assert signals[0].weight == 0.5


def test_fill_context_exposes_the_fill() -> None:
    from pulsar_contracts import Fill, Side

    fill = Fill(
        fill_id="f1",
        order_id="o1",
        symbol="600000",
        side=Side.BUY,
        price=10.0,
        quantity=100,
        ts=datetime(2026, 6, 1, 1, 0, tzinfo=SHANGHAI_TZ),
    )
    ctx = FillContext(
        fill=fill,
        now=fill.ts,
        state={},
        portfolio=PortfolioView(cash=900.0, equity=1_000.0),
    )
    assert ctx.fill is fill

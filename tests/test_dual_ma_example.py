"""The shipped dual-MA example runs as a real Research-mode session."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pulsar_contracts import Side

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))

from dual_ma import DualMAStrategy, ImmediateFillVenue, SyntheticDailyBars, main  # noqa: E402

from pulsar_core import (  # noqa: E402
    BacktestClock,
    EventBus,
    Params,
    ReplaySession,
    RiskGate,
    SinglePositionCapRule,
    StrategyRuntime,
)


class TestExampleStrategy:
    def test_declares_its_parameters(self) -> None:
        assert DualMAStrategy.params.names() == ("fast", "slow")
        assert DualMAStrategy().params.fast == 5
        assert DualMAStrategy({"fast": 3, "slow": 8}).params.slow == 8

    def test_research_run_produces_a_full_round_trip(self, capsys) -> None:
        main()  # prints a run summary; also proves the example is runnable
        output = capsys.readouterr().out
        assert "run_id" in output
        assert "final positions       : []" in output  # flat at the finish

        # and again programmatically, asserting on the objects themselves
        port = SyntheticDailyBars()
        start, end = port.span()
        bus = EventBus(BacktestClock(_synthetic_day_start(start)))
        venue = ImmediateFillVenue(now=lambda: bus.now)
        strategy = DualMAStrategy()
        runtime = StrategyRuntime(
            bus=bus,
            port=venue,
            strategy=strategy,
            gate=RiskGate((SinglePositionCapRule(1.0),)),
            initial_cash=100_000.0,
        )
        session = ReplaySession(
            port=port,
            symbols=["600000"],
            start=start,
            end=end,
            seed=7,
            config={"strategy": {"params": strategy.params.to_dict()}},
            bus=bus,
            on_manifest=runtime.bind_manifest,
        )
        result = session.run()

        sides = [intent.side for intent in runtime.intents]
        assert sides == [Side.BUY, Side.SELL]  # golden cross in, death cross out
        buy, sell = runtime.intents
        assert buy.quantity % 100 == 0  # board-lot buys
        assert sell.quantity == buy.quantity  # full liquidation
        assert sell.limit_price > buy.limit_price  # the rally paid for the trip
        assert runtime.rejections == []  # within the 1.0 position cap
        final = runtime.account.snapshot()
        assert final.positions == ()
        assert final.equity == pytest.approx(final.cash)
        assert final.equity > 100_000.0
        assert result.bar_events == 100

    def test_identical_example_runs_are_bit_identical(self) -> None:
        outcomes = []
        for _ in range(2):
            port = SyntheticDailyBars()
            start, end = port.span()
            bus = EventBus(BacktestClock(_synthetic_day_start(start)))
            venue = ImmediateFillVenue(now=lambda: bus.now)
            runtime = StrategyRuntime(
                bus=bus,
                port=venue,
                strategy=DualMAStrategy(),
                gate=RiskGate((SinglePositionCapRule(1.0),)),
                initial_cash=100_000.0,
            )
            session = ReplaySession(
                port=port,
                symbols=["600000"],
                start=start,
                end=end,
                seed=7,
                bus=bus,
                on_manifest=runtime.bind_manifest,
            )
            result = session.run()
            outcomes.append(
                (
                    result.run_id,
                    result.journal_digest,
                    runtime.intents,
                    runtime.account.snapshot(),
                )
            )
        first, second = outcomes
        assert first == second


def _synthetic_day_start(day):
    from datetime import datetime, time

    from pulsar_contracts import SHANGHAI_TZ

    return datetime.combine(day, time(), tzinfo=SHANGHAI_TZ)

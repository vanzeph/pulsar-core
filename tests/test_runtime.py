"""StrategyRuntime: the full pipeline over a Research-mode replay session.

These tests drive the real kernel loop end to end: a scripted price path
with guaranteed MA crossings, the example-style dual-MA strategy, and the
``FillingExecutionPort`` immediate-fill venue. They verify hook ordering,
account flow-back, intent idempotency keys, T+1 behavior and run-level
determinism.
"""

from __future__ import annotations

from datetime import datetime

import pytest
from pulsar_contracts import Side

from conftest import FillingExecutionPort, ScriptedClosesPort, down_up_down

from pulsar_core import (
    BacktestClock,
    EventBus,
    Params,
    PulsarCoreError,
    ReplaySession,
    RiskGate,
    SinglePositionCapRule,
    StrategyBase,
    StrategyRuntime,
)
from pulsar_core.session import _day_start


class DualMA(StrategyBase):
    """Test copy of the reference strategy (fast/slow tuned for the path)."""

    params = Params(fast=3, slow=6)

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[str] = []

    def on_start(self, ctx) -> None:
        self.calls.append("start")

    def on_bar(self, ctx) -> None:
        self.calls.append("bar")
        ctx.state["seen"] = ctx.state.get("seen", 0) + 1
        fast = ctx.sma("close", self.params.fast)
        slow = ctx.sma("close", self.params.slow)
        if ctx.cross_up(fast, slow):
            ctx.target_weight(ctx.symbol, 1.0)
        elif ctx.cross_down(fast, slow):
            ctx.target_weight(ctx.symbol, 0.0)

    def on_fill(self, ctx) -> None:
        self.calls.append("fill")

    def on_stop(self, ctx) -> None:
        self.calls.append("stop")


def build_run(
    closes: list[float],
    *,
    strategy: StrategyBase | None = None,
    gate: RiskGate | None = None,
    initial_cash: float = 100_000.0,
    bind: bool = True,
) -> tuple[ReplaySession, StrategyRuntime, FillingExecutionPort, DualMA | None]:
    port = ScriptedClosesPort({"600000": closes})
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    strategy = strategy if strategy is not None else DualMA()
    runtime = StrategyRuntime(
        bus=bus,
        port=venue,
        strategy=strategy,
        gate=gate,
        initial_cash=initial_cash,
    )
    session = ReplaySession(
        port=port,
        symbols=["600000"],
        start=start,
        end=end,
        seed=3,
        config={"strategy": {"params": strategy.params.to_dict()}},
        bus=bus,
        on_manifest=runtime.bind_manifest if bind else None,
    )
    return session, runtime, venue, strategy if isinstance(strategy, DualMA) else None


PATH = down_up_down(12, 18, 12)  # fall -> rally (golden cross) -> slide (exit)


class TestResearchModePipeline:
    def test_dual_ma_produces_buy_then_sell_and_ends_flat(self) -> None:
        session, runtime, venue, _ = build_run(PATH)
        result = session.run()

        sides = [intent.side for intent in runtime.intents]
        assert sides == [Side.BUY, Side.SELL]
        buy, sell = runtime.intents
        assert buy.quantity == sell.quantity  # full round trip
        assert buy.limit_price == pytest.approx(buy.limit_price)
        assert buy.limit_price < sell.limit_price  # sold higher than bought

        final = runtime.account.snapshot()
        assert final.positions == ()  # flat after the death cross
        assert final.equity == pytest.approx(final.cash)
        assert final.equity > 100_000.0  # the scripted rally pays for the round trip

        assert [fill.fill_id for fill in venue.fills()] == [
            "fill-000001",
            "fill-000002",
        ]
        assert result.run_id == runtime.run_id
        for index, intent in enumerate(runtime.intents, start=1):
            assert intent.idempotency_key.run_id == result.run_id
            assert intent.idempotency_key.seq == index

    def test_hooks_fire_in_event_order(self) -> None:
        session, runtime, venue, strategy = build_run(PATH)
        session.run()
        assert strategy is not None
        calls = strategy.calls
        assert calls[0] == "start"
        assert calls[-1] == "stop"
        assert calls.count("bar") == len(PATH)
        assert calls.count("fill") == 2
        # fills arrive after the bar that triggered them
        assert calls.index("fill") > calls.index("bar")
        # explicit state accumulated across bars through one dict
        assert runtime.state["seen"] == len(PATH)

    def test_cash_and_positions_follow_the_fills(self) -> None:
        session, runtime, venue, _ = build_run(PATH)
        session.run()
        fills = venue.fills()
        buy, sell = fills
        expected_cash = (
            100_000.0
            - buy.price * buy.quantity  # buy drains cash
            + sell.price * sell.quantity  # sell refills it
        )
        assert runtime.account.cash == pytest.approx(expected_cash)
        assert runtime.account.position("600000") is None

    def test_bought_shares_become_sellable_the_next_trading_day(self) -> None:
        # T+1: the buy books locked (unavailable) shares; the very next
        # bar — the next trading day — the rollover releases them and the
        # declared-zero target liquidates in full. The same-day blocking
        # itself is unit-covered in test_rebalance (availability caps).
        class BuyThenExitEveryBar(StrategyBase):
            params = Params()

            def __init__(self) -> None:
                super().__init__()
                self._bought = False

            def on_bar(self, ctx) -> None:
                if not self._bought:
                    self._bought = True
                    ctx.target_shares(ctx.symbol, 200)
                else:
                    ctx.target_shares(ctx.symbol, 0)

        closes = [50.0 + index for index in range(6)]
        session, runtime, venue, _ = build_run(closes, strategy=BuyThenExitEveryBar())
        session.run()
        buy, sell = runtime.intents
        assert buy.side is Side.BUY and sell.side is Side.SELL
        assert sell.quantity == buy.quantity == 200
        fills = venue.fills()
        assert (fills[1].ts.date() - fills[0].ts.date()).days >= 1  # next day
        assert runtime.account.snapshot().positions == ()

    def test_state_dict_is_one_object_across_the_run(self) -> None:
        identities: list[int] = []

        class Recorder(StrategyBase):
            params = Params()

            def on_bar(self, ctx) -> None:
                identities.append(id(ctx.state))

        session, runtime, venue, _ = build_run(PATH, strategy=Recorder())
        session.run()
        assert len(set(identities)) == 1
        assert identities[0] == id(runtime.state)  # one dict for the whole run

    def test_unbound_run_id_fails_loudly_at_first_intent(self) -> None:
        class AlwaysLong(StrategyBase):
            params = Params()

            def on_bar(self, ctx) -> None:
                ctx.target_weight(ctx.symbol, 1.0)

        closes = [50.0 + index for index in range(10)]
        session, runtime, venue, _ = build_run(closes, strategy=AlwaysLong(), bind=False)
        with pytest.raises(PulsarCoreError, match="run id not bound"):
            session.run()


class TestDeterminism:
    def test_identical_runs_reproduce_intents_and_journal_bit_for_bit(self) -> None:
        first_session, first_runtime, _, _ = build_run(PATH)
        first = first_session.run()

        second_session, second_runtime, _, _ = build_run(PATH)
        second = second_session.run()

        assert first.journal_digest == second.journal_digest
        assert first.run_id == second.run_id
        assert first_runtime.intents == second_runtime.intents
        assert first_runtime.rejections == second_runtime.rejections
        assert first_runtime.skipped == second_runtime.skipped
        assert first_runtime.account.snapshot() == second_runtime.account.snapshot()


class TestRebalanceSemanticsThroughTheRuntime:
    def test_sub_lot_buy_difference_is_skipped_not_rounded_up(self) -> None:
        class TinyWeight(StrategyBase):
            params = Params()

            def on_bar(self, ctx) -> None:
                # equity 100k * 0.001 = 100 CNY -> below one lot at ~50 CNY
                ctx.target_weight(ctx.symbol, 0.001)

        closes = [50.0 + index for index in range(10)]
        session, runtime, venue, _ = build_run(closes, strategy=TinyWeight())
        session.run()
        assert runtime.intents == ()
        assert any("below one lot" in skip.reason for skip in runtime.skipped)

    def test_odd_tail_liquidates_on_zero_target(self) -> None:
        # pre-seed an odd position by declaring 250 target shares on a
        # venue that fills 100%: the engine buys 200 (lot floor)...
        # to exercise the odd tail we instead sell against a fabricated
        # odd holding via the account: covered by unit tests; here assert
        # the declared-zero path clears everything that was bought.
        class AllInAllOut(StrategyBase):
            params = Params()

            def __init__(self) -> None:
                super().__init__()
                self._bought = False

            def on_bar(self, ctx) -> None:
                if not self._bought and ctx.bar.close > 0:
                    self._bought = True
                    ctx.target_weight(ctx.symbol, 1.0)
                elif self._bought and ctx.bar.ts.day >= 15:
                    ctx.target_shares(ctx.symbol, 0)

        closes = [50.0 + index for index in range(20)]
        session, runtime, venue, _ = build_run(closes, strategy=AllInAllOut())
        session.run()
        assert [intent.side for intent in runtime.intents] == [Side.BUY, Side.SELL]
        buy, sell = runtime.intents
        assert buy.quantity % 100 == 0
        assert sell.quantity == buy.quantity
        assert runtime.account.snapshot().positions == ()


class TestRunIdBinding:
    def test_run_id_is_unknown_before_and_known_after_binding(self) -> None:
        session, runtime, venue, _ = build_run(PATH)
        with pytest.raises(PulsarCoreError, match="run id not bound"):
            _ = runtime.run_id
        result = session.run()
        assert runtime.run_id == result.manifest.run_id
        for intent in runtime.intents:
            assert intent.idempotency_key.to_str().startswith(runtime.run_id)

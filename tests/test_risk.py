"""RiskGate: chain semantics and the design's five pre-trade rules.

Coverage mirrors the acceptance criterion "五类规则各自越界场景被拒绝且
留痕": every rule family gets at least one passing and one rejecting case,
rejections carry the structured rule name / threshold / actual triple, and
the daily-loss halt's explicit state behavior (trip, persist, re-arm) is
exercised both at the gate level and through a real runtime loop.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from pandas import DataFrame
import pytest
from pulsar_contracts import (
    AdjustMode,
    CorporateAction,
    Freq,
    Instrument,
    InstrumentStatus,
    Side,
    Snapshot,
    Subscription,
    SHANGHAI_TZ,
)

from conftest import FillingExecutionPort, ScriptedClosesPort

from pulsar_core import (
    BacktestClock,
    DailyLossHaltRule,
    EventBus,
    LiquidityFloorRule,
    OrderDraft,
    PortfolioExposureCapRule,
    Params,
    PositionView,
    ReplaySession,
    RejectionRecord,
    RiskGate,
    RiskRule,
    RiskView,
    RuleRejection,
    SinglePositionCapRule,
    StrategyBase,
    StrategyRuntime,
    SymbolBlacklistRule,
    standard_risk_chain,
)
from pulsar_core.session import _day_start

DAY1 = date(2026, 6, 1)


def at(day: date, hour: int = 10) -> datetime:
    return datetime.combine(day, time(hour), tzinfo=SHANGHAI_TZ)


TS = at(DAY1)
DAY2 = DAY1 + timedelta(days=1)

CANONICAL_COLUMNS = [
    "symbol",
    "ts",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "amount",
    "adjust_factor",
    "quality",
]


class _RejectWith:
    """Test rule rejecting everything with a fixed message."""

    def __init__(self, name: str, reason: str) -> None:
        self.name = name
        self._reason = reason

    def check(self, draft: OrderDraft, view: RiskView) -> str | None:
        return self._reason


class _AlwaysPass:
    name = "always_pass"

    def check(self, draft: OrderDraft, view: RiskView) -> str | None:
        return None


def make_view(
    *,
    equity: float = 10_000.0,
    ts: datetime = TS,
    positions: dict[str, PositionView] | None = None,
    prices: dict[str, float] | None = None,
    amounts: dict[str, float] | None = None,
    pending: dict[str, int] | None = None,
) -> RiskView:
    return RiskView(
        ts=ts,
        equity=equity,
        cash=equity,
        positions=positions or {},
        last_prices=prices or {},
        last_amounts=amounts or {},
        pending=pending or {},
    )


def buy(symbol: str = "A", quantity: int = 100) -> OrderDraft:
    return OrderDraft(side=Side.BUY, symbol=symbol, quantity=quantity)


def sell(symbol: str = "A", quantity: int = 100) -> OrderDraft:
    return OrderDraft(side=Side.SELL, symbol=symbol, quantity=quantity)


def holding(symbol: str, quantity: int, available: int | None = None) -> PositionView:
    return PositionView(
        symbol=symbol,
        quantity=quantity,
        available_quantity=quantity if available is None else available,
    )


def instrument(
    symbol: str = "600001",
    *,
    is_st: bool = False,
    status: InstrumentStatus = InstrumentStatus.LISTED,
    delist_date: date | None = None,
) -> Instrument:
    return Instrument(
        symbol=symbol,
        exchange="SSE",
        board="main",
        is_st=is_st,
        status=status,
        list_date=date(2000, 1, 1),
        delist_date=delist_date,
    )


class TestChainSemantics:
    def test_passing_drafts_keep_their_order(self) -> None:
        drafts = [buy("A"), sell("B", 200)]
        outcome = RiskGate((_AlwaysPass(),)).review(drafts, make_view())
        assert outcome.approved == tuple(drafts)
        assert outcome.rejections == ()

    def test_rejected_drafts_are_dropped_with_reason_recorded(self) -> None:
        drafts = [buy("A"), buy("B", 200)]
        gate = RiskGate((_RejectWith("no_trading", "market closed"),))
        outcome = gate.review(drafts, make_view())
        assert outcome.approved == ()
        assert [record.rule for record in outcome.rejections] == ["no_trading"] * 2
        assert {record.reason for record in outcome.rejections} == {"market closed"}
        assert all(record.ts == TS for record in outcome.rejections)

    def test_first_rejecting_rule_wins(self) -> None:
        gate = RiskGate(
            (
                _RejectWith("first", "rejected by first"),
                _RejectWith("second", "rejected by second"),
            )
        )
        outcome = gate.review([buy()], make_view())
        assert outcome.rejections[0].rule == "first"

    def test_mixed_outcome_splits_approved_and_rejected(self) -> None:
        class RejectB:
            name = "reject_b"

            def check(self, draft: OrderDraft, view: RiskView) -> str | None:
                return "B is off limits" if draft.symbol == "B" else None

        drafts = [buy("A"), buy("B")]
        outcome = RiskGate((RejectB(),)).review(drafts, make_view())
        assert [draft.symbol for draft in outcome.approved] == ["A"]
        assert [record.symbol for record in outcome.rejections] == ["B"]

    def test_duplicate_rule_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            RiskGate((_AlwaysPass(), _AlwaysPass()))

    def test_rule_names_expose_the_chain_order(self) -> None:
        gate = RiskGate((_RejectWith("a", "x"), _AlwaysPass()))
        assert gate.rule_names == ("a", "always_pass")

    def test_structured_verdicts_carry_threshold_and_actual(self) -> None:
        class RejectStructured(RiskRule):
            name = "structured"

            def check(
                self, draft: OrderDraft, view: RiskView
            ) -> str | RuleRejection | None:
                return RuleRejection(
                    reason="over the line", threshold=0.5, actual=0.75
                )

        outcome = RiskGate((RejectStructured(),)).review([buy()], make_view())
        record = outcome.rejections[0]
        assert record.rule == "structured"
        assert record.reason == "over the line"
        assert record.threshold == 0.5
        assert record.actual == 0.75

    def test_plain_string_verdicts_stay_unstructured(self) -> None:
        # backward compatibility: a rule may still return a bare string
        outcome = RiskGate((_RejectWith("plain", "nope"),)).review([buy()], make_view())
        record = outcome.rejections[0]
        assert record.reason == "nope"
        assert record.threshold is None
        assert record.actual is None


class TestSinglePositionCap:
    def test_buy_within_cap_passes(self) -> None:
        rule = SinglePositionCapRule(0.5)
        view = make_view(equity=10_000.0, prices={"A": 10.0})
        assert rule.check(buy("A", 500), view) is None  # exactly 50%

    def test_buy_pushing_past_the_cap_is_rejected(self) -> None:
        rule = SinglePositionCapRule(0.5)
        view = make_view(equity=10_000.0, prices={"A": 10.0})
        verdict = rule.check(buy("A", 501), view)  # 50.1%
        assert isinstance(verdict, RuleRejection)
        assert "exceeds cap" in verdict.reason
        assert verdict.threshold == 0.5
        assert verdict.actual == pytest.approx(0.501)

    def test_cap_accounts_for_holdings_and_pending_buys(self) -> None:
        rule = SinglePositionCapRule(0.5)
        view = make_view(
            equity=10_000.0,
            positions={"A": holding("A", 300)},
            prices={"A": 10.0},
            pending={"A": 100},  # 400 committed already = 40%
        )
        assert rule.check(buy("A", 100), view) is None  # exactly 50%
        assert rule.check(buy("A", 101), view) is not None

    def test_sells_always_pass(self) -> None:
        rule = SinglePositionCapRule(0.1)
        view = make_view(
            equity=10_000.0,
            positions={"A": holding("A", 900)},
            prices={"A": 10.0},
        )
        assert rule.check(sell("A", 900), view) is None

    def test_buy_without_a_price_is_rejected(self) -> None:
        rule = SinglePositionCapRule(1.0)
        verdict = rule.check(buy(), make_view())
        assert isinstance(verdict, RuleRejection)
        assert "no last price" in verdict.reason

    def test_buy_with_non_positive_equity_is_rejected(self) -> None:
        rule = SinglePositionCapRule(1.0)
        view = make_view(equity=0.0, prices={"A": 10.0})
        verdict = rule.check(buy(), view)
        assert isinstance(verdict, RuleRejection)
        assert "non-positive equity" in verdict.reason

    def test_invalid_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            SinglePositionCapRule(1.5)
        with pytest.raises(ValueError, match="max_weight"):
            SinglePositionCapRule(0.0)


class TestPortfolioExposureCap:
    def test_buy_exactly_at_the_cap_passes(self) -> None:
        rule = PortfolioExposureCapRule(1.0)
        view = make_view(equity=10_000.0, prices={"A": 10.0})
        assert rule.check(buy("A", 1000), view) is None  # gross = 100%

    def test_buy_pushing_gross_past_the_cap_is_rejected(self) -> None:
        rule = PortfolioExposureCapRule(1.0)
        view = make_view(equity=10_000.0, prices={"A": 10.0})
        verdict = rule.check(buy("A", 1001), view)
        assert isinstance(verdict, RuleRejection)
        assert "gross exposure" in verdict.reason
        assert verdict.threshold == 1.0
        assert verdict.actual == pytest.approx(1.001)

    def test_other_holdings_and_pending_buys_count_toward_gross(self) -> None:
        rule = PortfolioExposureCapRule(1.0)
        # B held 400 + 200 pending buy = 6,000 at 10; A draft 600 = 6,000
        view = make_view(
            equity=10_000.0,
            positions={"B": holding("B", 400)},
            prices={"A": 10.0, "B": 10.0},
            pending={"B": 200},
        )
        assert rule.check(buy("A", 399), view) is None  # 9,990 <= 10,000
        assert rule.check(buy("A", 401), view) is not None  # 10,010 > 10,000

    def test_sells_pass_even_when_gross_is_already_over(self) -> None:
        rule = PortfolioExposureCapRule(1.0)
        view = make_view(
            equity=10_000.0,
            positions={"A": holding("A", 1100)},
            prices={"A": 10.0},  # held gross = 11,000 = 110%
        )
        assert rule.check(sell("A", 100), view) is None

    def test_unmarkable_holding_rejects_new_buys(self) -> None:
        rule = PortfolioExposureCapRule(1.0)
        view = make_view(
            equity=10_000.0,
            positions={"B": holding("B", 500)},
            prices={"A": 10.0},  # no price for B -> gross unverifiable
        )
        verdict = rule.check(buy("A", 100), view)
        assert isinstance(verdict, RuleRejection)
        assert "cannot mark holding B" in verdict.reason

    def test_leveraged_assemblies_may_raise_the_cap(self) -> None:
        rule = PortfolioExposureCapRule(1.5)
        view = make_view(
            equity=10_000.0,
            positions={"B": holding("B", 400)},
            prices={"A": 10.0, "B": 10.0},
        )
        assert rule.check(buy("A", 1000), view) is None  # gross 140%

    def test_invalid_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_gross_weight"):
            PortfolioExposureCapRule(0.0)
        with pytest.raises(ValueError, match="max_gross_weight"):
            PortfolioExposureCapRule(-1.0)


class TestDailyLossHalt:
    def test_drawdown_below_the_limit_passes(self) -> None:
        rule = DailyLossHaltRule(0.05)
        gate = RiskGate((rule,))
        morning = make_view(equity=100_000.0, ts=at(DAY1, 10))
        assert gate.review([buy()], morning).approved == (buy(),)
        assert rule.halted is False
        assert rule.day_start_equity == 100_000.0

    def test_touching_the_limit_trips_the_halt(self) -> None:
        rule = DailyLossHaltRule(0.05)
        gate = RiskGate((rule,))
        gate.review([buy()], make_view(equity=100_000.0, ts=at(DAY1, 10)))
        afternoon = make_view(equity=95_000.0, ts=at(DAY1, 14))  # exactly -5%
        outcome = gate.review([buy()], afternoon)
        assert outcome.approved == ()
        record = outcome.rejections[0]
        assert record.rule == "daily_loss_halt"
        assert "halt active" in record.reason
        assert record.threshold == 0.05
        assert record.actual == pytest.approx(0.05)
        assert rule.halted is True
        assert rule.trip_drawdown == pytest.approx(0.05)

    def test_halted_day_refuses_every_new_intent_persistently(self) -> None:
        rule = DailyLossHaltRule(0.05)
        gate = RiskGate((rule,))
        gate.review([buy()], make_view(equity=100_000.0, ts=at(DAY1, 10)))
        gate.review([buy()], make_view(equity=90_000.0, ts=at(DAY1, 14)))  # -10%

        # the rest of the day refuses everything, even after equity recovers
        # to a drawdown back under the limit — the breaker latches
        for hour, equity in ((14, 90_000.0), (15, 97_000.0)):
            view = make_view(equity=equity, ts=at(DAY1, hour))
            for draft in (buy(), sell(), buy("B", 700)):
                outcome = gate.review([draft], view)
                assert outcome.approved == ()
                assert outcome.rejections[0].rule == "daily_loss_halt"
        assert rule.halted is True

    def test_the_halt_rearms_on_the_next_trading_day(self) -> None:
        rule = DailyLossHaltRule(0.05)
        gate = RiskGate((rule,))
        gate.review([buy()], make_view(equity=100_000.0, ts=at(DAY1, 10)))
        gate.review([buy()], make_view(equity=90_000.0, ts=at(DAY1, 14)))
        assert rule.halted is True

        next_morning = make_view(equity=90_000.0, ts=at(DAY2, 10))
        outcome = gate.review([buy()], next_morning)
        assert outcome.approved == (buy(),)
        assert rule.halted is False
        assert rule.current_day == DAY2
        assert rule.day_start_equity == 90_000.0  # fresh reference for the day
        assert rule.trip_drawdown is None

    def test_day_start_equity_latches_at_the_first_observation(self) -> None:
        rule = DailyLossHaltRule(0.10)
        gate = RiskGate((rule,))
        # the day's first view already shows 90k — that, not yesterday's
        # equity, is the reference the drawdown is measured against
        gate.review([buy()], make_view(equity=90_000.0, ts=at(DAY1, 10)))
        assert rule.day_start_equity == 90_000.0
        verdict_view = make_view(equity=80_000.0, ts=at(DAY1, 14))  # -11.1%
        outcome = gate.review([buy()], verdict_view)
        assert outcome.approved == ()
        assert outcome.rejections[0].actual == pytest.approx(1 / 9)

    def test_non_positive_day_start_equity_halts_fail_safe(self) -> None:
        rule = DailyLossHaltRule(0.05)
        gate = RiskGate((rule,))
        outcome = gate.review(
            [buy()], make_view(equity=0.0, ts=at(DAY1, 10))
        )
        assert outcome.approved == ()
        assert outcome.rejections[0].rule == "daily_loss_halt"
        assert rule.halted is True

    def test_invalid_limit_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_daily_loss"):
            DailyLossHaltRule(0.0)
        with pytest.raises(ValueError, match="max_daily_loss"):
            DailyLossHaltRule(1.5)


class TestSymbolBlacklist:
    def test_off_list_buys_pass(self) -> None:
        rule = SymbolBlacklistRule(symbols={"600999"})
        view = make_view(ts=at(DAY1))
        assert rule.check(buy("600001"), view) is None

    def test_custom_exclusion_rejects_the_buy(self) -> None:
        rule = SymbolBlacklistRule(symbols={"600999"})
        verdict = rule.check(buy("600999"), make_view(ts=at(DAY1)))
        assert isinstance(verdict, RuleRejection)
        assert "custom exclusion list" in verdict.reason
        assert verdict.threshold == (
            "tradeable (not ST / not delisting / not excluded)"
        )
        assert verdict.actual is not None and "custom exclusion" in verdict.actual

    def test_st_names_are_rejected(self) -> None:
        rule = SymbolBlacklistRule(instruments={"600001": instrument(is_st=True)})
        verdict = rule.check(buy("600001"), make_view(ts=at(DAY1)))
        assert isinstance(verdict, RuleRejection)
        assert "ST" in verdict.reason

    def test_delisted_names_are_rejected(self) -> None:
        rule = SymbolBlacklistRule(
            instruments={"600001": instrument(status=InstrumentStatus.DELISTED)}
        )
        verdict = rule.check(buy("600001"), make_view(ts=at(DAY1)))
        assert isinstance(verdict, RuleRejection)
        assert "delisted" in verdict.reason

    def test_delisting_arrangement_period_is_rejected(self) -> None:
        rule = SymbolBlacklistRule(delisting_window_days=30)
        arranging = instrument(delist_date=DAY1 + timedelta(days=10))
        rule._instruments["600002"] = arranging  # noqa: SLF001 - fixture wiring
        verdict = rule.check(buy("600002"), make_view(ts=at(DAY1)))
        assert isinstance(verdict, RuleRejection)
        assert "delisting arrangement period" in verdict.reason

    def test_a_delist_date_beyond_the_window_still_trades(self) -> None:
        rule = SymbolBlacklistRule(delisting_window_days=30)
        far_out = instrument(delist_date=DAY1 + timedelta(days=60))
        rule._instruments["600003"] = far_out  # noqa: SLF001 - fixture wiring
        assert rule.check(buy("600003"), make_view(ts=at(DAY1))) is None

    def test_an_already_passed_delist_date_is_rejected(self) -> None:
        rule = SymbolBlacklistRule()
        gone = instrument(delist_date=DAY1 - timedelta(days=5))
        rule._instruments["600004"] = gone  # noqa: SLF001 - fixture wiring
        verdict = rule.check(buy("600004"), make_view(ts=at(DAY1)))
        assert isinstance(verdict, RuleRejection)
        assert "passed its delist date" in verdict.reason

    def test_sells_of_blacklisted_names_still_pass(self) -> None:
        rule = SymbolBlacklistRule(symbols={"600999"})
        view = make_view(ts=at(DAY1))
        assert rule.check(sell("600999"), view) is None

    def test_unknown_symbols_are_judged_on_the_custom_list_alone(self) -> None:
        rule = SymbolBlacklistRule(symbols={"600999"})
        assert rule.check(buy("000001"), make_view(ts=at(DAY1))) is None

    def test_invalid_window_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="delisting_window_days"):
            SymbolBlacklistRule(delisting_window_days=-1)


class TestLiquidityFloor:
    def test_buys_of_liquid_names_pass(self) -> None:
        rule = LiquidityFloorRule(min_amount=1_000_000.0)
        view = make_view(amounts={"A": 2_000_000.0})
        assert rule.check(buy(), view) is None

    def test_buys_of_thin_names_are_rejected(self) -> None:
        rule = LiquidityFloorRule(min_amount=1_000_000.0)
        view = make_view(amounts={"A": 999_999.0})
        verdict = rule.check(buy(), view)
        assert isinstance(verdict, RuleRejection)
        assert "below" in verdict.reason and "floor" in verdict.reason
        assert verdict.threshold == 1_000_000.0
        assert verdict.actual == 999_999.0

    def test_buys_without_a_turnover_observation_are_rejected(self) -> None:
        rule = LiquidityFloorRule(min_amount=1_000_000.0)
        verdict = rule.check(buy(), make_view())  # no amounts at all
        assert isinstance(verdict, RuleRejection)
        assert "no turnover observation" in verdict.reason
        assert verdict.threshold == 1_000_000.0
        assert verdict.actual is None

    def test_sells_pass_without_any_turnover_data(self) -> None:
        rule = LiquidityFloorRule(min_amount=1_000_000.0)
        assert rule.check(sell(), make_view()) is None

    def test_invalid_floor_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="min_amount"):
            LiquidityFloorRule(-1.0)


class TestStandardChain:
    def test_the_five_rules_run_in_canonical_order(self) -> None:
        chain = standard_risk_chain()
        assert [rule.name for rule in chain] == [
            "daily_loss_halt",
            "symbol_blacklist",
            "liquidity_floor",
            "single_position_cap",
            "portfolio_exposure_cap",
        ]

    def test_factory_parameters_reach_their_rules(self) -> None:
        chain = standard_risk_chain(
            single_position_cap=0.2,
            max_gross_exposure=0.8,
            max_daily_loss=0.03,
            min_turnover=5_000_000.0,
            excluded_symbols=("600999",),
        )
        assert isinstance(chain[0], DailyLossHaltRule)
        assert chain[0].max_daily_loss == 0.03
        assert isinstance(chain[1], SymbolBlacklistRule)
        assert chain[1].excluded_symbols == frozenset({"600999"})
        assert isinstance(chain[2], LiquidityFloorRule)
        assert chain[2].min_amount == 5_000_000.0
        assert isinstance(chain[3], SinglePositionCapRule)
        assert chain[3].max_weight == 0.2
        assert isinstance(chain[4], PortfolioExposureCapRule)
        assert chain[4].max_gross_weight == 0.8

    def test_earlier_rules_win_the_rejection(self) -> None:
        gate = RiskGate(standard_risk_chain(excluded_symbols={"600999"}))
        # the draft violates both the blacklist and the position cap; the
        # blacklist sits earlier in the chain, so it owns the rejection
        view = make_view(equity=100.0, prices={"600999": 10.0}, amounts={"600999": 1e9})
        outcome = gate.review([buy("600999", 100)], view)
        assert outcome.approved == ()
        assert outcome.rejections[0].rule == "symbol_blacklist"

    def test_a_clean_draft_clears_the_whole_chain(self) -> None:
        gate = RiskGate(standard_risk_chain())
        view = make_view(
            equity=10_000.0,
            prices={"A": 10.0},
            amounts={"A": 50_000_000.0},
        )
        outcome = gate.review([buy("A", 500)], view)
        assert outcome.approved == (buy("A", 500),)
        assert outcome.rejections == ()


class TestRecords:
    def test_rejection_record_carries_the_full_context(self) -> None:
        record = RejectionRecord(
            ts=TS,
            rule="single_position_cap",
            reason="too big",
            symbol="A",
            side=Side.BUY,
            quantity=100,
            threshold=0.5,
            actual=0.75,
        )
        assert record.side is Side.BUY
        assert record.ts == TS  # timestamps normalize to Asia/Shanghai
        assert record.threshold == 0.5
        assert record.actual == 0.75


def test_risk_rule_base_is_abstract() -> None:
    rule = RiskRule()
    with pytest.raises(NotImplementedError):
        rule.check(buy(), make_view())


# -- end-to-end: the five-rule chain through a real runtime loop -------------------


class AlwaysLong(StrategyBase):
    params = Params()

    def on_bar(self, ctx) -> None:
        ctx.target_weight(ctx.symbol, 1.0)


def run_with(
    port: Any,
    *,
    gate: RiskGate | None = None,
    strategy: StrategyBase | None = None,
    initial_cash: float = 100_000.0,
) -> tuple[StrategyRuntime, FillingExecutionPort]:
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    runtime = StrategyRuntime(
        bus=bus,
        port=venue,
        strategy=strategy if strategy is not None else AlwaysLong(),
        gate=gate,
        initial_cash=initial_cash,
    )
    symbols = [view.symbol for view in port.list_instruments(as_of=start)]
    session = ReplaySession(
        port=port,
        symbols=symbols,
        start=start,
        end=end,
        seed=11,
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    session.run()
    return runtime, venue


class TestStandardChainThroughRuntime:
    def test_runtime_defaults_to_the_standard_five_rule_chain(self) -> None:
        closes = [50.0 + index for index in range(5)]
        runtime, venue = run_with(ScriptedClosesPort({"600000": closes}))
        assert runtime.gate.rule_names == (
            "daily_loss_halt",
            "symbol_blacklist",
            "liquidity_floor",
            "single_position_cap",
            "portfolio_exposure_cap",
        )
        # and the chain lets an ordinary clean run trade as before
        assert [intent.side for intent in runtime.intents] == [Side.BUY]

    def test_liquidity_floor_rejects_thin_names_at_the_intent_exit(self) -> None:
        closes = [50.0 + index for index in range(5)]
        gate = RiskGate(standard_risk_chain(min_turnover=10**12))
        runtime, venue = run_with(
            ScriptedClosesPort({"600000": closes}), gate=gate
        )
        assert venue.submitted == []  # nothing reached the venue
        assert runtime.intents == ()
        assert runtime.rejections, "rejections must be recorded"
        for record in runtime.rejections:
            assert record.rule == "liquidity_floor"
            assert record.symbol == "600000"
            assert record.threshold == 10**12
            # the observed turnover is the bar's amount (close x 500,000)
            assert record.actual is not None and record.actual > 10**6

    def test_blacklist_rejects_excluded_names_at_the_intent_exit(self) -> None:
        closes = [50.0 + index for index in range(5)]
        gate = RiskGate(standard_risk_chain(excluded_symbols={"600000"}))
        runtime, venue = run_with(
            ScriptedClosesPort({"600000": closes}), gate=gate
        )
        assert venue.submitted == []
        assert runtime.rejections
        assert all(record.rule == "symbol_blacklist" for record in runtime.rejections)
        assert runtime.account.snapshot().positions == ()
        assert runtime.account.cash == 100_000.0

    def test_daily_loss_halt_trips_persists_and_rearms_next_day(self) -> None:
        port = _IntradayHaltingPort()
        runtime, venue = run_with(port, strategy=_HalfThenFull())

        # day 1, 10:00 — half-position buy goes through the whole chain
        # day 1, 14:00 and 15:00 — equity marked down > 5% intraday; every
        #   new intent is refused by the breaker, twice in a row (state)
        # day 2, 10:00 — the breaker re-armed and the buy clears again
        assert [intent.side for intent in runtime.intents] == [Side.BUY, Side.BUY]
        assert [intent.quantity for intent in runtime.intents] == [500, 600]
        assert len(venue.submitted) == 2

        assert len(runtime.rejections) == 2
        for record in runtime.rejections:
            assert record.rule == "daily_loss_halt"
            assert record.threshold == 0.05
            assert record.actual == pytest.approx(0.075)
            assert "halt active" in record.reason

        halt = runtime.gate.rules[0]
        assert isinstance(halt, DailyLossHaltRule)
        # the run finished on day 2: armed again against that day's equity
        assert halt.halted is False
        assert halt.current_day == DAY2
        assert halt.day_start_equity == 90_000.0


class _HalfThenFull(StrategyBase):
    """Half position on the first bar, full weight afterwards."""

    params = Params()

    def __init__(self) -> None:
        super().__init__()
        self._first = True

    def on_bar(self, ctx) -> None:
        if self._first:
            self._first = False
            ctx.target_weight(ctx.symbol, 0.5)
        else:
            ctx.target_weight(ctx.symbol, 1.0)


class _IntradayHaltingPort:
    """Three bars: two intraday marks on day 1, one on day 2.

    Day 1: 10:00 close 100 (buy 500 at half weight), 14:00 close 85 and
    15:00 close 84 — marked equity falls from 100k to 92.5k / 92k, a 7.5%
    intraday drawdown against the 5% breaker. Day 2: 10:00 close 80.
    Turnover (close x 1,000,000 shares) stays far above every floor so the
    halt is the only rule that can reject.
    """

    def __init__(self) -> None:
        symbol = "600000"
        rows: list[dict[str, Any]] = []
        for day, hour, close in (
            (DAY1, 10, 100.0),
            (DAY1, 14, 85.0),
            (DAY1, 15, 84.0),
            (DAY2, 10, 80.0),
        ):
            rows.append(
                {
                    "symbol": symbol,
                    "ts": at(day, hour),
                    "open": close,
                    "high": close * 1.01,
                    "low": close * 0.99,
                    "close": close,
                    "volume": 1_000_000.0,
                    "amount": round(close * 1_000_000.0, 2),
                    "adjust_factor": 1.0,
                    "quality": "ok",
                }
            )
        self._symbol = symbol
        self._frame = DataFrame(rows, columns=CANONICAL_COLUMNS)

    # -- MarketDataPort -------------------------------------------------------

    def list_instruments(self, as_of: date) -> list[Instrument]:
        return [instrument(self._symbol)]

    def fetch_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: Freq,
        adjust: AdjustMode,
    ) -> DataFrame:
        frame = self._frame
        mask = frame["ts"].map(lambda ts: start <= ts.date() <= end)
        return frame[mask].reset_index(drop=True)

    def fetch_corporate_actions(self, symbol: str) -> list[CorporateAction]:
        return []

    def calendar(self, start: date, end: date) -> list[date]:
        days = sorted({ts.date() for ts in self._frame["ts"]})
        return [day for day in days if start <= day <= end]

    def subscribe(
        self, symbols: list[str], on_snapshot: Callable[[Snapshot], None]
    ) -> Subscription:
        raise NotImplementedError("intraday test port carries no realtime feed")

    # -- helpers --------------------------------------------------------------

    def span(self) -> tuple[date, date]:
        days = self.calendar(date(2000, 1, 1), date(2100, 1, 1))
        return days[0], days[-1]

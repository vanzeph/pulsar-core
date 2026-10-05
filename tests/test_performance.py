"""Performance accounting: hand-computed scenario, edges and the report.

The main test builds a small known event stream (three trading days, two
fills with explicit fees and a slipped sell) and checks every metric
against arithmetic derived by hand in the comments — the module formulas
must never drift from these derivations.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime, time

import pytest
from pulsar_contracts import (
    Bar,
    ExecutionEvent,
    ExecutionEventType,
    Fill,
    Freq,
    OrderId,
    Side,
    SHANGHAI_TZ,
)

from pulsar_core import (
    EquityPoint,
    Event,
    EventKind,
    SessionPhase,
    build_metrics_report,
    compute_equity_curve,
    load_metrics_report,
)

# -- the known event stream -----------------------------------------------------
#
# Initial cash 100_000, symbol "600000", three trading days in June 2026.
#
#   D1 (06-01): bar closes at 10.0; we buy 1_000 shares at 10.0 paying
#               commission 5.00 + transfer fee 1.00 (stamp duty: buy side).
#   D2 (06-02): bar closes at 11.0 (no trade).
#   D3 (06-03): bar closes at 9.0; we sell the full 1_000 shares at 8.90 —
#               0.10 below the 9.0 reference the decision was made against —
#               paying commission 4.45 + stamp duty 8.90 + transfer fee 0.89.
#
# Hand-derived account states:
#
#   start   : cash 100_000.00, equity 100_000.00, nav 1.0
#   D1 close: cash = 100_000 - (10.0*1_000 + 5.00 + 1.00) = 89_994.00
#             position 1_000 marked at 10.0 -> market value 10_000.00
#             equity = 99_994.00, nav = 0.99994
#   D2 close: cash 89_994.00, market value 11_000.00
#             equity = 100_994.00, nav = 1.00994
#   D3 close: cash = 89_994.00 + 8.90*1_000 - (4.45 + 8.90 + 0.89)
#                                         = 98_879.76
#             flat -> equity = 98_879.76, nav = 0.9887976

D1, D2, D3 = (date(2026, 6, day) for day in (1, 2, 3))
INITIAL_CASH = 100_000.0

BUY_FILL = Fill(
    fill_id="fill-1",
    order_id=OrderId("ord-1"),
    symbol="600000",
    side=Side.BUY,
    price=10.0,
    quantity=1_000,
    commission=5.0,
    stamp_duty=0.0,
    transfer_fee=1.0,
    ts=datetime.combine(D1, time(), tzinfo=SHANGHAI_TZ),
)

SELL_FILL = Fill(
    fill_id="fill-2",
    order_id=OrderId("ord-2"),
    symbol="600000",
    side=Side.SELL,
    price=8.90,
    quantity=1_000,
    commission=4.45,
    stamp_duty=8.90,
    transfer_fee=0.89,
    ts=datetime.combine(D3, time(), tzinfo=SHANGHAI_TZ),
)


def bar(day: date, close: float) -> Bar:
    return Bar(
        symbol="600000",
        ts=datetime.combine(day, time(), tzinfo=SHANGHAI_TZ),
        freq=Freq.DAILY,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        volume=500_000.0,
        amount=close * 500_000.0,
    )


def fill_event(fill: Fill) -> ExecutionEvent:
    return ExecutionEvent(
        event_type=ExecutionEventType.FILL,
        order_id=fill.order_id,
        ts=fill.ts,
        fill=fill,
    )


def known_stream() -> list[Event]:
    started = datetime.combine(D1, time(), tzinfo=SHANGHAI_TZ)
    return [
        Event(
            kind=EventKind.SESSION, ts=started, session_phase=SessionPhase.STARTED
        ),
        Event.of_bar(bar(D1, 10.0)),
        Event.of_execution(fill_event(BUY_FILL)),
        Event.of_bar(bar(D2, 11.0)),
        Event.of_bar(bar(D3, 9.0)),
        Event.of_execution(fill_event(SELL_FILL)),
        Event(
            kind=EventKind.SESSION, ts=SELL_FILL.ts, session_phase=SessionPhase.FINISHED
        ),
    ]


class TestEquityCurve:
    def test_curve_matches_hand_derived_points(self) -> None:
        curve = compute_equity_curve(known_stream(), initial_cash=INITIAL_CASH)

        # start point: equity == initial cash, nav == 1.0
        assert curve[0] == EquityPoint(
            ts=datetime.combine(D1, time(), tzinfo=SHANGHAI_TZ),
            kind="start",
            cash=100_000.0,
            market_value=0.0,
            equity=100_000.0,
            nav=1.0,
        )
        kinds = [point.kind for point in curve]
        assert kinds == ["start", "eod", "eod", "eod"]
        assert [point.ts.date() for point in curve] == [D1, D1, D2, D3]

        # D1 close: cash 89_994.00 + 1_000 * 10.0 -> equity 99_994.00
        assert curve[1].cash == pytest.approx(89_994.0)
        assert curve[1].market_value == pytest.approx(10_000.0)
        assert curve[1].equity == pytest.approx(99_994.0)
        assert curve[1].nav == pytest.approx(0.99994)

        # D2 close: 1_000 * 11.0 -> equity 100_994.00
        assert curve[2].equity == pytest.approx(100_994.0)
        assert curve[2].nav == pytest.approx(1.00994)

        # D3 close: flat after the sell, equity == cash == 98_879.76
        assert curve[3].market_value == pytest.approx(0.0)
        assert curve[3].cash == pytest.approx(98_879.76)
        assert curve[3].equity == pytest.approx(98_879.76)
        assert curve[3].nav == pytest.approx(0.9887976)

    def test_replaying_the_same_stream_reproduces_the_curve(self) -> None:
        first = compute_equity_curve(known_stream(), initial_cash=INITIAL_CASH)
        second = compute_equity_curve(known_stream(), initial_cash=INITIAL_CASH)
        assert first == second

    def test_non_positive_initial_cash_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="initial_cash must be positive"):
            compute_equity_curve([], initial_cash=0.0)


class TestMetricsAgainstHandArithmetic:
    def test_every_metric_matches_the_hand_derivation(self) -> None:
        report = build_metrics_report(
            known_stream(), initial_cash=INITIAL_CASH, run_id="hand-run-0001"
        )
        metrics = report.metrics

        # navs: [1.0, 0.99994, 1.00994, 0.9887976]
        navs = [1.0, 0.99994, 1.00994, 0.9887976]
        # daily returns (first differences):
        #   r1 = 0.99994/1.0     - 1 = -0.00006
        #   r2 = 1.00994/0.99994 - 1 =  0.010000600036...
        #   r3 = 0.9887976/1.00994 - 1 = -0.02093455...
        returns = [later / earlier - 1.0 for earlier, later in zip(navs, navs[1:])]
        assert len(returns) == 3

        assert metrics.trading_days == 3
        assert metrics.fills == 2

        # total return: final nav - 1 = -0.0112024
        assert metrics.total_return == pytest.approx(navs[-1] - 1.0)
        assert metrics.total_return == pytest.approx(-0.0112024)

        # annualized: (1 + total) ** (252 / 3) - 1
        expected_annual = (1.0 + navs[-1] - 1.0) ** (252.0 / 3.0) - 1.0
        assert metrics.annual_return == pytest.approx(expected_annual)

        # sample stdev (ddof=1) of the three returns, scaled by sqrt(252)
        mean = sum(returns) / 3.0
        stdev = math.sqrt(
            sum((value - mean) ** 2 for value in returns) / (3 - 1)
        )
        assert metrics.annual_volatility == pytest.approx(stdev * math.sqrt(252.0))

        # sharpe: mean/stdev * sqrt(252) with a zero risk-free rate
        assert metrics.sharpe == pytest.approx(mean / stdev * math.sqrt(252.0))

        # max drawdown: peak nav 1.00994 (D2) -> trough 0.9887976 (D3)
        #   mdd = (1.00994 - 0.9887976) / 1.00994 = 0.0209345...
        assert metrics.max_drawdown == pytest.approx((1.00994 - 0.9887976) / 1.00994)
        assert metrics.max_drawdown_peak_ts is not None
        assert metrics.max_drawdown_peak_ts.date() == D2
        assert metrics.max_drawdown_trough_ts is not None
        assert metrics.max_drawdown_trough_ts.date() == D3

        # turnover (double-sided): buy 10.0*1_000 + sell 8.90*1_000 = 18_900
        assert metrics.buy_value == pytest.approx(10_000.0)
        assert metrics.sell_value == pytest.approx(8_900.0)
        assert metrics.turnover_value == pytest.approx(18_900.0)
        # average EOD equity = (99_994 + 100_994 + 98_879.76) / 3
        average_equity = (99_994.0 + 100_994.0 + 98_879.76) / 3.0
        assert metrics.turnover_ratio == pytest.approx(18_900.0 / average_equity)
        # annualized: ratio * 252 / 3
        assert metrics.turnover_annualized == pytest.approx(
            18_900.0 / average_equity * 252.0 / 3.0
        )

        assert metrics.final_equity == pytest.approx(98_879.76)
        assert metrics.final_nav == pytest.approx(0.9887976)

    def test_fee_attribution_matches_the_hand_derivation(self) -> None:
        report = build_metrics_report(
            known_stream(), initial_cash=INITIAL_CASH, run_id="hand-run-0001"
        )
        fees = report.fee_attribution

        # amounts: commission 5.00 + 4.45; stamp duty 8.90 (sell only);
        # transfer fee 1.00 + 0.89; slippage only on the slipped sell:
        #   buy  at 10.0 vs reference 10.0 -> 0.0
        #   sell at 8.9  vs reference  9.0 -> (9.0 - 8.9) * 1_000 = 100.0
        assert fees.commission == pytest.approx(9.45)
        assert fees.stamp_duty == pytest.approx(8.90)
        assert fees.transfer_fee == pytest.approx(1.89)
        assert fees.slippage == pytest.approx(100.0)
        assert fees.total == pytest.approx(9.45 + 8.90 + 1.89 + 100.0)
        assert fees.fills == 2

        # drags: each amount as a fraction of the 100_000 initial cash
        assert fees.drag_commission == pytest.approx(9.45e-5)
        assert fees.drag_stamp_duty == pytest.approx(8.90e-5)
        assert fees.drag_transfer_fee == pytest.approx(1.89e-5)
        assert fees.drag_slippage == pytest.approx(100.0e-5)
        assert fees.drag_total == pytest.approx(120.24e-5)

    def test_favorable_execution_gives_negative_slippage(self) -> None:
        # a buy filled 0.05 below its reference is favorable: -0.05 * 2_000
        fill = BUY_FILL.model_copy(update={"price": 9.95, "quantity": 2_000})
        stream = [
            Event.of_bar(bar(D1, 10.0)),
            Event.of_execution(fill_event(fill)),
        ]
        report = build_metrics_report(
            stream, initial_cash=INITIAL_CASH, run_id="favorable"
        )
        assert report.fee_attribution.slippage == pytest.approx(-100.0)

    def test_fill_without_prior_mark_contributes_zero_slippage(self) -> None:
        stream = [Event.of_execution(fill_event(BUY_FILL))]
        report = build_metrics_report(
            stream, initial_cash=INITIAL_CASH, run_id="unmarked"
        )
        assert report.fee_attribution.slippage == pytest.approx(0.0)


class TestEdges:
    def test_empty_stream_yields_a_flat_report(self) -> None:
        report = build_metrics_report([], initial_cash=INITIAL_CASH, run_id="empty")
        assert report.equity_curve == ()
        assert report.start is None and report.end is None
        assert report.metrics.trading_days == 0
        assert report.metrics.total_return == 0.0
        assert report.metrics.annual_return == 0.0
        assert report.metrics.annual_volatility == 0.0
        assert report.metrics.sharpe == 0.0
        assert report.metrics.max_drawdown == 0.0
        assert report.metrics.max_drawdown_peak_ts is None
        assert report.metrics.max_drawdown_trough_ts is None
        assert report.metrics.turnover_ratio == 0.0
        assert report.metrics.final_equity == pytest.approx(INITIAL_CASH)
        assert report.fee_attribution.total == 0.0

    def test_flat_market_without_trades(self) -> None:
        stream = [
            Event.of_bar(bar(D1, 10.0)),
            Event.of_bar(bar(D2, 10.0)),
            Event.of_bar(bar(D3, 10.0)),
        ]
        report = build_metrics_report(
            stream, initial_cash=INITIAL_CASH, run_id="flat"
        )
        # zero dispersion: sharpe defined as 0.0, no drawdown anywhere
        assert report.metrics.sharpe == 0.0
        assert report.metrics.max_drawdown == 0.0
        assert all(point.nav == pytest.approx(1.0) for point in report.equity_curve)

    def test_non_positive_periods_per_year_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="periods_per_year must be positive"):
            build_metrics_report(
                known_stream(),
                initial_cash=INITIAL_CASH,
                run_id="bad",
                periods_per_year=0.0,
            )

    def test_custom_periods_per_year_changes_annualization(self) -> None:
        daily = build_metrics_report(
            known_stream(), initial_cash=INITIAL_CASH, run_id="a"
        )
        custom = build_metrics_report(
            known_stream(),
            initial_cash=INITIAL_CASH,
            run_id="b",
            periods_per_year=244.0,
        )
        # same per-period data, different annualization factor
        assert daily.metrics.total_return == custom.metrics.total_return
        assert daily.metrics.annual_return != custom.metrics.annual_return
        assert custom.metrics.annual_return == pytest.approx(
            0.9887976 ** (244.0 / 3.0) - 1.0
        )


class TestReportDocument:
    def test_report_json_carries_the_stable_contract_shape(self, tmp_path) -> None:
        report = build_metrics_report(
            known_stream(),
            initial_cash=INITIAL_CASH,
            run_id="contract-run",
            journal_digest="deadbeef" * 8,
        )
        document = json.loads(report.to_json())

        assert document["schema_version"] == 1
        assert document["run_id"] == "contract-run"
        assert document["initial_cash"] == INITIAL_CASH
        assert document["journal_digest"] == "deadbeef" * 8
        # the three structured sections the UI consumes
        assert set(document) == {
            "schema_version",
            "run_id",
            "initial_cash",
            "start",
            "end",
            "journal_digest",
            "equity_curve",
            "metrics",
            "fee_attribution",
        }
        assert len(document["equity_curve"]) == 4
        assert set(document["equity_curve"][0]) == {
            "ts",
            "kind",
            "cash",
            "market_value",
            "equity",
            "nav",
        }
        assert "sharpe" in document["metrics"]
        assert "max_drawdown" in document["metrics"]
        assert "turnover_ratio" in document["metrics"]
        assert set(document["fee_attribution"]) >= {
            "commission",
            "stamp_duty",
            "transfer_fee",
            "slippage",
            "total",
            "drag_total",
        }

    def test_write_and_load_round_trip(self, tmp_path) -> None:
        report = build_metrics_report(
            known_stream(), initial_cash=INITIAL_CASH, run_id="roundtrip"
        )
        path = report.write(tmp_path / "nested" / "metrics_report.json")
        loaded = load_metrics_report(path)
        assert loaded == report
        assert loaded.to_json() == report.to_json()

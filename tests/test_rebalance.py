"""Diff calculation: targets vs holdings + in-flight, A-share lot rules."""

from __future__ import annotations

from pulsar_contracts import Side

from pulsar_core import (
    PositionView,
    TargetLeg,
    TargetPortfolio,
    compute_rebalance,
)


def leg_weight(symbol: str, weight: float) -> TargetLeg:
    return TargetLeg(symbol=symbol, weight=weight)


def leg_shares(symbol: str, shares: int) -> TargetLeg:
    return TargetLeg(symbol=symbol, shares=shares)


def portfolio(**legs: TargetLeg) -> TargetPortfolio:
    return TargetPortfolio(targets=dict(legs))


def held(symbol: str, quantity: int, available: int | None = None) -> PositionView:
    return PositionView(
        symbol=symbol,
        quantity=quantity,
        available_quantity=quantity if available is None else available,
        avg_cost=10.0,
    )


class TestBuySide:
    def test_buy_rounds_down_to_board_lots(self) -> None:
        # equity 10_000, price 40 -> 250 shares wanted -> buy 200
        result = compute_rebalance(
            portfolio(A=leg_weight("A", 1.0)),
            positions={},
            equity=10_000.0,
            prices={"A": 40.0},
            pending={},
        )
        assert [(d.side, d.symbol, d.quantity) for d in result.drafts] == [
            (Side.BUY, "A", 200)
        ]

    def test_buy_below_one_lot_produces_no_order_but_a_reason(self) -> None:
        result = compute_rebalance(
            portfolio(A=leg_weight("A", 1.0)),
            positions={},
            equity=3_500.0,  # / 40 -> 87 shares < one lot
            prices={"A": 40.0},
            pending={},
        )
        assert result.drafts == ()
        assert result.skipped[0].symbol == "A"
        assert "below one lot" in result.skipped[0].reason

    def test_exact_shares_leg_buys_whole_lots(self) -> None:
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 300)),
            positions={},
            equity=1_000_000.0,
            prices={"A": 10.0},
            pending={},
        )
        assert result.drafts[0].quantity == 300

    def test_no_price_skips_the_leg_with_reason(self) -> None:
        result = compute_rebalance(
            portfolio(A=leg_weight("A", 0.5)),
            positions={},
            equity=10_000.0,
            prices={},
            pending={},
        )
        assert result.drafts == ()
        assert "no last price" in result.skipped[0].reason

    def test_tiny_weight_that_cannot_afford_one_share_targets_zero(self) -> None:
        result = compute_rebalance(
            portfolio(A=leg_weight("A", 0.0), B=leg_weight("B", 0.000001)),
            positions={"A": held("A", 500)},
            equity=10_000.0,
            prices={"A": 10.0, "B": 50.0},
            pending={},
        )
        # A: explicit zero liquidates; B: 10000*1e-6/50 -> 0 shares, nothing held
        assert [(d.side, d.quantity) for d in result.drafts] == [(Side.SELL, 500)]


class TestSellSide:
    def test_reduce_toward_nonzero_target_rounds_to_lots(self) -> None:
        # current 1000, target 750 -> sell 250 -> rounded down to 200
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 750)),
            positions={"A": held("A", 1000)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={},
        )
        assert result.drafts[0] .side is Side.SELL
        assert result.drafts[0].quantity == 200

    def test_zero_target_liquidates_the_whole_position_including_odd_tail(self) -> None:
        # 1250 held (odd 50-share tail) -> full liquidation sells 1250
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 0)),
            positions={"A": held("A", 1250)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={},
        )
        assert result.drafts[0].quantity == 1250

    def test_tplus_one_availability_caps_the_sell(self) -> None:
        # 500 held but only 200 sellable today -> sell 200, remainder waits
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 0)),
            positions={"A": held("A", 500, available=200)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={},
        )
        assert result.drafts[0].quantity == 200

    def test_fully_blocked_sell_is_skipped_with_reason(self) -> None:
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 0)),
            positions={"A": held("A", 500, available=0)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={},
        )
        assert result.drafts == ()
        assert "T+1" in result.skipped[0].reason


class TestInFlight:
    def test_pending_buys_count_as_holdings(self) -> None:
        # target 300, pending buy 300 already in flight -> nothing to do
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 300)),
            positions={},
            equity=1_000_000.0,
            prices={"A": 10.0},
            pending={"A": 300},
        )
        assert result.drafts == ()

    def test_partial_pending_buy_top_up_is_lot_rounded(self) -> None:
        # target 500, pending buy 150 -> delta 350 -> buy 300
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 500)),
            positions={},
            equity=1_000_000.0,
            prices={"A": 10.0},
            pending={"A": 150},
        )
        assert result.drafts[0].quantity == 300

    def test_pending_sells_reserve_availability(self) -> None:
        # 500 held, pending sell 500 already in flight -> nothing to do
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 0)),
            positions={"A": held("A", 500)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={"A": -500},
        )
        assert result.drafts == ()

    def test_pending_sell_reduces_what_may_still_be_sold(self) -> None:
        # 500 held, 200 already being sold -> effective 300 left to liquidate
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 0)),
            positions={"A": held("A", 500)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={"A": -200},
        )
        assert result.drafts[0].quantity == 300


class TestDeterminismAndValidation:
    def test_drafts_iterate_symbols_in_sorted_order(self) -> None:
        result = compute_rebalance(
            portfolio(C=leg_shares("C", 100), A=leg_shares("A", 100), B=leg_shares("B", 100)),
            positions={},
            equity=1_000_000.0,
            prices={"A": 10.0, "B": 10.0, "C": 10.0},
            pending={},
        )
        assert [draft.symbol for draft in result.drafts] == ["A", "B", "C"]

    def test_identical_inputs_produce_identical_drafts(self) -> None:
        kwargs = dict(
            positions={"A": held("A", 1250)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={},
        )
        target = portfolio(A=leg_shares("A", 0))
        assert compute_rebalance(target, **kwargs) == compute_rebalance(target, **kwargs)

    def test_at_target_produces_nothing(self) -> None:
        result = compute_rebalance(
            portfolio(A=leg_shares("A", 500)),
            positions={"A": held("A", 500)},
            equity=100_000.0,
            prices={"A": 10.0},
            pending={},
        )
        assert result.drafts == ()
        assert result.skipped == ()

    def test_non_positive_lot_size_is_rejected(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="lot_size"):
            compute_rebalance(
                portfolio(A=leg_shares("A", 100)),
                positions={},
                equity=1_000.0,
                prices={"A": 10.0},
                pending={},
                lot_size=0,
            )

"""RiskGate: chain semantics and the built-in single-position cap."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pulsar_contracts import Side

from pulsar_core import (
    OrderDraft,
    PositionView,
    RejectionRecord,
    RiskGate,
    RiskRule,
    RiskView,
    SinglePositionCapRule,
)

TS = datetime(2026, 6, 1, tzinfo=timezone.utc)


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
    positions: dict[str, PositionView] | None = None,
    prices: dict[str, float] | None = None,
    pending: dict[str, int] | None = None,
) -> RiskView:
    return RiskView(
        ts=TS,
        equity=equity,
        cash=equity,
        positions=positions or {},
        last_prices=prices or {},
        pending=pending or {},
    )


class TestChainSemantics:
    def test_passing_drafts_keep_their_order(self) -> None:
        drafts = [
            OrderDraft(side=Side.BUY, symbol="A", quantity=100),
            OrderDraft(side=Side.SELL, symbol="B", quantity=200),
        ]
        outcome = RiskGate((_AlwaysPass(),)).review(drafts, make_view())
        assert outcome.approved == tuple(drafts)
        assert outcome.rejections == ()

    def test_rejected_drafts_are_dropped_with_reason_recorded(self) -> None:
        drafts = [
            OrderDraft(side=Side.BUY, symbol="A", quantity=100),
            OrderDraft(side=Side.BUY, symbol="B", quantity=200),
        ]
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
        outcome = gate.review(
            [OrderDraft(side=Side.BUY, symbol="A", quantity=100)], make_view()
        )
        assert outcome.rejections[0].rule == "first"

    def test_mixed_outcome_splits_approved_and_rejected(self) -> None:
        class RejectB:
            name = "reject_b"

            def check(self, draft: OrderDraft, view: RiskView) -> str | None:
                return "B is off limits" if draft.symbol == "B" else None

        drafts = [
            OrderDraft(side=Side.BUY, symbol="A", quantity=100),
            OrderDraft(side=Side.BUY, symbol="B", quantity=100),
        ]
        outcome = RiskGate((RejectB(),)).review(drafts, make_view())
        assert [draft.symbol for draft in outcome.approved] == ["A"]
        assert [record.symbol for record in outcome.rejections] == ["B"]

    def test_duplicate_rule_names_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="unique"):
            RiskGate((_AlwaysPass(), _AlwaysPass()))

    def test_rule_names_expose_the_chain_order(self) -> None:
        gate = RiskGate((_RejectWith("a", "x"), _AlwaysPass()))
        assert gate.rule_names == ("a", "always_pass")


class TestSinglePositionCap:
    def test_buy_within_cap_passes(self) -> None:
        rule = SinglePositionCapRule(0.5)
        view = make_view(equity=10_000.0, prices={"A": 10.0})
        draft = OrderDraft(side=Side.BUY, symbol="A", quantity=500)  # exactly 50%
        assert rule.check(draft, view) is None

    def test_buy_pushing_past_the_cap_is_rejected(self) -> None:
        rule = SinglePositionCapRule(0.5)
        view = make_view(equity=10_000.0, prices={"A": 10.0})
        draft = OrderDraft(side=Side.BUY, symbol="A", quantity=501)  # 50.1%
        reason = rule.check(draft, view)
        assert reason is not None and "exceeds cap" in reason

    def test_cap_accounts_for_holdings_and_pending_buys(self) -> None:
        rule = SinglePositionCapRule(0.5)
        view = make_view(
            equity=10_000.0,
            positions={
                "A": PositionView(symbol="A", quantity=300, available_quantity=300)
            },
            prices={"A": 10.0},
            pending={"A": 100},  # 400 committed already = 40%
        )
        assert rule.check(
            OrderDraft(side=Side.BUY, symbol="A", quantity=100), view
        ) is None  # exactly 50%
        reason = rule.check(
            OrderDraft(side=Side.BUY, symbol="A", quantity=101), view
        )
        assert reason is not None

    def test_sells_always_pass(self) -> None:
        rule = SinglePositionCapRule(0.1)
        view = make_view(
            equity=10_000.0,
            positions={
                "A": PositionView(symbol="A", quantity=900, available_quantity=900)
            },
            prices={"A": 10.0},
        )
        assert (
            rule.check(OrderDraft(side=Side.SELL, symbol="A", quantity=900), view)
            is None
        )

    def test_buy_without_a_price_is_rejected(self) -> None:
        rule = SinglePositionCapRule(1.0)
        reason = rule.check(
            OrderDraft(side=Side.BUY, symbol="A", quantity=100), make_view()
        )
        assert reason is not None and "no last price" in reason

    def test_buy_with_non_positive_equity_is_rejected(self) -> None:
        rule = SinglePositionCapRule(1.0)
        view = make_view(equity=0.0, prices={"A": 10.0})
        reason = rule.check(
            OrderDraft(side=Side.BUY, symbol="A", quantity=100), view
        )
        assert reason is not None and "non-positive equity" in reason

    def test_invalid_cap_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="max_weight"):
            SinglePositionCapRule(1.5)
        with pytest.raises(ValueError, match="max_weight"):
            SinglePositionCapRule(0.0)


class TestRecords:
    def test_rejection_record_carries_the_full_context(self) -> None:
        record = RejectionRecord(
            ts=TS,
            rule="single_position_cap",
            reason="too big",
            symbol="A",
            side=Side.BUY,
            quantity=100,
        )
        assert record.side is Side.BUY
        assert record.ts == TS  # timestamps normalize to Asia/Shanghai


def test_risk_rule_base_is_abstract() -> None:
    rule = RiskRule()
    with pytest.raises(NotImplementedError):
        rule.check(OrderDraft(side=Side.BUY, symbol="A", quantity=1), make_view())

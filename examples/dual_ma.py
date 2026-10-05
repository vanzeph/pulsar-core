"""Dual moving-average crossover — the reference example strategy.

Demonstrates the strategy contract end to end:

* declare parameters on the class (``Params(fast=5, slow=20)``) and let the
  run configuration override them;
* read indicators through the typed ``BarContext``;
* declare *targets* (``ctx.target_weight``), never place orders — the
  engine computes the difference, risk-checks it and emits intents.

The example is fully self-contained: it carries a synthetic market-data
port (deterministic down-up-down price path, weekday calendar) and a
minimal immediate-fill venue, so ``python examples/dual_ma.py`` runs a
complete Research-mode session offline. Real data sources and venues are
injected the same way by their own packages.

Requires the repo on ``sys.path`` (``python examples/dual_ma.py`` from the
checkout root, or ``pip install -e .``).
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import Any, Callable

from pandas import DataFrame
from pulsar_contracts import (
    AdjustMode,
    CancelResult,
    CorporateAction,
    ExecutionEvent,
    ExecutionEventType,
    Fill,
    Freq,
    Instrument,
    InstrumentStatus,
    MarketDataPort,
    OrderId,
    OrderIntent,
    OrderState,
    OrderStatus,
    Position,
    PriceMode,
    Side,
    Snapshot,
    Subscription,
    SHANGHAI_TZ,
)

from pulsar_core import (
    BacktestClock,
    BarContext,
    EventBus,
    Params,
    ReplaySession,
    RiskGate,
    StrategyBase,
    StrategyRuntime,
    standard_risk_chain,
)

SYMBOL = "600000"


# -- the strategy -----------------------------------------------------------------


class DualMAStrategy(StrategyBase):
    """Go long while the fast MA is above the slow MA, flat otherwise."""

    params = Params(fast=5, slow=20)

    def on_bar(self, ctx: BarContext) -> None:
        fast = ctx.sma("close", self.params.fast)
        slow = ctx.sma("close", self.params.slow)
        if ctx.cross_up(fast, slow):
            ctx.target_weight(ctx.symbol, 1.0)  # declare a target, not an order
        elif ctx.cross_down(fast, slow):
            ctx.target_weight(ctx.symbol, 0.0)


# -- synthetic market data (any MarketDataPort implementation plugs in here) -------


class SyntheticDailyBars:
    """Deterministic down-up-down price path on a weekday calendar."""

    def __init__(self, symbol: str = SYMBOL) -> None:
        self._symbol = symbol

    def _closes(self) -> list[float]:
        closes: list[float] = []
        price = 100.0
        for day in range(100):
            if day < 30:
                drift = -0.008  # decline: fast MA falls below slow MA
            elif day < 70:
                drift = +0.010  # rally: golden cross, stay long
            else:
                drift = -0.012  # slide: death cross, exit
            price = round(price * (1.0 + drift), 4)
            closes.append(price)
        return closes

    def list_instruments(self, as_of: date) -> list[Instrument]:
        return [
            Instrument(
                symbol=self._symbol,
                exchange="SSE",
                board="main",
                status=InstrumentStatus.LISTED,
                list_date=date(2000, 1, 1),
            )
        ]

    def fetch_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: Freq,
        adjust: AdjustMode,
    ) -> DataFrame:
        if self._symbol not in symbols:
            return DataFrame()
        rows: list[dict[str, Any]] = []
        volume_rng = random.Random("volume")
        closes = self._closes()
        for index, close in enumerate(closes):
            day = self._calendar_day(index)
            if not start <= day <= end:
                continue
            previous = closes[index - 1] if index else close
            high = round(max(previous, close) * 1.005, 4)
            low = round(min(previous, close) * 0.995, 4)
            volume = float(volume_rng.randrange(200_000, 400_000))
            rows.append(
                {
                    "symbol": self._symbol,
                    "ts": datetime.combine(day, time(), tzinfo=SHANGHAI_TZ),
                    "open": previous,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": volume,
                    "amount": round(close * volume, 2),
                    "adjust_factor": 1.0,
                    "quality": "ok",
                }
            )
        return DataFrame(rows)

    def _calendar_day(self, index: int) -> date:
        """The ``index``-th weekday counting from 2026-01-05 (a Monday)."""
        days = self.calendar(date(2026, 1, 5), date(2027, 1, 1))
        return days[index]

    def span(self) -> tuple[date, date]:
        """First and last trading day the synthesized path covers."""
        return self._calendar_day(0), self._calendar_day(len(self._closes()) - 1)

    def fetch_corporate_actions(self, symbol: str) -> list[CorporateAction]:
        return []

    def calendar(self, start: date, end: date) -> list[date]:
        days: list[date] = []
        day = start
        while day <= end:
            if day.weekday() < 5:
                days.append(day)
            day += timedelta(days=1)
        return days

    def subscribe(
        self, symbols: list[str], on_snapshot: Callable[[Snapshot], None]
    ) -> Subscription:
        raise NotImplementedError("synthetic port carries no realtime feed")


# -- minimal immediate-fill venue (any ExecutionPort implementation plugs in here) --


class ImmediateFillVenue:
    """Fills every intent fully at its limit price, zero fees.

    Fill timestamps come from the run's clock through ``now`` so the whole
    session stays deterministic. Idempotency: re-submitting a known key
    returns the original order id without a second fill.
    """

    def __init__(self, now: Callable[[], datetime]) -> None:
        self._now = now
        self._next_order = 0
        self._by_key: dict[str, OrderId] = {}
        self._quantities: dict[OrderId, int] = {}
        self.submitted: list[OrderIntent] = []
        self.fills: list[Fill] = []

    def on_event(self, callback: Callable[[ExecutionEvent], None]) -> None:
        self._callback = callback

    def submit(self, intent: OrderIntent) -> OrderId:
        key = intent.idempotency_key.to_str()
        known = self._by_key.get(key)
        if known is not None:
            return known
        self._next_order += 1
        order_id = OrderId(f"ord-{self._next_order:06d}")
        self._by_key[key] = order_id
        self._quantities[order_id] = intent.quantity
        self.submitted.append(intent)
        if intent.price_mode is PriceMode.LIMIT and intent.limit_price is not None:
            price = intent.limit_price
        else:  # pragma: no cover - the runtime emits limit intents
            raise ValueError("this venue only fills priced limit intents")
        fill = Fill(
            fill_id=f"fill-{self._next_order:06d}",
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            price=price,
            quantity=intent.quantity,
            ts=self._now(),
        )
        self.fills.append(fill)
        self._callback(
            ExecutionEvent(
                event_type=ExecutionEventType.FILL,
                order_id=order_id,
                ts=fill.ts,
                fill=fill,
            )
        )
        return order_id

    def cancel(self, order_id: OrderId) -> CancelResult:
        return CancelResult(
            order_id=order_id, accepted=False, reason="already filled"
        )

    def query(self, order_id: OrderId) -> OrderState:
        return OrderState(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=self._quantities.get(order_id, 0),
            avg_fill_price=self._avg_price(order_id),
            updated_at=self._now(),
        )

    def positions(self) -> list[Position]:
        net: dict[str, int] = {}
        cost: dict[str, float] = {}
        for fill in self.fills:
            signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
            net[fill.symbol] = net.get(fill.symbol, 0) + signed
            cost[fill.symbol] = cost.get(fill.symbol, 0.0) + fill.quantity * fill.price
        return [
            Position(
                symbol=symbol,
                quantity=quantity,
                available_quantity=quantity,
                avg_cost=(cost[symbol] / quantity) if quantity else None,
            )
            for symbol, quantity in sorted(net.items())
            if quantity > 0
        ]

    def _avg_price(self, order_id: OrderId) -> float | None:
        for fill in self.fills:
            if fill.order_id == order_id:
                return fill.price
        return None


# -- one Research-mode run -----------------------------------------------------------


def main() -> None:
    port = SyntheticDailyBars()
    start, end = port.span()  # the window exactly covered by the synthetic path

    bus = EventBus(BacktestClock(datetime.combine(start, time(), tzinfo=SHANGHAI_TZ)))
    venue = ImmediateFillVenue(now=lambda: bus.now)  # bus.now is a property

    strategy = DualMAStrategy({"fast": 5, "slow": 20})
    runtime = StrategyRuntime(
        bus=bus,
        port=venue,
        strategy=strategy,
        # the design's five pre-trade rules at standard parameters; a real
        # assembly passes its own thresholds via standard_risk_chain(...)
        gate=RiskGate(standard_risk_chain()),
        initial_cash=100_000.0,
    )
    session = ReplaySession(
        port=port,
        symbols=[SYMBOL],
        start=start,
        end=end,
        seed=7,
        config={
            "strategy": {"name": "dual_ma", "params": strategy.params.to_dict()},
            "engine": {"initial_cash": 100_000.0, "lot_size": 100},
        },
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )

    result = session.run()

    buys = [s.intent for s in runtime.submissions if s.intent.side is Side.BUY]
    sells = [s.intent for s in runtime.submissions if s.intent.side is Side.SELL]
    final = runtime.account.snapshot()
    print("run_id                :", result.run_id)
    print("journal digest        :", result.journal_digest[:16], "...")
    print("trading days / bars   :", result.trading_days, "/", result.bar_events)
    print("intents (buy / sell)  :", len(buys), "/", len(sells))
    print("risk rejections       :", len(runtime.rejections))
    for submission in runtime.submissions:
        intent = submission.intent
        print(
            f"  {intent.idempotency_key.to_str()}  {intent.side.value:>4}"
            f"  {intent.symbol}  x{intent.quantity}  @ {intent.limit_price}"
        )
    print("final cash / equity   : %.2f / %.2f" % (final.cash, final.equity))
    print("final positions       :", [p.model_dump() for p in final.positions])
    print("strategy state        :", runtime.state)


if __name__ == "__main__":
    main()

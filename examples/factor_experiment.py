"""Factor-model experiment — the zero-code research example.

Loads ``experiments/momentum_value.toml`` (pure configuration: factor
subset, preprocessing, modeler, portfolio, backtest window) and runs it
end to end over a synthetic five-symbol market — factor values → modeler
scores → target portfolio → the C2 Signal → RiskGate → OrderIntent
pipeline. Changing the TOML (different factors, another modeler, another
``top_n``) is a new experiment; no code changes.

With ``--sweep`` the companion ``experiments/momentum_value_sweep.toml``
runs instead: the template's two axes expand into a run family that
shares one experiment id while every run keeps its own run id.

The synthetic port and the immediate-fill venue mirror the dual-MA
example: real data sources and venues are injected exactly the same way
by their own packages.

Requires the repo on ``sys.path`` (``python examples/factor_experiment.py``
from the checkout root, or ``pip install -e .``).
"""

from __future__ import annotations

import random
import sys
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Callable

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

from pulsar_core import load_experiment, run_experiment, run_sweep

SYMBOLS = ("600000", "600009", "600016", "600028", "600030")

#: Distinct deterministic drifts so momentum ranks clearly.
_DRIFT = {
    "600000": 0.0040,
    "600009": 0.0020,
    "600016": 0.0000,
    "600028": -0.0020,
    "600030": -0.0040,
}


# -- synthetic multi-symbol daily bars (any MarketDataPort plugs in here) ----------


class SyntheticCrossSection:
    """~320 deterministic daily bars per symbol, distinct drift paths."""

    _EPOCH = date(2025, 7, 1)  # first generated trading day (a Tuesday)
    _BARS = 320

    def _trading_days(self) -> list[date]:
        days: list[date] = []
        day = self._EPOCH
        while len(days) < self._BARS:
            if day.weekday() < 5:
                days.append(day)
            day += timedelta(days=1)
        return days

    def _closes(self, symbol: str) -> list[float]:
        rng = random.Random(f"noise:{symbol}")
        price = 20.0
        closes: list[float] = []
        for _ in range(self._BARS):
            noise = rng.uniform(-0.006, 0.006)
            price = round(price * (1.0 + _DRIFT[symbol] + noise), 4)
            closes.append(price)
        return closes

    def list_instruments(self, as_of: date) -> list[Instrument]:
        return [
            Instrument(
                symbol=symbol,
                exchange="SSE",
                board="main",
                status=InstrumentStatus.LISTED,
                list_date=date(2000, 1, 1),
            )
            for symbol in SYMBOLS
        ]

    def fetch_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: Freq,
        adjust: AdjustMode,
    ) -> DataFrame:
        days = self._trading_days()
        rows: list[dict] = []
        for symbol in sorted(symbols):
            if symbol not in _DRIFT:
                continue
            closes = self._closes(symbol)
            for index, close in enumerate(closes):
                day = days[index]
                if not start <= day <= end:
                    continue
                previous = closes[index - 1] if index else close
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": datetime.combine(day, time(), tzinfo=SHANGHAI_TZ),
                        "open": previous,
                        "high": round(max(previous, close) * 1.003, 4),
                        "low": round(min(previous, close) * 0.997, 4),
                        "close": close,
                        "volume": 500_000.0,
                        "amount": round(close * 500_000.0, 2),
                        "adjust_factor": 1.0,
                        "quality": "ok",
                    }
                )
        return DataFrame(rows)

    def fetch_corporate_actions(self, symbol: str) -> list[CorporateAction]:
        return []

    def calendar(self, start: date, end: date) -> list[date]:
        return [day for day in self._trading_days() if start <= day <= end]

    def subscribe(
        self, symbols: list[str], on_snapshot: Callable[[Snapshot], None]
    ) -> Subscription:
        raise NotImplementedError("synthetic port carries no realtime feed")


# -- minimal immediate-fill venue (any ExecutionPort plugs in here) -----------------


class ImmediateFillVenue:
    """Fills every intent fully at its limit price, zero fees."""

    def __init__(self, now: Callable[[], datetime]) -> None:
        self._now = now
        self._orders = 0
        self._by_key: dict[str, OrderId] = {}
        self._callback: Callable[[ExecutionEvent], None] | None = None

    def on_event(self, callback: Callable[[ExecutionEvent], None]) -> None:
        self._callback = callback

    def submit(self, intent: OrderIntent) -> OrderId:
        key = intent.idempotency_key.to_str()
        known = self._by_key.get(key)
        if known is not None:
            return known
        assert self._callback is not None
        self._orders += 1
        order_id = OrderId(f"ord-{self._orders:06d}")
        self._by_key[key] = order_id
        if intent.price_mode is not PriceMode.LIMIT or intent.limit_price is None:
            raise ValueError("this venue only fills priced limit intents")
        fill = Fill(
            fill_id=f"fill-{self._orders:06d}",
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            price=intent.limit_price,
            quantity=intent.quantity,
            ts=self._now(),
        )
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
        return CancelResult(order_id=order_id, accepted=False, reason="already filled")

    def query(self, order_id: OrderId) -> OrderState:
        return OrderState(
            order_id=order_id,
            status=OrderStatus.FILLED,
            filled_quantity=0,
            avg_fill_price=None,
            updated_at=self._now(),
        )

    def positions(self) -> list[Position]:
        return []


# -- entry point ---------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    port = SyntheticCrossSection()
    make_venue: Callable = lambda bus: ImmediateFillVenue(now=lambda: bus.now)  # noqa: E731

    if "--sweep" in sys.argv:
        config = load_experiment(_repo_root() / "experiments" / "momentum_value_sweep.toml")
        report = run_sweep(config, port=port, make_venue=make_venue, initial_cash=200_000.0)
        print(f"experiment_id (shared): {report.experiment_id}")
        for result in report.runs:
            buys = sum(
                1 for s in result.runtime.submissions if s.intent.side is Side.BUY
            )
            sells = sum(
                1 for s in result.runtime.submissions if s.intent.side is Side.SELL
            )
            equity = result.runtime.account.snapshot().equity
            print(
                f"  run #{result.run_index + 1}  run_id={result.run_id}"
                f"  label=[{result.run_label}]"
                f"  intents(b/s)={buys}/{sells}"
                f"  equity={equity:,.2f}"
            )
        print(f"run family size: {len(report.runs)}; distinct run ids: {len(set(report.run_ids))}")
        return 0

    config = load_experiment(_repo_root() / "experiments" / "momentum_value.toml")
    result = run_experiment(config, port=port, venue=make_venue, initial_cash=200_000.0)

    print("experiment_id         :", result.experiment_id)
    print("run_id                :", result.run_id)
    print("rebalance days        :", [day.isoformat() for day in result.strategy.rebalance_days])
    print("factors               :", config.factor_names)
    print("intents (buy / sell)  :", end=" ")
    buys = [s.intent for s in result.runtime.submissions if s.intent.side is Side.BUY]
    sells = [s.intent for s in result.runtime.submissions if s.intent.side is Side.SELL]
    print(f"{len(buys)} / {len(sells)}")
    final = result.runtime.account.snapshot()
    print("final cash / equity   : %.2f / %.2f" % (final.cash, final.equity))
    print("final positions       :", [(p.symbol, p.quantity) for p in final.positions])
    print("risk rejections       :", len(result.runtime.rejections))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

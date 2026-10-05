"""Shared fixtures: deterministic in-memory ports for kernel tests.

``FakeMarketDataPort`` synthesizes bars from per-symbol ``random.Random``
instances seeded with stable strings, so two ports built with the same seed
serve byte-identical data — never the global ``random`` state, never wall
time. Rows are deliberately shuffled (with a fixed seed) to prove the
session does not rely on frame row order.

``ScriptedClosesPort`` serves one deterministic bar per configured close
price (down/up/down regimes), for tests that need guaranteed moving-average
crossings.

``FillingExecutionPort`` is the minimal Research-mode venue: every intent
fills fully at its limit price with zero fees, timestamps drawn from the
run's own clock so journals stay bit-reproducible.
"""

from __future__ import annotations

import random
from datetime import date, datetime, time, timedelta
from typing import Callable, Iterable

import pytest
from pandas import DataFrame
from pulsar_contracts import (
    AdjustMode,
    Bar,
    CancelResult,
    CorporateAction,
    ExecutionEvent,
    ExecutionEventType,
    Fill,
    Freq,
    Instrument,
    InstrumentStatus,
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


def trading_days(start: date, end: date) -> list[date]:
    """Weekday-only calendar (tests need no exchange holidays)."""
    days: list[date] = []
    day = start
    while day <= end:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


class FakeMarketDataPort:
    """Deterministic lake-style port serving synthesized daily bars."""

    def __init__(
        self,
        *,
        seed: int | str = 7,
        gap_dates: Iterable[date] = (),
        start_price: float = 10.0,
    ) -> None:
        self._seed = seed
        self._gap_dates = frozenset(gap_dates)
        self._start_price = start_price

    # -- MarketDataPort -----------------------------------------------------

    def list_instruments(self, as_of: date) -> list[Instrument]:
        return [
            Instrument(
                symbol=symbol,
                exchange="SSE",
                board="main",
                status=InstrumentStatus.LISTED,
                list_date=date(2000, 1, 1),
            )
            for symbol in ("000001", "600000")
        ]

    def fetch_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: Freq,
        adjust: AdjustMode,
    ) -> DataFrame:
        if freq is not Freq.DAILY:
            raise NotImplementedError("fake port serves daily bars only")
        rows: list[dict] = []
        for symbol in sorted(symbols):
            rng = random.Random(f"bars:{self._seed}:{symbol}")
            price = self._start_price
            for day in trading_days(start, end):
                if day in self._gap_dates:
                    continue  # simulate a partition gap for that day
                ret = rng.uniform(-0.03, 0.03)
                open_ = price
                close = round(price * (1.0 + ret), 4)
                high = round(max(open_, close) * (1.0 + rng.uniform(0.0, 0.01)), 4)
                low = round(min(open_, close) * (1.0 - rng.uniform(0.0, 0.01)), 4)
                volume = float(rng.randrange(10_000, 1_000_000))
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": datetime.combine(day, time(), tzinfo=SHANGHAI_TZ),
                        "open": open_,
                        "high": high,
                        "low": low,
                        "close": close,
                        "volume": volume,
                        "amount": round(close * volume, 2),
                        "adjust_factor": 1.0,
                        "quality": "ok",
                    }
                )
                price = close
        frame = DataFrame(rows, columns=CANONICAL_COLUMNS)
        # Deterministic shuffle: frame row order must not matter downstream.
        return frame.sample(frac=1.0, random_state=1234).reset_index(drop=True)

    def fetch_corporate_actions(self, symbol: str) -> list[CorporateAction]:
        return []

    def calendar(self, start: date, end: date) -> list[date]:
        return trading_days(start, end)

    def subscribe(
        self, symbols: list[str], on_snapshot: Callable[[Snapshot], None]
    ) -> Subscription:
        raise NotImplementedError("fake port carries no realtime feed")

    # -- helpers ------------------------------------------------------------

    def bars(
        self, symbols: list[str], start: date, end: date
    ) -> list[Bar]:
        """The exact bars ``fetch_bars`` serves, as validated Bar objects."""
        frame = self.fetch_bars(symbols, start, end, Freq.DAILY, AdjustMode.RAW)
        records = frame.sort_values(by=["ts", "symbol"], kind="stable").to_dict(
            orient="records"
        )
        return [Bar(freq=Freq.DAILY, **record) for record in records]


@pytest.fixture
def window() -> tuple[date, date]:
    """Two full ISO weeks: 2026-06-01 (Mon) .. 2026-06-12 (Fri)."""
    return date(2026, 6, 1), date(2026, 6, 12)


class ScriptedClosesPort:
    """One deterministic bar per configured close, on a weekday calendar.

    The path is derived from the closes themselves (open = previous close,
    high/low straddle open and close), so a test only spells out the closes
    it wants — e.g. a decline, a rally, a slide — and gets guaranteed
    moving-average crossings.
    """

    def __init__(self, closes_by_symbol: dict[str, list[float]]) -> None:
        self._closes = {symbol: list(closes) for symbol, closes in closes_by_symbol.items()}
        length = {len(closes) for closes in self._closes.values()}
        if len(length) > 1:
            raise ValueError("all symbols must carry the same number of closes")
        self._days = trading_days(date(2026, 6, 1), date(2027, 6, 1))[: max(length or {0})]

    # -- MarketDataPort ------------------------------------------------------

    def list_instruments(self, as_of: date) -> list[Instrument]:
        return [
            Instrument(
                symbol=symbol,
                exchange="SSE",
                board="main",
                status=InstrumentStatus.LISTED,
                list_date=date(2000, 1, 1),
            )
            for symbol in sorted(self._closes)
        ]

    def fetch_bars(
        self,
        symbols: list[str],
        start: date,
        end: date,
        freq: Freq,
        adjust: AdjustMode,
    ) -> DataFrame:
        rows: list[dict] = []
        for symbol in sorted(symbols):
            closes = self._closes.get(symbol)
            if not closes:
                continue
            for index, close in enumerate(closes):
                day = self._days[index]
                if not start <= day <= end:
                    continue
                previous = closes[index - 1] if index else close
                rows.append(
                    {
                        "symbol": symbol,
                        "ts": datetime.combine(day, time(), tzinfo=SHANGHAI_TZ),
                        "open": previous,
                        "high": round(max(previous, close) * 1.004, 4),
                        "low": round(min(previous, close) * 0.996, 4),
                        "close": close,
                        "volume": 500_000.0,
                        "amount": round(close * 500_000.0, 2),
                        "adjust_factor": 1.0,
                        "quality": "ok",
                    }
                )
        return DataFrame(rows, columns=CANONICAL_COLUMNS)

    def fetch_corporate_actions(self, symbol: str) -> list[CorporateAction]:
        return []

    def calendar(self, start: date, end: date) -> list[date]:
        return [day for day in self._days if start <= day <= end]

    def subscribe(
        self, symbols: list[str], on_snapshot: Callable[[Snapshot], None]
    ) -> Subscription:
        raise NotImplementedError("scripted port carries no realtime feed")

    # -- helpers ---------------------------------------------------------------

    def span(self) -> tuple[date, date]:
        """First and last trading day covered by the scripted closes."""
        return self._days[0], self._days[len(next(iter(self._closes.values()))) - 1]


def down_up_down(n_down: int, n_up: int, n_slide: int, start: float = 100.0) -> list[float]:
    """A deterministic close path: decline, rally, slide."""
    closes: list[float] = []
    price = start
    for phase, drift in ((n_down, -0.008), (n_up, 0.011), (n_slide, -0.012)):
        for _ in range(phase):
            price = round(price * (1.0 + drift), 4)
            closes.append(price)
    return closes


class FillingExecutionPort:
    """Minimal Research-mode venue: full immediate fills at the limit price.

    ``now`` must be a callable returning the run's kernel time (wire it to
    the bus: ``FillingExecutionPort(now=lambda: bus.now)``) so fill
    timestamps stay deterministic. Re-submitting a known idempotency key
    returns the original order id without a second fill, per the port
    contract. Fees are zero — fee models belong to real venues.
    """

    def __init__(self, now: Callable[[], datetime]) -> None:
        self._now = now
        self._callback: Callable[[ExecutionEvent], None] | None = None
        self._orders = 0
        self._by_key: dict[str, OrderId] = {}
        self._fills_by_order: dict[OrderId, list[Fill]] = {}
        self.submitted: list[OrderIntent] = []

    # -- ExecutionPort ---------------------------------------------------------

    def on_event(self, callback: Callable[[ExecutionEvent], None]) -> None:
        self._callback = callback

    def submit(self, intent: OrderIntent) -> OrderId:
        key = intent.idempotency_key.to_str()
        known = self._by_key.get(key)
        if known is not None:
            return known
        if self._callback is None:
            raise RuntimeError("no execution callback registered")
        self._orders += 1
        order_id = OrderId(f"ord-{self._orders:06d}")
        self._by_key[key] = order_id
        self._fills_by_order[order_id] = []
        self.submitted.append(intent)
        if intent.price_mode is not PriceMode.LIMIT or intent.limit_price is None:
            raise ValueError("filling stub requires priced limit intents")
        fill = Fill(
            fill_id=f"fill-{self._orders:06d}",
            order_id=order_id,
            symbol=intent.symbol,
            side=intent.side,
            price=intent.limit_price,
            quantity=intent.quantity,
            ts=self._now(),
        )
        self._fills_by_order[order_id].append(fill)
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
        fills = self._fills_by_order.get(order_id, [])
        quantity = sum(fill.quantity for fill in fills)
        avg = (
            sum(fill.price * fill.quantity for fill in fills) / quantity
            if quantity
            else None
        )
        return OrderState(
            order_id=order_id,
            status=OrderStatus.FILLED if quantity else OrderStatus.CREATED,
            filled_quantity=quantity,
            avg_fill_price=avg,
            updated_at=self._now(),
        )

    def positions(self) -> list[Position]:
        net: dict[str, int] = {}
        for fills in self._fills_by_order.values():
            for fill in fills:
                signed = fill.quantity if fill.side is Side.BUY else -fill.quantity
                net[fill.symbol] = net.get(fill.symbol, 0) + signed
        return [
            Position(
                symbol=symbol, quantity=quantity, available_quantity=quantity
            )
            for symbol, quantity in sorted(net.items())
            if quantity > 0
        ]

    # -- helpers ---------------------------------------------------------------

    def fills(self) -> list[Fill]:
        return [fill for fills in self._fills_by_order.values() for fill in fills]

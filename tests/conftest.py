"""Shared fixtures: a deterministic in-memory MarketDataPort.

The fake port synthesizes bars from per-symbol ``random.Random`` instances
seeded with stable strings, so two ports built with the same seed serve
byte-identical data — never the global ``random`` state, never wall time.
Rows are deliberately shuffled (with a fixed seed) to prove the session does
not rely on frame row order.
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
    CorporateAction,
    Freq,
    Instrument,
    InstrumentStatus,
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

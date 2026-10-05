"""Performance accounting computed by replaying the run's event stream.

Per the core-engine design ("绩效从事件流重放计算，不依赖任何通道实现，
回测与实盘共用同一套指标定义"):

* the only input is the run's dispatched event journal — MARKET events mark
  prices (bar close / snapshot last price), EXECUTION events with a fill
  payload move cash and positions, SESSION events delimit the run;
* bookkeeping reuses :class:`~pulsar_core.account.TradingAccount`, so
  performance accounting and decision-side accounting share one
  implementation and can never drift apart;
* one equity point is emitted per trading day (at the last event of the
  day) plus a ``start`` point when the journal opens with a session-started
  event — that sequence is the net-value (NAV) curve;
* metrics are plain documented formulas over the curve and the fill list:
  total / annualized return, annualized volatility, Sharpe, max drawdown,
  turnover and fee attribution (commission / stamp duty / transfer fee /
  slippage, each also expressed as a drag fraction of initial capital).

Everything here is a pure function of ``(events, initial_cash)``. Backtest
and live runs therefore produce comparable numbers, and identical inputs
reproduce identical reports bit for bit.

Formula conventions (all hand-checkable):

* ``nav = equity / initial_cash``;
* daily returns are first differences of consecutive NAV points;
* ``annual_return = (1 + total_return) ** (periods_per_year / n) - 1`` with
  ``n`` the number of return periods (0 periods -> 0.0);
* ``annual_volatility`` is the sample standard deviation (ddof=1) of daily
  returns scaled by ``sqrt(periods_per_year)`` (< 2 returns -> 0.0);
* ``sharpe = mean(r) / stdev(r) * sqrt(periods_per_year)`` with a zero
  risk-free rate by default (zero dispersion -> 0.0);
* ``max_drawdown`` is the largest relative NAV decline from a running peak
  (reported as a non-negative fraction, with peak/trough timestamps);
* turnover is double-sided: ``sum(fill price * quantity)`` over all fills
  divided by the mean end-of-day equity, plus an annualized variant;
* slippage cost of a fill is its adverse move against the last marked
  price of the same symbol observed before the fill in dispatch order —
  the price the decision was made against. It is signed: negative means
  favorable execution. Fills with no prior mark contribute zero.
"""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Literal, Sequence

from pydantic import Field

from pulsar_contracts import ContractModel, Fill, Side, Timestamp

from .account import TradingAccount
from .events import Event, EventKind, SessionPhase

__all__ = [
    "PERIODS_PER_YEAR",
    "EquityPoint",
    "PerformanceMetrics",
    "FeeAttribution",
    "MetricsReport",
    "compute_equity_curve",
    "build_metrics_report",
    "load_metrics_report",
]

#: Annualization factor: return periods per year. Daily equity points give
#: ``252`` periods, the near-universal convention (the A-share calendar
#: carries ~243 sessions; override per report if exactness matters).
PERIODS_PER_YEAR = 252.0

_REPORT_SCHEMA_VERSION = 1


class EquityPoint(ContractModel):
    """One point of the net-value curve.

    ``kind="start"`` marks the run's opening point (equity == initial
    cash, NAV == 1.0); ``kind="eod"`` marks a trading-day close, stamped
    with the day's last event timestamp.
    """

    ts: Timestamp
    kind: Literal["start", "eod"]
    cash: float
    market_value: float
    equity: float
    nav: float


class FeeAttribution(ContractModel):
    """Fee attribution: what costs dragged on the run, by component.

    Amounts are in CNY. ``slippage`` is the estimated adverse execution
    cost of fills against their pre-fill reference price — signed, negative
    means favorable — so ``total``/``drag_total`` (the net of all four
    components) may be negative when favorable execution outweighs the
    booked fees. Each ``drag_*`` field is the component's amount as a
    fraction of initial cash.
    """

    fills: int = Field(ge=0)
    commission: float = Field(ge=0)
    stamp_duty: float = Field(ge=0)
    transfer_fee: float = Field(ge=0)
    slippage: float = 0.0
    total: float
    drag_commission: float = Field(ge=0)
    drag_stamp_duty: float = Field(ge=0)
    drag_transfer_fee: float = Field(ge=0)
    drag_slippage: float = 0.0
    drag_total: float


class PerformanceMetrics(ContractModel):
    """Headline metrics of one run (see module docstring for formulas)."""

    trading_days: int = Field(ge=0)
    total_return: float
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float = Field(ge=0)
    max_drawdown_peak_ts: datetime | None = None
    max_drawdown_trough_ts: datetime | None = None
    buy_value: float = Field(ge=0)
    sell_value: float = Field(ge=0)
    turnover_value: float = Field(ge=0)
    turnover_ratio: float = Field(ge=0)
    turnover_annualized: float = Field(ge=0)
    fills: int = Field(ge=0)
    final_equity: float
    final_nav: float


class MetricsReport(ContractModel):
    """The run's metrics report — one of the three stable UI contracts.

    Structure: ``schema_version`` plus ``equity_curve`` / ``metrics`` /
    ``fee_attribution``. The JSON form is the input contract of the
    visualization layer, so the layout is append-only across versions.
    """

    schema_version: int = Field(default=_REPORT_SCHEMA_VERSION)
    run_id: str = Field(min_length=1)
    initial_cash: float = Field(gt=0)
    start: datetime | None = None
    end: datetime | None = None
    journal_digest: str | None = None
    equity_curve: tuple[EquityPoint, ...] = ()
    metrics: PerformanceMetrics
    fee_attribution: FeeAttribution

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        """Deterministic JSON document of the whole report."""
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )

    def write(self, path: str | Path) -> Path:
        """Write the report document to ``path`` (parent dirs created)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target


def load_metrics_report(path: str | Path) -> MetricsReport:
    """Load a report from its JSON document (fails loudly when malformed)."""
    report: MetricsReport = MetricsReport.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    return report


# -- replay -----------------------------------------------------------------


def _replay_events(
    events: Iterable[Event], initial_cash: float
) -> tuple[tuple[EquityPoint, ...], tuple[tuple[Fill, float | None], ...]]:
    """One deterministic pass over the journal.

    Returns the equity curve and every fill paired with the reference price
    (the symbol's last mark before the fill, ``None`` when never marked).
    """
    if initial_cash <= 0:
        raise ValueError(f"initial_cash must be positive, got {initial_cash}")
    account = TradingAccount(cash=initial_cash)
    points: list[EquityPoint] = []
    fills: list[tuple[Fill, float | None]] = []
    day: date | None = None
    last_ts: datetime | None = None

    def record(ts: datetime, kind: Literal["start", "eod"]) -> None:
        equity = account.equity
        points.append(
            EquityPoint(
                ts=ts,
                kind=kind,
                cash=account.cash,
                market_value=equity - account.cash,
                equity=equity,
                nav=equity / initial_cash,
            )
        )

    for event in events:
        if day is None:
            day = event.ts.date()
        elif event.ts.date() != day:
            assert last_ts is not None  # day was set together with last_ts
            record(last_ts, "eod")
            day = event.ts.date()
        last_ts = event.ts

        if event.kind is EventKind.SESSION:
            if event.session_phase is SessionPhase.STARTED and not points:
                record(event.ts, "start")
        elif event.kind is EventKind.MARKET:
            if event.bar is not None:
                account.mark_price(event.bar.symbol, event.bar.close)
            elif event.snapshot is not None:
                account.mark_price(event.snapshot.symbol, event.snapshot.last_price)
        elif event.kind is EventKind.EXECUTION:
            execution = event.execution
            if execution is not None and execution.fill is not None:
                fills.append(
                    (execution.fill, account.last_price(execution.fill.symbol))
                )
                account.apply_fill(execution.fill)
        # TIMER events carry no accounting effect.

    if day is not None:
        assert last_ts is not None
        record(last_ts, "eod")
    return tuple(points), tuple(fills)


def compute_equity_curve(
    events: Iterable[Event], *, initial_cash: float
) -> tuple[EquityPoint, ...]:
    """Replay the event stream and return the equity (NAV) curve."""
    return _replay_events(events, initial_cash)[0]


# -- metrics ----------------------------------------------------------------


def _daily_returns(points: Sequence[EquityPoint]) -> list[float]:
    """First differences of consecutive NAV points.

    A preceding NAV of exactly zero means the account was wiped out; the
    following period is defined as a total loss (``-1.0``).
    """
    returns: list[float] = []
    for earlier, later in zip(points, points[1:]):
        if earlier.nav <= 0.0:
            returns.append(-1.0)
        else:
            returns.append(later.nav / earlier.nav - 1.0)
    return returns


def _sample_stdev(values: Sequence[float]) -> float:
    """Sample standard deviation (ddof=1); 0.0 for fewer than two values."""
    n = len(values)
    if n < 2:
        return 0.0
    mean = math.fsum(values) / n
    variance = math.fsum((value - mean) ** 2 for value in values) / (n - 1)
    return math.sqrt(variance)


def _max_drawdown(
    points: Sequence[EquityPoint],
) -> tuple[float, datetime | None, datetime | None]:
    """Largest relative NAV decline from a running peak, with its window.

    Returns ``(drawdown, peak_ts, trough_ts)``; timestamps are ``None``
    when the curve never drew down.
    """
    drawdown = 0.0
    peak_ts: datetime | None = None
    trough_ts: datetime | None = None
    peak = 0.0
    current_peak_ts: datetime | None = None
    for index, point in enumerate(points):
        if index == 0 or point.nav > peak:
            peak = point.nav
            current_peak_ts = point.ts
        decline = (peak - point.nav) / peak if peak > 0.0 else 0.0
        if decline > drawdown:
            drawdown = decline
            peak_ts = current_peak_ts
            trough_ts = point.ts
    return drawdown, peak_ts, trough_ts


def _slippage_cost(fill: Fill, reference: float | None) -> float:
    """Adverse (signed) execution cost of one fill against ``reference``."""
    if reference is None or reference <= 0.0:
        return 0.0
    if fill.side is Side.BUY:
        return float((fill.price - reference) * fill.quantity)
    return float((reference - fill.price) * fill.quantity)


def build_metrics_report(
    events: Iterable[Event],
    *,
    initial_cash: float,
    run_id: str,
    journal_digest: str | None = None,
    periods_per_year: float = PERIODS_PER_YEAR,
) -> MetricsReport:
    """Replay ``events`` and derive the full metrics report.

    ``journal_digest`` (from :attr:`pulsar_core.bus.EventBus.journal_digest`
    or :attr:`pulsar_core.session.RunResult.journal_digest`) ties the report
    to the exact event stream it was computed from.
    """
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be positive, got {periods_per_year}")
    points, fills = _replay_events(events, initial_cash)
    returns = _daily_returns(points)
    periods = len(returns)

    # -- return side ---------------------------------------------------------
    final_nav = points[-1].nav if points else 1.0
    final_equity = points[-1].equity if points else initial_cash
    total_return = final_nav - 1.0
    if periods == 0 or final_nav <= 0.0:
        annual_return = 0.0 if periods == 0 else -1.0
    else:
        annual_return = (1.0 + total_return) ** (periods_per_year / periods) - 1.0

    stdev = _sample_stdev(returns)
    annual_volatility = stdev * math.sqrt(periods_per_year)
    mean_return = math.fsum(returns) / periods if periods else 0.0
    sharpe = (
        mean_return / stdev * math.sqrt(periods_per_year) if stdev > 0.0 else 0.0
    )
    drawdown, peak_ts, trough_ts = _max_drawdown(points)

    # -- trading side --------------------------------------------------------
    buy_value = math.fsum(
        fill.price * fill.quantity for fill, _ in fills if fill.side is Side.BUY
    )
    sell_value = math.fsum(
        fill.price * fill.quantity for fill, _ in fills if fill.side is Side.SELL
    )
    turnover_value = buy_value + sell_value
    eod_equities = [point.equity for point in points if point.kind == "eod"]
    average_equity = (
        math.fsum(eod_equities) / len(eod_equities) if eod_equities else initial_cash
    )
    turnover_ratio = (
        turnover_value / average_equity if average_equity > 0.0 else 0.0
    )
    turnover_annualized = (
        turnover_ratio * (periods_per_year / periods) if periods else 0.0
    )

    # -- fee attribution -----------------------------------------------------
    commission = math.fsum(fill.commission for fill, _ in fills)
    stamp_duty = math.fsum(fill.stamp_duty for fill, _ in fills)
    transfer_fee = math.fsum(fill.transfer_fee for fill, _ in fills)
    slippage = math.fsum(_slippage_cost(fill, ref) for fill, ref in fills)
    total_fees = commission + stamp_duty + transfer_fee + slippage

    metrics = PerformanceMetrics(
        trading_days=sum(1 for point in points if point.kind == "eod"),
        total_return=total_return,
        annual_return=annual_return,
        annual_volatility=annual_volatility,
        sharpe=sharpe,
        max_drawdown=drawdown,
        max_drawdown_peak_ts=peak_ts,
        max_drawdown_trough_ts=trough_ts,
        buy_value=buy_value,
        sell_value=sell_value,
        turnover_value=turnover_value,
        turnover_ratio=turnover_ratio,
        turnover_annualized=turnover_annualized,
        fills=len(fills),
        final_equity=final_equity,
        final_nav=final_nav,
    )
    attribution = FeeAttribution(
        fills=len(fills),
        commission=commission,
        stamp_duty=stamp_duty,
        transfer_fee=transfer_fee,
        slippage=slippage,
        total=total_fees,
        drag_commission=commission / initial_cash,
        drag_stamp_duty=stamp_duty / initial_cash,
        drag_transfer_fee=transfer_fee / initial_cash,
        drag_slippage=slippage / initial_cash,
        drag_total=total_fees / initial_cash,
    )
    return MetricsReport(
        schema_version=_REPORT_SCHEMA_VERSION,
        run_id=run_id,
        initial_cash=initial_cash,
        start=points[0].ts if points else None,
        end=points[-1].ts if points else None,
        journal_digest=journal_digest,
        equity_curve=points,
        metrics=metrics,
        fee_attribution=attribution,
    )

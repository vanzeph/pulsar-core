"""The factor pipeline: bars -> factors -> preprocess -> model -> weights.

This module assembles the research-side pipeline of the core-engine
design (universe -> 因子计算 -> 预处理 -> 模型器 -> 打分 -> 组合构建)
on top of the C2 strategy framework:

* :func:`rebalance_dates` picks decision dates from the trading calendar
  (``daily`` / ``weekly`` / ``monthly`` / ``quarterly``);
* :class:`FactorEngine` computes, per decision date, the cross-sectional
  factor panel (oriented values, then the configured preprocess steps in
  order), hands it to the modeler for scoring and the portfolio
  constructor for target weights;
* :class:`FactorModelStrategy` is the :class:`~pulsar_core.strategy.StrategyBase`
  adapter that replays those precomputed weights through the live intent
  pipeline — each symbol declares its own target weight on its own bar,
  and the existing Signal -> TargetPortfolio -> RiskGate -> OrderIntent
  machinery does the rest.

Precomputation is not look-ahead: factor values at date ``t`` are built
from bars with ``ts <= t`` only, and the strategy declares them during
the replay of ``t`` — exactly what an incremental computation would see.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import date
from typing import Any, Mapping, Sequence

from pulsar_contracts import Bar

from .factors import FactorDefinition, factor_value
from .modelers import CrossSection, FactorHistoryView, ModelScorer
from .portfolio import PortfolioConstructor
from .preprocess import PreprocessStep
from .strategy import BarContext, StrategyBase

__all__ = [
    "REBALANCE_FREQUENCIES",
    "rebalance_dates",
    "required_warmup",
    "FactorEngine",
    "FactorModelStrategy",
]

#: Rebalance frequencies understood by the experiment runner.
REBALANCE_FREQUENCIES: tuple[str, ...] = ("daily", "weekly", "monthly", "quarterly")


def rebalance_dates(trading_days: Sequence[date], frequency: str) -> list[date]:
    """Decision dates within ``trading_days`` for ``frequency``.

    Weekly / monthly / quarterly pick the *first* trading day of each
    ISO week / calendar month / calendar quarter respectively; ``daily``
    keeps every trading day. Output preserves input order.
    """
    if frequency not in REBALANCE_FREQUENCIES:
        raise ValueError(
            f"unknown rebalance frequency {frequency!r}; "
            f"expected one of {REBALANCE_FREQUENCIES}"
        )
    selected: list[date] = []
    seen: set[tuple[int, ...]] = set()
    for day in trading_days:
        if frequency == "daily":
            key: tuple[int, ...] = (day.year, day.month, day.day)
        elif frequency == "weekly":
            iso = day.isocalendar()
            key = (iso[0], iso[1])
        elif frequency == "monthly":
            key = (day.year, day.month)
        else:
            key = (day.year, (day.month - 1) // 3 + 1)
        if key not in seen:
            seen.add(key)
            selected.append(day)
    return selected


def required_warmup(
    factors: Sequence[FactorDefinition], model: ModelScorer
) -> int:
    """Bars of pre-start history the pipeline needs (factors + model)."""
    factor_need = max((factor.min_bars for factor in factors), default=0)
    return factor_need + model.warmup_bars


class _EngineView(FactorHistoryView):
    """The engine's history access, clipped at one decision date."""

    def __init__(
        self,
        engine: "FactorEngine",
        *,
        as_of: date,
        factor_names: tuple[str, ...],
    ) -> None:
        self._engine = engine
        self._as_of = as_of
        self._factor_names = factor_names

    def factor_names(self) -> tuple[str, ...]:
        return self._factor_names

    def common_dates(self) -> tuple[date, ...]:
        return self._engine._dates_up_to(self._as_of)

    def factor_values(self, factor: str, as_of: date) -> Mapping[str, "float | None"]:
        return self._engine._raw_oriented(factor, as_of)

    def preprocessed_values(
        self, factor: str, as_of: date
    ) -> Mapping[str, "float | None"]:
        # the same chain _section_at applies, so ML modelers train and score
        # on one consistent panel (never a raw/preprocessed mix)
        column = self._engine._raw_oriented(factor, as_of)
        for step in self._engine._preprocess:
            column = step.apply(column)
        return column

    def forward_returns(self, from_date: date, horizon: int) -> Mapping[str, float]:
        return self._engine._forward_returns(from_date, horizon)


class FactorEngine:
    """Computes target weights per decision date from bar history.

    ``bars`` maps symbol → that symbol's bars (oldest first); the engine
    never mutates them and slices by date for every computation so each
    decision date sees exactly its own past.
    """

    def __init__(
        self,
        *,
        symbols: Sequence[str],
        bars: Mapping[str, Sequence[Bar]],
        factors: Sequence[FactorDefinition],
        preprocess: Sequence[PreprocessStep],
        model: ModelScorer,
        portfolio: PortfolioConstructor,
    ) -> None:
        self._symbols = tuple(sorted(symbols))
        self._bars: dict[str, tuple[Bar, ...]] = {
            symbol: tuple(bars.get(symbol, ())) for symbol in self._symbols
        }
        self._factors = tuple(factors)
        self._preprocess = tuple(preprocess)
        self._model = model
        self._portfolio = portfolio
        self._factor_names = tuple(factor.name for factor in self._factors)
        self._dates: tuple[date, ...] = tuple(
            sorted({bar.ts.date() for series in self._bars.values() for bar in series})
        )
        self._symbol_dates: dict[str, list[date]] = {
            symbol: [bar.ts.date() for bar in series]
            for symbol, series in self._bars.items()
        }
        self._date_index: dict[str, dict[date, int]] = {
            symbol: {bar.ts.date(): index for index, bar in enumerate(series)}
            for symbol, series in self._bars.items()
        }

    # -- public surface ---------------------------------------------------------

    def targets(self, rebalance_schedule: Sequence[date]) -> dict[date, dict[str, float]]:
        """Target weights ``{symbol: weight}`` for every scheduled date.

        Weights cover every universe symbol: unselected — and unscorable
        (missing data) — symbols carry ``0.0``, so holdings flow out of
        names the model no longer prefers at the next rebalance.
        """
        targets: dict[date, dict[str, float]] = {}
        for day in rebalance_schedule:
            weights = self._weights_at(day)
            targets[day] = {
                symbol: weights.get(symbol, 0.0) for symbol in self._symbols
            }
        return targets

    # -- internals -----------------------------------------------------------------

    def _weights_at(self, day: date) -> dict[str, float]:
        section = self._section_at(day)
        view = _EngineView(self, as_of=day, factor_names=self._factor_names)
        scores = self._model.score(section, view)
        return self._portfolio.construct(scores)

    def _section_at(self, day: date) -> CrossSection:
        columns: dict[str, dict[str, "float | None"]] = {}
        for factor in self._factors:
            column: dict[str, "float | None"] = self._raw_oriented(factor.name, day)
            for step in self._preprocess:
                column = step.apply(column)
            columns[factor.name] = column
        return CrossSection(as_of=day, values=columns)

    def _raw_oriented(self, factor_name: str, as_of: date) -> dict[str, "float | None"]:
        factor = self._factor_by_name(factor_name)
        values: dict[str, "float | None"] = {}
        for symbol in self._symbols:
            series = self._bars_up_to(symbol, as_of)
            values[symbol] = factor_value(factor, series)
        return values

    def _factor_by_name(self, name: str) -> FactorDefinition:
        for factor in self._factors:
            if factor.name == name:
                return factor
        raise KeyError(name)  # pragma: no cover - engine is built from its factors

    def _bars_up_to(self, symbol: str, as_of: date) -> tuple[Bar, ...]:
        series = self._bars[symbol]
        count = bisect_right(self._symbol_dates[symbol], as_of)
        return series[:count]

    def _dates_up_to(self, as_of: date) -> tuple[date, ...]:
        return tuple(day for day in self._dates if day <= as_of)

    def _forward_returns(self, from_date: date, horizon: int) -> dict[str, float]:
        returns: dict[str, float] = {}
        for symbol in self._symbols:
            series = self._bars[symbol]
            index = self._date_index[symbol].get(from_date)
            if index is None or index + horizon >= len(series):
                continue
            base = series[index].close
            ahead = series[index + horizon].close
            if base > 0:
                returns[symbol] = ahead / base - 1.0
        return returns


class FactorModelStrategy(StrategyBase):
    """Replays precomputed factor-model targets through the intent pipeline.

    On every bar the strategy checks whether the bar's date is a
    scheduled rebalance date; if it is, the symbol declares its own
    target weight from the precomputed table (``0.0`` when the model
    does not select it). Declarations then flow through the standard C2
    pipeline — portfolio build, diff calculation, risk gate, intent
    emission — exactly like any hand-written strategy.
    """

    params = StrategyBase.params  # no parameters: the table is the model

    def __init__(
        self,
        targets: Mapping[date, Mapping[str, float]],
        params: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(params)
        self._targets: dict[date, dict[str, float]] = {
            day: dict(weights) for day, weights in targets.items()
        }

    @property
    def rebalance_days(self) -> tuple[date, ...]:
        """Scheduled dates this strategy acts on, sorted."""
        return tuple(sorted(self._targets))

    def on_bar(self, ctx: BarContext) -> None:
        weights = self._targets.get(ctx.bar.ts.date())
        if weights is None:
            return
        ctx.target_weight(ctx.symbol, float(weights.get(ctx.symbol, 0.0)))

"""pulsar-core: the deterministic engine core of the Pulsar quant system.

This package implements the single-threaded deterministic event kernel, the
Clock abstraction that lets backtest and realtime runs share one loop, the
bar-level historical replay session, the RunManifest reproducibility
mechanism, the strategy framework with its Signal -> TargetPortfolio ->
RiskGate -> OrderIntent pipeline guarded by the five-rule pre-trade risk
chain (single-position cap, gross exposure cap, daily-loss halt, symbol
blacklist, liquidity floor), and performance accounting with its run
artifacts (events.parquet archive + metrics report).

Dependency policy (architecture baseline): pulsar-core depends only on
pulsar-contracts plus basic libraries. It must never import a data-source or
broker SDK, a network client, or any sibling pulsar package — a static
import-purity test enforces this.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bus import EventBus, Handler, canonical_event_json
from .clock import BacktestClock, Clock, RealtimeClock
from .errors import ClockWentBackwardsError, DataGapError, PulsarCoreError
from .events import Event, EventKind, SessionPhase, TimerPayload
from .manifest import RunManifest, bars_watermark, code_version, load_manifest
from .session import ReplaySession, RunResult

# -- strategy framework & intent pipeline (C2) --------------------------------

from .account import PortfolioView, PositionView, TradingAccount
from .params import BoundParams, Param, Params
from .rebalance import (
    DEFAULT_LOT_SIZE,
    OrderDraft,
    RebalanceResult,
    SkippedLeg,
    compute_rebalance,
)
from .risk import (
    DailyLossHaltRule,
    GateOutcome,
    LiquidityFloorRule,
    PortfolioExposureCapRule,
    RejectionRecord,
    RiskGate,
    RiskRule,
    RiskView,
    RuleRejection,
    SinglePositionCapRule,
    SymbolBlacklistRule,
    standard_risk_chain,
)
from .runtime import DEFAULT_HISTORY_DEPTH, StrategyRuntime, Submission
from .signals import (
    EqualWeightBuilder,
    PassThroughBuilder,
    PortfolioBuilder,
    Signal,
    TargetLeg,
    TargetPortfolio,
)
from .strategy import (
    FIELD_NAMES,
    BaseContext,
    BarContext,
    FillContext,
    IndicatorValue,
    StrategyBase,
    TickContext,
)

# -- performance accounting & run artifacts (C5) -------------------------------

from .artifacts import (
    EVENTS_FILENAME,
    EVENTS_SCHEMA_VERSION,
    MANIFEST_FILENAME,
    METRICS_FILENAME,
    RunArtifacts,
    read_event_archive,
    write_event_archive,
    write_run_artifacts,
)
from .performance import (
    PERIODS_PER_YEAR,
    EquityPoint,
    FeeAttribution,
    MetricsReport,
    PerformanceMetrics,
    build_metrics_report,
    compute_equity_curve,
    load_metrics_report,
)

try:
    __version__ = version("pulsar-core")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0.dev0"

__all__ = [
    "__version__",
    # kernel loop
    "EventBus",
    "Handler",
    "canonical_event_json",
    # time sources
    "Clock",
    "BacktestClock",
    "RealtimeClock",
    # events
    "Event",
    "EventKind",
    "SessionPhase",
    "TimerPayload",
    # reproducibility
    "RunManifest",
    "bars_watermark",
    "code_version",
    "load_manifest",
    # replay session
    "ReplaySession",
    "RunResult",
    # errors
    "PulsarCoreError",
    "DataGapError",
    "ClockWentBackwardsError",
    # strategy parameters
    "Param",
    "Params",
    "BoundParams",
    # signals & target portfolios
    "Signal",
    "TargetLeg",
    "TargetPortfolio",
    "PortfolioBuilder",
    "PassThroughBuilder",
    "EqualWeightBuilder",
    # account (decision-side book)
    "TradingAccount",
    "PositionView",
    "PortfolioView",
    # strategy framework
    "StrategyBase",
    "BaseContext",
    "BarContext",
    "TickContext",
    "FillContext",
    "IndicatorValue",
    "FIELD_NAMES",
    # rebalance (diff calculation)
    "DEFAULT_LOT_SIZE",
    "OrderDraft",
    "SkippedLeg",
    "RebalanceResult",
    "compute_rebalance",
    # risk chain
    "RiskRule",
    "RiskGate",
    "RiskView",
    "RuleRejection",
    "RejectionRecord",
    "GateOutcome",
    "SinglePositionCapRule",
    "PortfolioExposureCapRule",
    "DailyLossHaltRule",
    "SymbolBlacklistRule",
    "LiquidityFloorRule",
    "standard_risk_chain",
    # intent pipeline
    "StrategyRuntime",
    "Submission",
    "DEFAULT_HISTORY_DEPTH",
    # performance accounting & run artifacts
    "PERIODS_PER_YEAR",
    "EquityPoint",
    "PerformanceMetrics",
    "FeeAttribution",
    "MetricsReport",
    "compute_equity_curve",
    "build_metrics_report",
    "load_metrics_report",
    "EVENTS_SCHEMA_VERSION",
    "MANIFEST_FILENAME",
    "EVENTS_FILENAME",
    "METRICS_FILENAME",
    "RunArtifacts",
    "write_event_archive",
    "read_event_archive",
    "write_run_artifacts",
]

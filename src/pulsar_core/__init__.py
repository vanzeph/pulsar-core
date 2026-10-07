"""pulsar-core: the deterministic engine core of the Pulsar quant system.

This package implements the single-threaded deterministic event kernel, the
Clock abstraction that lets backtest and realtime runs share one loop, the
bar-level historical replay session, the RunManifest reproducibility
mechanism, the strategy framework with its Signal -> TargetPortfolio ->
RiskGate -> OrderIntent pipeline guarded by the five-rule pre-trade risk
chain (single-position cap, gross exposure cap, daily-loss halt, symbol
blacklist, liquidity floor), performance accounting with its run artifacts
(events.parquet archive + metrics report), and the research layer: a
registered factor library, cross-sectional preprocessing, the modeler
registry, experiment TOML configuration and parameter sweeps.

Dependency policy (architecture baseline): pulsar-core depends only on
pulsar-contracts plus basic libraries. It must never import a data-source or
broker SDK, a network client, or any sibling pulsar package — a static
import-purity test enforces this.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from .bus import EventBus, Handler, canonical_event_json
from .clock import BacktestClock, Clock, RealtimeClock
from .errors import (
    ClockWentBackwardsError,
    DataGapError,
    ExperimentConfigError,
    LifecycleError,
    PulsarCoreError,
)
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

# -- research layer: factors, experiments, sweeps (C3) -------------------------

from .experiment import (
    KNOWN_SECTIONS,
    UNIVERSE_REGISTRY,
    ExperimentConfig,
    SweepAxis,
    SweepExpansion,
    expand_sweep,
    load_experiment,
    parse_experiment,
    register_universe,
)
from .factors import (
    FACTOR_REGISTRY,
    FactorDefinition,
    factor_value,
    momentum_factor,
    range_factor,
    register_factor,
    reversal_factor,
    volatility_factor,
)
from .modelers import (
    MODEL_REGISTRY,
    CrossSection,
    EqualWeightScorer,
    FactorHistoryView,
    IcWeightedScorer,
    LinearScoreScorer,
    ModelDefinition,
    ModelScorer,
    register_model,
    spearman_ic,
)
from .pipeline import (
    REBALANCE_FREQUENCIES,
    FactorEngine,
    FactorModelStrategy,
    rebalance_dates,
    required_warmup,
)
from .portfolio import PORTFOLIO_REGISTRY, PortfolioConstructor, TopNConstructor
from .preprocess import (
    PREPROCESS_REGISTRY,
    FillNaStep,
    PreprocessStep,
    WinsorizeStep,
    ZscoreStep,
    build_step,
)
from .registry import Registry
from .runner import (
    ExperimentRunResult,
    SweepReport,
    VenueFactory,
    run_experiment,
    run_sweep,
)

# -- experiment lifecycle: candidate -> active -> retired (C6) -------------------

from .lifecycle import (
    RUN_MODES,
    STATUS_ALLOWED_MODES,
    ExperimentStatus,
    LifecycleRecord,
    activate_experiment,
    experiment_commit,
    retire_experiment,
    validate_assembly,
)

# -- ML modelers: torch behind the optional [ml] extra (ML1) --------------------
# Importing pulsar_core.ml registers mlp_torch / lstm_torch in MODEL_REGISTRY
# without importing torch (registration is metadata-only; the torch import
# happens inside the ML code paths themselves).

from .ml import (  # noqa: E402
    ARTIFACT_DIRNAME,
    ARTIFACT_FILENAMES,
    TRAINING_CONFIG_FILENAME,
    HASHES_FILENAME,
    ML_EXTRA_INSTALL_HINT,
    WEIGHTS_FILENAME,
    LstmTorchScorer,
    MlpTorchScorer,
    TorchModelScorer,
    enable_determinism,
    load_pinned,
    model_artifact_dir,
    register_torch_modelers,
    require_torch,
    resolve_device,
    save_model_artifact,
    sha256_file,
    training_environment,
)
from .manifest import ModelArtifactRecord

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
    "ExperimentConfigError",
    "LifecycleError",
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
    # registries (name -> registered code things)
    "Registry",
    "FACTOR_REGISTRY",
    "PREPROCESS_REGISTRY",
    "MODEL_REGISTRY",
    "PORTFOLIO_REGISTRY",
    "UNIVERSE_REGISTRY",
    # factor library
    "FactorDefinition",
    "factor_value",
    "register_factor",
    "momentum_factor",
    "volatility_factor",
    "reversal_factor",
    "range_factor",
    "register_model",
    # cross-sectional preprocessing
    "PreprocessStep",
    "WinsorizeStep",
    "ZscoreStep",
    "FillNaStep",
    "build_step",
    # modelers
    "CrossSection",
    "FactorHistoryView",
    "ModelScorer",
    "ModelDefinition",
    "EqualWeightScorer",
    "LinearScoreScorer",
    "IcWeightedScorer",
    "spearman_ic",
    # portfolio construction
    "PortfolioConstructor",
    "TopNConstructor",
    # factor pipeline
    "REBALANCE_FREQUENCIES",
    "rebalance_dates",
    "required_warmup",
    "FactorEngine",
    "FactorModelStrategy",
    # experiment configuration & sweeps
    "KNOWN_SECTIONS",
    "ExperimentConfig",
    "SweepAxis",
    "SweepExpansion",
    "load_experiment",
    "parse_experiment",
    "expand_sweep",
    "register_universe",
    # run entry points
    "ExperimentRunResult",
    "SweepReport",
    "VenueFactory",
    "run_experiment",
    "run_sweep",
    # experiment lifecycle (上下线)
    "RUN_MODES",
    "ExperimentStatus",
    "STATUS_ALLOWED_MODES",
    "LifecycleRecord",
    "validate_assembly",
    "activate_experiment",
    "retire_experiment",
    "experiment_commit",
    # ML modelers (torch via the optional [ml] extra)
    "ModelArtifactRecord",
    "TorchModelScorer",
    "MlpTorchScorer",
    "LstmTorchScorer",
    "ML_EXTRA_INSTALL_HINT",
    "require_torch",
    "resolve_device",
    "enable_determinism",
    "training_environment",
    "ARTIFACT_DIRNAME",
    "WEIGHTS_FILENAME",
    "TRAINING_CONFIG_FILENAME",
    "HASHES_FILENAME",
    "ARTIFACT_FILENAMES",
    "sha256_file",
    "model_artifact_dir",
    "save_model_artifact",
    "load_pinned",
    "register_torch_modelers",
]

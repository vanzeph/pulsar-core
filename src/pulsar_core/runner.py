"""Experiment run entry points: TOML config in, backtest runs out.

:func:`run_experiment` wires the research pipeline into the C2 intent
pipeline end to end (core-engine design: 因子值 → 模型器打分 → 目标组合
→ 既有 RiskGate 出口):

1. resolve the rebalance schedule from the port's trading calendar;
2. fetch bars with enough pre-start warmup for the configured factors
   and model, build the :class:`~pulsar_core.pipeline.FactorEngine` and
   precompute target weights per rebalance date;
3. replay the window through a :class:`~pulsar_core.session.ReplaySession`
   driving a :class:`~pulsar_core.pipeline.FactorModelStrategy` inside a
   :class:`~pulsar_core.runtime.StrategyRuntime` — declarations flow
   through Signal → TargetPortfolio → diff → RiskGate → OrderIntent
   exactly as in C2; nothing here bypasses the risk exit;
4. stamp the experiment document (and, for sweeps, the run's point)
   into the :class:`~pulsar_core.manifest.RunManifest` config snapshot,
   so every run is reproducible and every run of a sweep family is
   distinguishable by its ``run_id``.

:func:`run_sweep` runs the whole family produced by
:func:`~pulsar_core.experiment.expand_sweep` and asserts the family
contract: shared ``experiment_id``, pairwise-distinct ``run_id``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Callable, Union

from pulsar_contracts import AdjustMode, Bar, ExecutionPort, Freq, MarketDataPort

from .bus import EventBus
from .clock import BacktestClock
from .errors import PulsarCoreError
from .experiment import ExperimentConfig, SweepExpansion, expand_sweep
from .lifecycle import experiment_commit, validate_assembly
from .manifest import ModelArtifactRecord, RunManifest, bars_watermark
from .pipeline import FactorEngine, FactorModelStrategy, rebalance_dates, required_warmup
from .rebalance import DEFAULT_LOT_SIZE
from .risk import RiskGate
from .runtime import StrategyRuntime
from .session import ReplaySession, RunResult, _bars_from_frame, _day_start

__all__ = ["ExperimentRunResult", "SweepReport", "VenueFactory", "run_experiment", "run_sweep"]

#: A fresh venue per run; receives the run's bus so fills timestamp off
#: the run's own kernel clock (see the dual-MA example).
VenueFactory = Callable[[EventBus], ExecutionPort]
_VenueSource = Union[ExecutionPort, VenueFactory]


class ExperimentRunResult:
    """One executed experiment run: manifest, session result, runtime."""

    def __init__(
        self,
        *,
        experiment_id: str,
        run_label: str,
        run_index: int,
        run: RunResult,
        runtime: StrategyRuntime,
        strategy: FactorModelStrategy,
    ) -> None:
        self.experiment_id = experiment_id
        self.run_label = run_label
        self.run_index = run_index
        self.run = run
        self.runtime = runtime
        self.strategy = strategy

    @property
    def manifest(self) -> RunManifest:
        return self.run.manifest

    @property
    def run_id(self) -> str:
        return self.run.run_id


class SweepReport:
    """The run family of one sweep: same experiment id, distinct run ids."""

    def __init__(self, *, experiment_id: str, runs: "list[ExperimentRunResult]") -> None:
        self.experiment_id = experiment_id
        self.runs = tuple(runs)

    @property
    def run_ids(self) -> tuple[str, ...]:
        return tuple(result.run_id for result in self.runs)


def _resolve_venue(source: _VenueSource, bus: EventBus) -> ExecutionPort:
    if isinstance(source, ExecutionPort):
        return source
    return source(bus)


def _warmup_start(port: MarketDataPort, start: date, warmup_bars: int) -> date:
    """Enough trading days before ``start`` to warm factors and model.

    Looks back a generous calendar span, takes the last ``warmup_bars``
    trading days and returns their earliest; a data-limited calendar
    simply yields fewer warmup bars (factors then report missing values
    on early dates — visible, not fatal).
    """
    if warmup_bars <= 0:
        return start
    lookback_start = start - timedelta(days=warmup_bars * 2 + 14)
    days = port.calendar(lookback_start, start - timedelta(days=1))
    if not days:
        return start
    window = days[-warmup_bars:]
    warmup_start: date = window[0]
    return warmup_start


def _session_config_block(experiment: ExperimentConfig) -> dict[str, Any]:
    """The session block :meth:`ReplaySession._snapshot_config` prepends.

    Kept beside the preliminary-manifest derivation below so the two stay
    in lockstep — the derived run id is only useful while it equals the
    one the session itself will build from the same inputs.
    """
    return {
        "session": {
            "kind": "bar_replay",
            "symbols": list(experiment.symbols),
            "start": experiment.start,
            "end": experiment.end,
            "freq": Freq.DAILY.value,
            "adjust": AdjustMode.FORWARD.value,
        }
    }


def _preliminary_run_id(
    experiment: ExperimentConfig,
    *,
    bars: "list[Bar]",
    config: dict[str, Any],
) -> str:
    """Derive the run id before the session builds its own manifest.

    ML artifact reuse needs the run id before training (the artifact lives
    at ``runs/<run_id>/model_artifact``), so the same manifest inputs the
    session will use — mode, seed, config snapshot (session block +
    experiment document + sweep point), code version and the bar
    watermarks — are run through the same :meth:`RunManifest.build`
    derivation here. The bars this runner fetched are a warmup-prefixed
    superset of the session's window, so they are clipped to the replay
    window first — the session's watermark set is exactly the partitions
    it fetched, and a drift between this derivation and the session's own
    fails the runner's post-run invariant check loudly.
    """
    window_bars = [
        bar
        for bar in bars
        if experiment.start <= bar.ts.date() <= experiment.end
    ]
    merged = {**_session_config_block(experiment), **config}
    preliminary = RunManifest.build(
        mode="research",
        seed=experiment.seed,
        config=merged,
        data_watermarks=bars_watermark(window_bars),
    )
    return preliminary.run_id


def _torch_scorer(model: Any) -> "Any | None":
    """The model when it is a torch artifact carrier, else ``None``."""
    from .ml import TorchModelScorer

    return model if isinstance(model, TorchModelScorer) else None


def _record_artifact(
    manifest: RunManifest,
    *,
    model: Any,
    artifact_dir: Any,
    origin: str,
    hashes: "dict[str, str] | None",
) -> None:
    from .ml.artifacts import (
        TRAINING_CONFIG_FILENAME,
        WEIGHTS_FILENAME,
        read_pinned_manifest,
    )

    if hashes is None:
        hashes = read_pinned_manifest(artifact_dir)["files"]
    manifest.record_model_artifact(
        ModelArtifactRecord(
            model_type=model.name,
            path=str(artifact_dir),
            weights_sha256=str(hashes[WEIGHTS_FILENAME]),
            config_sha256=str(hashes[TRAINING_CONFIG_FILENAME]),
            origin=origin,
            environment=model.environment,
        )
    )


def run_experiment(
    experiment: ExperimentConfig,
    *,
    port: MarketDataPort,
    venue: _VenueSource,
    initial_cash: float = 1_000_000.0,
    lot_size: int = DEFAULT_LOT_SIZE,
    gate: "RiskGate | None" = None,
    bus: "EventBus | None" = None,
    expansion: SweepExpansion | None = None,
    config_commit: "str | None" = None,
    runs_root: "str | Path | None" = None,
) -> ExperimentRunResult:
    """Run one (already validated) experiment over the injected ports.

    ``venue`` is either a ready :class:`ExecutionPort` or a factory
    receiving the run's bus (sweeps must pass a factory: every run needs
    its own venue). ``expansion`` carries the sweep point when this run
    belongs to a family — it is stamped into the manifest so run ids
    differ per point while the experiment id stays shared.

    This is the Research assembler, so the lifecycle gate applies
    (模型配置生命周期): a ``retired`` experiment is refused outright —
    post-mortems replay archives, they do not re-run — while ``candidate``
    and ``active`` both admit research. ``config_commit`` records the git
    commit of the experiment configuration this assembly used (explicit
    sha wins; otherwise resolved from the file the config was loaded
    from), keeping "which version ran" traceable.

    ``runs_root`` opts the run into ML artifact versioning (训练产物工
    件化): a torch modeler's weights + training config + sha256 land in
    ``<runs_root>/<run_id>/model_artifact/`` and the RunManifest gains
    the model-artifact record. A rerun of the same run id finds the
    pinned artifact, hash-verifies it and scores from it *without
    retraining* — set ``model.params.retrain = true`` to force a fresh
    training pass over the reuse.
    """
    validate_assembly("research", experiment)
    if config_commit is None and experiment.source_path is not None:
        config_commit = experiment_commit(experiment.source_path)
    if experiment.sweep and expansion is None:
        raise PulsarCoreError(
            "this experiment declares sweep axes; run it through run_sweep "
            "(or expand_sweep + run_experiment with the expansion passed)"
        )
    trading_days = port.calendar(experiment.start, experiment.end)
    if not trading_days:
        raise PulsarCoreError(
            f"trading calendar is empty for [{experiment.start}, {experiment.end}]"
        )
    schedule = rebalance_dates(trading_days, experiment.rebalance)

    warmup_bars = required_warmup(experiment.factors, experiment.model)
    warmup_start = _warmup_start(port, experiment.start, warmup_bars)
    frame = port.fetch_bars(
        list(experiment.symbols),
        warmup_start,
        experiment.end,
        Freq.DAILY,
        AdjustMode.FORWARD,
    )
    bars = _bars_from_frame(frame, Freq.DAILY)
    bars_by_symbol: dict[str, list[Bar]] = {}
    for bar in bars:
        bars_by_symbol.setdefault(bar.symbol, []).append(bar)

    # The config snapshot the session will archive (plus the sweep point).
    config: dict[str, Any] = experiment.config_snapshot()
    if expansion is not None and expansion.assignments:
        config["sweep"] = {
            "index": expansion.index,
            "label": expansion.label,
            "point": dict(expansion.assignments),
        }

    # ML artifact reuse: same run id -> pinned artifact -> no retraining.
    torch_model = _torch_scorer(experiment.model) if runs_root is not None else None
    artifact_dir: "Path | None" = None
    preliminary_run_id: "str | None" = None
    reused_pinned = False
    if torch_model is not None and runs_root is not None:
        from .ml.artifacts import model_artifact_dir

        preliminary_run_id = _preliminary_run_id(experiment, bars=bars, config=config)
        artifact_dir = model_artifact_dir(runs_root, preliminary_run_id)
        if artifact_dir.is_dir() and not torch_model.force_retrain:
            torch_model.load_pinned(artifact_dir)
            reused_pinned = True

    engine = FactorEngine(
        symbols=experiment.symbols,
        bars=bars_by_symbol,
        factors=experiment.factors,
        preprocess=experiment.preprocess,
        model=experiment.model,
        portfolio=experiment.portfolio,
    )
    targets = engine.targets(schedule)
    strategy = FactorModelStrategy(targets)

    if bus is None:
        bus = EventBus(BacktestClock(_day_start(experiment.start)))
    execution = _resolve_venue(venue, bus)
    runtime = StrategyRuntime(
        bus=bus,
        port=execution,
        strategy=strategy,
        gate=gate,
        initial_cash=initial_cash,
        lot_size=lot_size,
    )

    session = ReplaySession(
        port=port,
        symbols=list(experiment.symbols),
        start=experiment.start,
        end=experiment.end,
        seed=experiment.seed,
        config=config,
        config_commit=config_commit,
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    run = session.run()

    if artifact_dir is not None and torch_model is not None:
        assert preliminary_run_id is not None  # artifact path was derived from it
        if preliminary_run_id != run.run_id:
            raise PulsarCoreError(
                "preliminary run id drifted from the session's manifest; "
                "the ML artifact reuse decision was made against a stale id"
            )
        if reused_pinned:
            _record_artifact(
                run.manifest,
                model=torch_model,
                artifact_dir=artifact_dir,
                origin="pinned",
                hashes=None,
            )
        elif torch_model.artifact_ready:
            hashes = torch_model.save_artifact(artifact_dir)
            _record_artifact(
                run.manifest,
                model=torch_model,
                artifact_dir=artifact_dir,
                origin="trained",
                hashes=hashes,
            )

    run_label = expansion.label if expansion is not None else ""
    run_index = expansion.index if expansion is not None else 0
    return ExperimentRunResult(
        experiment_id=experiment.experiment_id,
        run_label=run_label,
        run_index=run_index,
        run=run,
        runtime=runtime,
        strategy=strategy,
    )


def run_sweep(
    experiment: ExperimentConfig,
    *,
    port: MarketDataPort,
    make_venue: VenueFactory,
    initial_cash: float = 1_000_000.0,
    lot_size: int = DEFAULT_LOT_SIZE,
    gate: "RiskGate | None" = None,
    runs_root: "str | Path | None" = None,
) -> SweepReport:
    """Expand and run the whole sweep family sequentially.

    Every run gets a fresh bus and a fresh venue from ``make_venue``.
    ``runs_root`` passes the ML artifact root through to every member run
    (each expansion trains or reuses its own pinned artifact under its
    own run id). The family contract is asserted, not assumed: one shared
    experiment id in every manifest, and pairwise-distinct run ids.
    """
    expansions = expand_sweep(experiment)
    results: list[ExperimentRunResult] = []
    for expansion in expansions:
        results.append(
            run_experiment(
                expansion.config,
                port=port,
                venue=make_venue,
                initial_cash=initial_cash,
                lot_size=lot_size,
                gate=gate,
                expansion=expansion,
                runs_root=runs_root,
            )
        )
    experiment_ids = {result.experiment_id for result in results}
    if len(experiment_ids) != 1:
        raise PulsarCoreError(
            f"sweep family leaked experiment ids: {sorted(experiment_ids)}"
        )
    run_ids = [result.run_id for result in results]
    if len(set(run_ids)) != len(run_ids):
        raise PulsarCoreError(
            "sweep family produced duplicate run ids; swept values do not "
            "distinguish the runs"
        )
    return SweepReport(experiment_id=experiment_ids.pop(), runs=results)

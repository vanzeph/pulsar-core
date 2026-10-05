"""Experiment configuration: one TOML file, one experiment definition.

Layered configuration (core-engine design, 因子库与实验配置): factors,
modelers and portfolio construction are registered code; an *experiment*
is a pure TOML document that references them — factor subset, preprocess
composition, model type and hyper-parameters, universe, portfolio and
backtest settings. A new experiment is a new file, zero code::

    [experiment]
    id = "momentum_value_2026q4"
    universe = "hs300"                      # registered universe name
    status = "candidate"                    # candidate | active | retired

    [factors]
    names = ["momentum_20", "volatility_20", "reversal_5"]
    preprocess = ["winsorize", "zscore"]

    [model]
    type = "ic_weighted"
    params = { lookback = 60, horizon = 5 }

    [portfolio]
    method = "top_n"
    top_n = 30
    rebalance = "monthly"

    [backtest]
    start = 2020-01-01
    end = 2026-09-30
    costs = "a_share_default"

The universe may alternatively be spelled as its own section with an
explicit symbol list (``[universe] symbols = [...]``) — exactly one of
the two forms is required. Loading validates everything: unknown
sections, unknown keys inside every section, wrong types, missing
required values (including the lifecycle ``status``), unregistered names
and sweep axes that do not address the template all raise
:class:`~pulsar_core.errors.ExperimentConfigError` at load time.

Parameter sweeps (参数扫描) extend the same file with a ``[sweep]``
section of axes; :func:`expand_sweep` materializes the cartesian product
into concrete runs that share the template's ``experiment.id`` while
each run keeps its own configuration (and therefore its own
``run_id``).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import tomllib

from .errors import ExperimentConfigError, PulsarCoreError
from .factors import FACTOR_REGISTRY, FactorDefinition
from .lifecycle import STATUS_ALLOWED_MODES, ExperimentStatus
from .modelers import MODEL_REGISTRY, ModelScorer
from .pipeline import REBALANCE_FREQUENCIES
from .portfolio import PORTFOLIO_REGISTRY, PortfolioConstructor
from .preprocess import PREPROCESS_REGISTRY, PreprocessStep, build_step
from .registry import Registry

__all__ = [
    "UniverseDefinition",
    "UNIVERSE_REGISTRY",
    "register_universe",
    "SweepAxis",
    "SweepExpansion",
    "ExperimentConfig",
    "load_experiment",
    "parse_experiment",
    "expand_sweep",
]

#: Top-level sections an experiment document may carry.
KNOWN_SECTIONS: tuple[str, ...] = (
    "experiment",
    "universe",
    "factors",
    "model",
    "portfolio",
    "backtest",
    "sweep",
)

_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "experiment": ("id", "universe", "description", "status"),
    "universe": ("name", "symbols"),
    "factors": ("names", "preprocess"),
    "model": ("type", "params"),
    "portfolio": ("method", "top_n", "rebalance"),
    "backtest": ("start", "end", "costs", "seed"),
    "sweep": ("axis",),
}

# -- universes ---------------------------------------------------------------------


class UniverseDefinition:
    """A registered stock universe (股票域): name → symbol provider.

    ``symbols`` is either an explicit sequence or a callable taking the
    backtest start date (dynamically composed universes); resolution
    happens once at config-load time with ``as_of = backtest.start``, so
    the resolved list is frozen into the run's session configuration and
    the manifest stays reproducible.
    """

    def __init__(
        self,
        *,
        name: str,
        symbols: "Sequence[str] | Callable[[date], Sequence[str]]",
        description: str = "",
    ) -> None:
        if not name:
            raise ValueError("universe name must be non-empty")
        self.name = name
        self.symbols = symbols
        self.description = description

    def resolve(self, as_of: date) -> tuple[str, ...]:
        provided = self.symbols(as_of) if callable(self.symbols) else self.symbols
        resolved = sorted(set(provided))
        if not resolved:
            raise ExperimentConfigError(
                f"universe {self.name!r} resolves to an empty symbol list"
            )
        return tuple(resolved)


#: The default registry of stock universes.
UNIVERSE_REGISTRY: Registry[UniverseDefinition] = Registry(
    kind="universe", name_of=lambda universe: universe.name
)


def register_universe(
    name: str,
    symbols: "Sequence[str] | Callable[[date], Sequence[str]]",
    *,
    description: str = "",
) -> None:
    """Register a universe experiments reference by name."""
    UNIVERSE_REGISTRY.register(
        UniverseDefinition(name=name, symbols=symbols, description=description)
    )


# -- sweep --------------------------------------------------------------------------


class SweepAxis:
    """One sweep dimension: a dotted ``path`` into the template and values."""

    def __init__(self, *, path: str, values: Sequence[Any]) -> None:
        if not path or any(not part.isidentifier() for part in path.split(".")):
            raise ExperimentConfigError(
                f"sweep axis path must be dotted identifiers, got {path!r}"
            )
        if not values:
            raise ExperimentConfigError(
                f"sweep axis {path!r} needs at least one value"
            )
        self.path = path
        self.values = tuple(values)


def _render_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(_render_value(item) for item in value) + "]"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    return str(value)


class SweepExpansion:
    """One concrete run produced by :func:`expand_sweep`.

    ``index`` orders the run family, ``assignments`` maps each swept path
    to its value in this run, and ``label`` is a deterministic rendering
    of the assignments (manifest-friendly).
    """

    def __init__(
        self,
        *,
        config: "ExperimentConfig",
        index: int,
        assignments: Mapping[str, Any],
        label: str,
    ) -> None:
        self.config = config
        self.index = index
        self.assignments = dict(assignments)
        self.label = label


# -- experiment config ---------------------------------------------------------------


class ExperimentConfig:
    """A validated, resolved experiment definition.

    Carries both the as-written TOML tree (``raw``, JSON-native apart
    from TOML dates — used for manifest snapshots and sweep expansion)
    and the resolved building blocks (factor definitions, preprocess
    steps, scorer, portfolio constructor, universe symbols). The lifecycle
    ``status`` (candidate / active / retired, required) resolves together
    with the modes that status admits (``allowed_modes``); assembly layers
    enforce the pairing via
    :func:`~pulsar_core.lifecycle.validate_assembly`.
    """

    def __init__(
        self,
        *,
        raw: Mapping[str, Any],
        experiment_id: str,
        description: str,
        symbols: Sequence[str],
        factors: Sequence[FactorDefinition],
        preprocess: Sequence[PreprocessStep],
        model: ModelScorer,
        portfolio: PortfolioConstructor,
        rebalance: str,
        start: date,
        end: date,
        costs: str,
        seed: int,
        sweep: Sequence[SweepAxis],
        status: str = ExperimentStatus.CANDIDATE,
        source_path: "str | None" = None,
    ) -> None:
        self.raw: dict[str, Any] = _clone_tree(raw)
        self.experiment_id = experiment_id
        self.description = description
        self.symbols = tuple(symbols)
        self.factors = tuple(factors)
        self.preprocess = tuple(preprocess)
        self.model = model
        self.portfolio = portfolio
        self.rebalance = rebalance
        self.start = start
        self.end = end
        self.costs = costs
        self.seed = seed
        self.sweep = tuple(sweep)
        if status not in STATUS_ALLOWED_MODES:
            raise ExperimentConfigError(
                f"experiment.status must be one of {ExperimentStatus.ALL}, "
                f"got {status!r}"
            )
        self.status: str = status
        #: Modes this status may be assembled in (derived from the design's
        #: lifecycle table, not declarable in TOML — a config cannot grant
        #: itself paper/live rights).
        self.allowed_modes: tuple[str, ...] = STATUS_ALLOWED_MODES[status]
        #: Where the document was loaded from, when it came from a file;
        #: the assembly layer resolves the registry git commit from it.
        self.source_path = source_path

    @property
    def factor_names(self) -> tuple[str, ...]:
        return tuple(factor.name for factor in self.factors)

    def config_snapshot(self) -> dict[str, Any]:
        """The as-written document as a manifest-ready dict (deep copy)."""
        snapshot = _clone_tree(self.raw)
        assert isinstance(snapshot, dict)  # the raw tree is always a table
        return snapshot


# -- loading ------------------------------------------------------------------------


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load and validate an experiment TOML file."""
    try:
        with open(Path(path), "rb") as handle:
            tree = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise ExperimentConfigError(f"{path}: invalid TOML ({exc})") from exc
    except OSError as exc:
        raise ExperimentConfigError(f"{path}: cannot read experiment ({exc})") from exc
    return parse_experiment(tree, source_path=str(path))


def parse_experiment(
    tree: Mapping[str, Any],
    *,
    source_path: "str | None" = None,
) -> ExperimentConfig:
    """Validate a parsed TOML tree into an :class:`ExperimentConfig`."""
    if not isinstance(tree, Mapping):
        raise ExperimentConfigError("experiment document must be a TOML table")
    unknown_sections = sorted(set(tree) - set(KNOWN_SECTIONS))
    if unknown_sections:
        raise ExperimentConfigError(
            f"unknown section(s) {unknown_sections}; expected subset of "
            f"{list(KNOWN_SECTIONS)}"
        )
    experiment = _section(tree, "experiment")
    universe = _section(tree, "universe")
    factors = _section(tree, "factors")
    model = _section(tree, "model")
    portfolio = _section(tree, "portfolio")
    backtest = _section(tree, "backtest")
    sweep = _section(tree, "sweep")

    experiment_id = _string(experiment.get("id"), "experiment.id")
    description = _string(experiment.get("description", ""), "experiment.description")
    if not experiment_id:
        raise ExperimentConfigError("experiment.id must be a non-empty string")
    status = _status(experiment.get("status"))

    symbols = _resolve_symbols(experiment, universe, backtest)

    factor_names = _factor_names(factors)
    factor_defs = tuple(FACTOR_REGISTRY.resolve(name) for name in factor_names)
    steps = _preprocess_steps(factors.get("preprocess", []))
    scorer = _scorer(model, factor_names)
    constructor, rebalance = _portfolio(portfolio)
    start, end, costs, seed = _backtest(backtest)

    axes = _sweep_axes(sweep, tree)

    return ExperimentConfig(
        raw={key: value for key, value in tree.items()},
        experiment_id=experiment_id,
        description=description,
        symbols=symbols,
        factors=factor_defs,
        preprocess=steps,
        model=scorer,
        portfolio=constructor,
        rebalance=rebalance,
        start=start,
        end=end,
        costs=costs,
        seed=seed,
        sweep=axes,
        status=status,
        source_path=source_path,
    )


def _status(value: Any) -> str:
    """Validate the required lifecycle status (candidate/active/retired)."""
    if value is None:
        raise ExperimentConfigError(
            "experiment.status is required and must be one of "
            f"{ExperimentStatus.ALL} (candidate = research only; active = "
            "research/paper/live; retired = read-only post-mortem)"
        )
    if not isinstance(value, str):
        raise ExperimentConfigError(
            f"experiment.status must be a string, got {value!r}"
        )
    if value not in STATUS_ALLOWED_MODES:
        raise ExperimentConfigError(
            f"experiment.status must be one of {ExperimentStatus.ALL}, "
            f"got {value!r}"
        )
    return value


# -- section validators ---------------------------------------------------------------


def _section(tree: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = tree.get(name, {})
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ExperimentConfigError(f"[{name}] must be a TOML table")
    allowed = _SECTION_KEYS[name]
    unknown = sorted(set(value) - set(allowed))
    if unknown:
        raise ExperimentConfigError(
            f"unknown key(s) {unknown} in [{name}]; allowed: {list(allowed)}"
        )
    return dict(value)


def _string(value: Any, where: str, *, default: "str | None" = None) -> str:
    if value is None:
        if default is not None:
            return default
        raise ExperimentConfigError(f"{where} is required")
    if not isinstance(value, str):
        raise ExperimentConfigError(f"{where} must be a string, got {value!r}")
    return value


def _resolve_symbols(
    experiment: Mapping[str, Any],
    universe: Mapping[str, Any],
    backtest: Mapping[str, Any],
) -> tuple[str, ...]:
    name_form = experiment.get("universe")
    section_form = bool(universe)
    if name_form is not None and section_form:
        raise ExperimentConfigError(
            "universe is declared twice: experiment.universe and [universe]; pick one"
        )
    if name_form is None and not section_form:
        raise ExperimentConfigError(
            "no universe: set experiment.universe (registered name) or [universe]"
        )
    as_of = _date(backtest.get("start"), "backtest.start", required=False)
    anchor = as_of if as_of is not None else date(1970, 1, 1)
    if name_form is not None:
        if not isinstance(name_form, str) or not name_form:
            raise ExperimentConfigError(
                f"experiment.universe must be a registered name, got {name_form!r}"
            )
        try:
            definition = UNIVERSE_REGISTRY.resolve(name_form)
        except PulsarCoreError as exc:
            raise ExperimentConfigError(str(exc)) from exc
        return definition.resolve(anchor)
    if "name" in universe and "symbols" in universe:
        raise ExperimentConfigError(
            "[universe] carries both name and symbols; pick one"
        )
    if "name" in universe:
        name = _string(universe["name"], "universe.name")
        try:
            definition = UNIVERSE_REGISTRY.resolve(name)
        except PulsarCoreError as exc:
            raise ExperimentConfigError(str(exc)) from exc
        return definition.resolve(anchor)
    raw_symbols = universe.get("symbols")
    if not isinstance(raw_symbols, list) or not raw_symbols:
        raise ExperimentConfigError(
            "[universe] requires a non-empty symbols list (or a registered name)"
        )
    for symbol in raw_symbols:
        if not isinstance(symbol, str) or not symbol:
            raise ExperimentConfigError(
                f"universe symbols must be non-empty strings, got {symbol!r}"
            )
    if len(set(raw_symbols)) != len(raw_symbols):
        raise ExperimentConfigError("[universe] symbols contains duplicates")
    return tuple(sorted(raw_symbols))


def _factor_names(factors: Mapping[str, Any]) -> list[str]:
    raw = factors.get("names")
    if not isinstance(raw, list) or not raw:
        raise ExperimentConfigError(
            "[factors] names must be a non-empty list of registered factor names"
        )
    names: list[str] = []
    for name in raw:
        if not isinstance(name, str) or not name:
            raise ExperimentConfigError(
                f"factor names must be non-empty strings, got {name!r}"
            )
        if name in names:
            raise ExperimentConfigError(f"duplicate factor name {name!r}")
        if name not in FACTOR_REGISTRY:
            raise ExperimentConfigError(
                f"unknown factor {name!r}; registered: {FACTOR_REGISTRY.names()}"
            )
        names.append(name)
    return names


def _preprocess_steps(raw: Any) -> tuple[PreprocessStep, ...]:
    if not isinstance(raw, list):
        raise ExperimentConfigError(
            "[factors] preprocess must be a list of step names (or step tables)"
        )
    steps: list[PreprocessStep] = []
    for item in raw:
        if isinstance(item, str):
            if item not in PREPROCESS_REGISTRY:
                raise ExperimentConfigError(
                    f"unknown preprocess step {item!r}; "
                    f"registered: {PREPROCESS_REGISTRY.names()}"
                )
            steps.append(build_step(item))
        elif isinstance(item, Mapping):
            step_name = item.get("step")
            if not isinstance(step_name, str) or not step_name:
                raise ExperimentConfigError(
                    f"preprocess step table requires a 'step' key, got {dict(item)}"
                )
            overrides = {key: value for key, value in item.items() if key != "step"}
            try:
                steps.append(build_step(step_name, overrides))
            except Exception as exc:
                raise ExperimentConfigError(f"preprocess step {step_name!r}: {exc}") from exc
        else:
            raise ExperimentConfigError(
                f"preprocess entries must be strings or tables, got {item!r}"
            )
    return tuple(steps)


def _scorer(model: Mapping[str, Any], factor_names: Sequence[str]) -> ModelScorer:
    model_type = model.get("type")
    if not isinstance(model_type, str) or not model_type:
        raise ExperimentConfigError("[model] type must be a registered model name")
    if model_type not in MODEL_REGISTRY:
        raise ExperimentConfigError(
            f"unknown model {model_type!r}; registered: {MODEL_REGISTRY.names()}"
        )
    params = model.get("params", {})
    if params is None:
        params = {}
    if not isinstance(params, Mapping):
        raise ExperimentConfigError("[model] params must be a TOML table")
    try:
        return MODEL_REGISTRY.resolve(model_type).create(dict(params), factor_names)
    except ExperimentConfigError:
        raise
    except Exception as exc:
        raise ExperimentConfigError(f"model {model_type!r}: {exc}") from exc


def _portfolio(
    portfolio: Mapping[str, Any],
) -> tuple[PortfolioConstructor, str]:
    method = portfolio.get("method", "top_n")
    if not isinstance(method, str) or not method:
        raise ExperimentConfigError(
            "[portfolio] method must be a registered portfolio method name"
        )
    if method not in PORTFOLIO_REGISTRY:
        raise ExperimentConfigError(
            f"unknown portfolio method {method!r}; "
            f"registered: {PORTFOLIO_REGISTRY.names()}"
        )
    rebalance = portfolio.get("rebalance", "monthly")
    if rebalance not in REBALANCE_FREQUENCIES:
        raise ExperimentConfigError(
            f"[portfolio] rebalance must be one of {REBALANCE_FREQUENCIES}, "
            f"got {rebalance!r}"
        )
    top_n = portfolio.get("top_n", 10)
    if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
        raise ExperimentConfigError(
            f"[portfolio] top_n must be a positive integer, got {top_n!r}"
        )
    try:
        constructor = PORTFOLIO_REGISTRY.resolve(method).create({"top_n": top_n})
    except Exception as exc:
        raise ExperimentConfigError(f"portfolio method {method!r}: {exc}") from exc
    return constructor, rebalance


def _backtest(section: Mapping[str, Any]) -> tuple[date, date, str, int]:
    maybe_start = _date(section.get("start"), "backtest.start")
    maybe_end = _date(section.get("end"), "backtest.end")
    assert maybe_start is not None and maybe_end is not None  # required dates
    start, end = maybe_start, maybe_end
    if start > end:
        raise ExperimentConfigError(
            f"backtest.start {start} must not be after backtest.end {end}"
        )
    costs = section.get("costs", "none")
    if not isinstance(costs, str) or not costs:
        raise ExperimentConfigError("backtest.costs must be a non-empty string")
    seed = section.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ExperimentConfigError(f"backtest.seed must be an integer, got {seed!r}")
    return start, end, costs, seed


def _date(value: Any, where: str, *, required: bool = True) -> "date | None":
    if value is None:
        if required:
            raise ExperimentConfigError(f"{where} is required (TOML date, 2026-01-01)")
        return None
    if isinstance(value, datetime):
        raise ExperimentConfigError(
            f"{where} must be a plain TOML date (2026-01-01), got datetime {value!r}"
        )
    if not isinstance(value, date):
        raise ExperimentConfigError(f"{where} must be a TOML date, got {value!r}")
    return value


# -- sweep expansion -------------------------------------------------------------------


def _sweep_axes(sweep: Mapping[str, Any], tree: Mapping[str, Any]) -> tuple[SweepAxis, ...]:
    raw_axes = sweep.get("axis", [])
    if not isinstance(raw_axes, list):
        raise ExperimentConfigError("[sweep] axis must be an array of axis tables")
    template = {key: value for key, value in tree.items() if key != "sweep"}
    axes: list[SweepAxis] = []
    seen_paths: set[str] = set()
    for raw in raw_axes:
        if not isinstance(raw, Mapping):
            raise ExperimentConfigError(f"sweep axis must be a table, got {raw!r}")
        unknown = sorted(set(raw) - {"path", "values"})
        if unknown:
            raise ExperimentConfigError(
                f"unknown key(s) {unknown} in [[sweep.axis]]; allowed: ['path', 'values']"
            )
        path = raw.get("path")
        if not isinstance(path, str) or not path:
            raise ExperimentConfigError("sweep axis path must be a non-empty string")
        if path in seen_paths:
            raise ExperimentConfigError(f"duplicate sweep axis path {path!r}")
        seen_paths.add(path)
        values = raw.get("values")
        if not isinstance(values, list) or not values:
            raise ExperimentConfigError(
                f"sweep axis {path!r} requires a non-empty values list"
            )
        current = _resolve_path(template, path)
        for value in values:
            _check_compatible(current, value, path)
        axes.append(SweepAxis(path=path, values=values))
    return tuple(axes)


def expand_sweep(config: ExperimentConfig) -> tuple[SweepExpansion, ...]:
    """Materialize the sweep axes into the concrete run family.

    Without axes the experiment itself is the single run. Every expanded
    run re-validates the full mutated document, so a swept value that
    breaks the schema fails here — at expansion time, not mid-backtest.
    Duplicate resulting configurations are rejected (they would collide
    on the same run id and defeat the run family).
    """
    axes = config.sweep
    if not axes:
        return (
            SweepExpansion(config=config, index=0, assignments={}, label=""),
        )
    template = {key: value for key, value in config.raw.items() if key != "sweep"}
    combinations: list[dict[str, Any]] = [{}]
    for axis in axes:
        combinations = [
            {**combination, axis.path: value}
            for combination in combinations
            for value in axis.values
        ]
    expansions: list[SweepExpansion] = []
    seen: list[str] = []
    for index, assignments in enumerate(combinations):
        mutated = _clone_tree(template)
        for path, value in assignments.items():
            _set_path(mutated, path, value)
        fingerprint = _tree_fingerprint(mutated)
        if fingerprint in seen:
            raise ExperimentConfigError(
                f"sweep run #{index + 1} duplicates an earlier run's configuration; "
                "drop duplicate axis values"
            )
        seen.append(fingerprint)
        label = ",".join(
            f"{path}={_render_value(assignments[path])}" for path in sorted(assignments)
        )
        expansions.append(
            SweepExpansion(
                config=parse_experiment(mutated),
                index=index,
                assignments=assignments,
                label=label,
            )
        )
    return tuple(expansions)


def _resolve_path(tree: dict[str, Any], path: str) -> Any:
    node: Any = tree
    for part in path.split("."):
        if isinstance(node, dict):
            if part not in node:
                raise ExperimentConfigError(
                    f"sweep axis path {path!r} does not address the template "
                    f"(missing key {part!r})"
                )
            node = node[part]
        elif isinstance(node, list):
            try:
                index = int(part)
            except ValueError as exc:
                raise ExperimentConfigError(
                    f"sweep axis path {path!r} indexes a list with {part!r}"
                ) from exc
            if not 0 <= index < len(node):
                raise ExperimentConfigError(
                    f"sweep axis path {path!r} indexes out of range ({part!r})"
                )
            node = node[index]
        else:
            raise ExperimentConfigError(
                f"sweep axis path {path!r} descends into a scalar at {part!r}"
            )
    return node


def _set_path(tree: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    node: Any = tree
    for part in parts[:-1]:
        node = _step_into(node, part, path)
    last = parts[-1]
    if isinstance(node, dict):
        if last not in node:
            raise ExperimentConfigError(
                f"sweep axis path {path!r} does not address the template"
            )
        _check_compatible(node[last], value, path)
        node[last] = value
    elif isinstance(node, list):
        try:
            index = int(last)
        except ValueError as exc:
            raise ExperimentConfigError(
                f"sweep axis path {path!r} indexes a list with {last!r}"
            ) from exc
        if not 0 <= index < len(node):
            raise ExperimentConfigError(
                f"sweep axis path {path!r} indexes out of range ({last!r})"
            )
        _check_compatible(node[index], value, path)
        node[index] = value
    else:  # pragma: no cover - _step_into already rejects scalar descent
        raise ExperimentConfigError(f"sweep axis path {path!r} descends into a scalar")


def _step_into(node: Any, part: str, path: str) -> Any:
    if isinstance(node, dict):
        if part not in node:
            raise ExperimentConfigError(
                f"sweep axis path {path!r} does not address the template "
                f"(missing key {part!r})"
            )
        return node[part]
    if isinstance(node, list):
        try:
            index = int(part)
        except ValueError as exc:
            raise ExperimentConfigError(
                f"sweep axis path {path!r} indexes a list with {part!r}"
            ) from exc
        if not 0 <= index < len(node):
            raise ExperimentConfigError(
                f"sweep axis path {path!r} indexes out of range ({part!r})"
            )
        return node[index]
    raise ExperimentConfigError(
        f"sweep axis path {path!r} descends into a scalar at {part!r}"
    )


def _check_compatible(current: Any, value: Any, path: str) -> None:
    def same_kind(a: Any, b: Any) -> bool:
        if isinstance(a, bool) or isinstance(b, bool):
            return isinstance(a, bool) and isinstance(b, bool)
        if isinstance(a, float):
            return isinstance(b, (int, float)) and not isinstance(b, bool)
        if isinstance(a, int):
            return isinstance(b, int)
        if isinstance(a, str):
            return isinstance(b, str)
        if isinstance(a, (date, datetime)):
            return isinstance(b, (date, datetime))
        if isinstance(a, list):
            return isinstance(b, list)
        return False

    if not same_kind(current, value):
        raise ExperimentConfigError(
            f"sweep value for {path!r} has type {type(value).__name__}, "
            f"but the template holds {type(current).__name__}"
        )


def _clone_tree(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    return value


def _tree_fingerprint(tree: Mapping[str, Any]) -> str:
    def encode(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: encode(item) for key, item in value.items()}
        if isinstance(value, list):
            return [encode(item) for item in value]
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        return value

    return json.dumps(encode(tree), sort_keys=True, separators=(",", ":"))

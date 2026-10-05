# pulsar-core

The deterministic engine core of the [Pulsar](https://github.com/vanzeph)
A-share quant system: a single-threaded event kernel, a Clock abstraction
that makes backtest and realtime runs share one loop, a bar-level historical
replay session, the RunManifest reproducibility mechanism, the strategy
framework with its `Signal -> TargetPortfolio -> RiskGate -> OrderIntent`
pipeline, performance accounting with its run artifacts (`events.parquet` +
metrics report), and the research layer on top — a registered factor
library, cross-sectional preprocessing, modeler registry, experiment TOML
configuration and parameter sweeps.

Status: alpha — the kernel, replay, manifest, strategy framework, intent
pipeline, the factor/experiment layer, the five-rule risk set and
performance accounting are in; richer universes and modeler families land
in later releases.

## Design in one paragraph

Determinism wins over throughput. Every event — market, execution, timer —
is dispatched by one thread in strict `(timestamp, publish-sequence)` order,
so a run is a pure function of its inputs. The core reads market data only
through the `MarketDataPort` protocol and trades only through
`ExecutionPort` (both defined in
[pulsar-contracts](https://github.com/vanzeph/pulsar-contracts)); it never
imports a data-source or broker SDK, never touches the network, and a static
import-purity test keeps it that way. A `RunManifest` archives the config
snapshot, data watermarks, code version (git commit) and random seed of a
run; rerunning the same manifest on the same code version must reproduce the
event journal bit for bit.

## The strategy contract

Strategies declare *what they want*; the engine decides *how to get it
there*. A strategy:

- subclasses `StrategyBase` and implements any of the lifecycle hooks
  `on_start` / `on_bar` / `on_tick` / `on_fill` / `on_stop`;
- reads market data and the account book only through the typed context
  handed to each hook, and keeps cross-bar state explicitly in `ctx.state`
  (stateless-first, archived with the run);
- declares target positions (`ctx.target_weight` / `ctx.target_shares`) —
  it never sizes, places, amends or cancels an order. The runtime computes
  the difference against holdings plus in-flight orders, floors buys to
  whole board lots (100 shares), lets zero targets liquidate odd tails,
  runs every draft through the risk chain, and only then emits an
  `OrderIntent` keyed by `run_id + sequence`.

The risk exit is structural, not conventional: contexts expose no ordering
surface (no port, no gate, no engine reference), and a static test proves
`port.submit` is called from exactly one place in the package — the
post-gate emitter. The shipped `RiskGate` runs rules as a chain
(first rejection wins, every rejection recorded); `SinglePositionCapRule`
is the first built-in rule, the remaining design rules arrive with the risk
task.

```python
from pulsar_core import Params, StrategyBase, BarContext

class DualMA(StrategyBase):
    params = Params(fast=5, slow=20)

    def on_bar(self, ctx: BarContext) -> None:
        fast = ctx.sma("close", self.params.fast)
        slow = ctx.sma("close", self.params.slow)
        if ctx.cross_up(fast, slow):
            ctx.target_weight(ctx.symbol, 1.0)   # declare a target, not an order
        elif ctx.cross_down(fast, slow):
            ctx.target_weight(ctx.symbol, 0.0)
```

A complete runnable example — synthetic data port, immediate-fill venue,
full Research-mode session — lives in
[`examples/dual_ma.py`](examples/dual_ma.py):

```bash
python examples/dual_ma.py
```

## The research layer: factors, modelers, experiments

Layered configuration: factors, modelers and portfolio construction are
*code plus registration*; an *experiment* is pure TOML. The pipeline runs
`universe -> factor computation -> preprocessing -> modeler -> scores ->
portfolio -> target weights`, and the weights flow into the very same
`Signal -> TargetPortfolio -> RiskGate -> OrderIntent` pipeline as any
hand-written strategy — nothing bypasses the risk exit.

Built-in factors are bar-only by construction (momentum, volatility,
reversal, intraday-range classes; a real `ep_ttm` needs a fundamentals
channel the ports do not carry yet and would register through the same
surface). Built-in modelers: `equal_weight`, `linear_score` (fixed
weights) and `ic_weighted` (alias `linear_ic`, trailing mean rank-IC
weights). Preprocessing composes cross-sectionally in config order:
`winsorize`, `zscore`, `fillna`.

```toml
# experiments/momentum_value.toml
[experiment]
id = "momentum_value_2026q4"
status = "candidate"                # candidate | active | retired

[universe]
symbols = ["600000", "600009", "600016", "600028", "600030"]

[factors]
names = ["momentum_20", "volatility_20", "reversal_5"]
preprocess = ["winsorize", "zscore"]

[model]
type = "ic_weighted"
params = { lookback = 60, horizon = 5 }

[portfolio]
method = "top_n"
top_n = 2
rebalance = "monthly"

[backtest]
start = 2026-04-01
end = 2026-09-30
costs = "a_share_default"
seed = 7
```

```python
from pulsar_core import load_experiment, run_experiment, run_sweep

result = run_experiment(load_experiment("experiments/momentum_value.toml"),
                        port=my_market_data_port, venue=make_venue)
print(result.experiment_id, result.run_id, result.runtime.intents)
```

A new experiment is a new TOML file — swap the factor subset, the modeler
or `top_n` and rerun, zero code. Unknown sections or keys, wrong types and
unregistered names fail at load time (`ExperimentConfigError`).

Parameter sweeps extend the same document with `[[sweep.axis]]` entries;
`run_sweep` materializes the cartesian product into a run family that
shares one `experiment_id` while every run keeps its own configuration
and `run_id` in the `RunManifest` — ready for controlled comparison
views. A runnable demo of both entry points lives in
[`examples/factor_experiment.py`](examples/factor_experiment.py):

```bash
python examples/factor_experiment.py          # one experiment
python examples/factor_experiment.py --sweep  # a 4-run sweep family
```

Registered universes (`experiment.universe = "hs300"`) are added with
`register_universe`; a `[universe] symbols = [...]` section spells one
out inline — exactly one of the two forms per document.

## Wiring a Research run

```python
from datetime import date

from pulsar_core import (
    BacktestClock, EventBus, ReplaySession, RiskGate,
    SinglePositionCapRule, StrategyRuntime,
)

bus = EventBus(BacktestClock(<run start, Asia/Shanghai>))
runtime = StrategyRuntime(
    bus=bus,
    port=my_execution_port,          # any ExecutionPort implementation
    strategy=DualMA({"fast": 5, "slow": 20}),
    gate=RiskGate((SinglePositionCapRule(1.0),)),
    initial_cash=100_000.0,
)
session = ReplaySession(
    port=my_market_data_port,        # any MarketDataPort implementation
    symbols=["600000"],
    start=date(2026, 1, 1),
    end=date(2026, 6, 30),
    seed=42,
    config={"strategy": {"name": "dual_ma", "params": {"fast": 5, "slow": 20}}},
    bus=bus,
    on_manifest=runtime.bind_manifest,  # intents key off the manifest run id
)
result = session.run()
result.manifest.write("run.json")   # reproducibility record
print(runtime.intents, runtime.rejections, runtime.account.snapshot())
```

## Experiment lifecycle: candidate -> active -> retired

Every experiment document carries a required `experiment.status`, and the
assembly layer enforces the design's status/mode matrix:

| status | runnable modes |
|-|-|
| `candidate` | research only |
| `active` | research, paper, live |
| `retired` | none — read-only post-mortem (replays archives, never re-runs) |

```python
from pulsar_core import LifecycleError, validate_assembly

validate_assembly("paper", experiment)   # raises LifecycleError unless active
```

The registry is the `experiments/` directory itself under git — no
database. Promotion and retirement are human-driven transitions that
rewrite exactly the one `status` line in place (git history is the audit
trail) and return a `LifecycleRecord`:

```python
from pulsar_core import activate_experiment, retire_experiment

record = activate_experiment(   # 上线: needs explicit human confirmation
    "experiments/momentum_value.toml",
    reason="passed research acceptance 2026-09-30", operator="guanlan",
    confirmed=True,
)
record = retire_experiment(     # 下线
    "experiments/momentum_value.toml",
    reason="signal decayed after regime change", operator="guanlan",
)
```

A *running* session that receives a retire stops producing new order
intents immediately — `runtime.retire(reason=..., operator=...)` drops
every further declaration at the pipeline entry (existing positions are
left to the strategy's own exit rules; nothing is force-sold) — and
stamps the action into the run's `RunManifest` as a `lifecycle` audit
record (status change, reason, operator, timestamp). Every assembly also
records the git commit of the configuration it used
(`run_experiment(..., config_commit=...)`, auto-resolved from the file's
repository when omitted), so "which version went live" stays traceable.

## Run artifacts and performance accounting

Performance is computed by replaying the run's event stream — never from a
channel implementation — so backtest and live runs share one set of metric
definitions. After a run ends, `write_run_artifacts` persists the complete
run directory:

```python
from pulsar_core import write_run_artifacts

artifacts = write_run_artifacts(
    result, events=bus.journal, initial_cash=100_000.0,
    directory=f"runs/{result.run_id}",
)
```

producing, side by side:

| file | contract |
|-|-|
| `run_manifest.json` | the RunManifest reproducibility record |
| `events.parquet` | the full event journal (schema-versioned; one row per dispatched event with identity, denormalized query columns and the lossless canonical `payload`) |
| `metrics_report.json` | `schema_version` + `equity_curve` / `metrics` / `fee_attribution` |

The metrics report carries the NAV curve (one point per trading day),
total and annualized return, annualized volatility, Sharpe, max drawdown,
double-sided turnover and fee attribution (commission, stamp duty,
transfer fee, slippage drag — each also as a fraction of initial cash).
`read_event_archive` loads an archived run back into validated kernel
events, so an archived run recomputes its equity curve bit-identically —
that replay path is exactly how review and attribution tools (and the UI,
which depends only on these artifact files) consume a run.

## Reproducibility contract

- Same manifest inputs (config, seed, code version, data watermarks) rebuild
  the same `run_id` — it is the SHA-256 of their canonical JSON. Lifecycle
  audit records and the experiment config's git commit are provenance
  annotations: archived in the manifest, never part of the run-id identity.
- Two runs with identical inputs produce identical event journals and
  digests; `EventBus.journal_digest` is the bit-level identity check, and
  the emitted intents reproduce identically too.
- Data gaps relative to the trading calendar abort the run
  (`DataGapError`) — never silently skipped. Handler exceptions terminate
  the run with the journal of already-dispatched events preserved.

## Development

```bash
pip install -e ".[dev]"
pytest
mypy
```

The test suite includes the determinism check (two runs of the same replay
must match event by event), the import-purity scan of every module, and the
structural bypass tests proving a strategy cannot reach the venue except
through the risk gate.

## License

MIT

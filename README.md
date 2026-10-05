# pulsar-core

The deterministic engine core of the [Pulsar](https://github.com/vanzeph)
A-share quant system: a single-threaded event kernel, a Clock abstraction
that makes backtest and realtime runs share one loop, a bar-level historical
replay session, the RunManifest reproducibility mechanism, the strategy
framework with its `Signal -> TargetPortfolio -> RiskGate -> OrderIntent`
pipeline, and performance accounting with its run artifacts
(`events.parquet` + metrics report).

Status: alpha — the kernel, replay, manifest, strategy framework, intent
pipeline and performance accounting are in; the factor library and the full
five-rule risk set land in later releases.

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
  the same `run_id` — it is the SHA-256 of their canonical JSON.
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

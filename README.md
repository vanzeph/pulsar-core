# pulsar-core

The deterministic engine core of the [Pulsar](https://github.com/vanzeph)
A-share quant system: a single-threaded event kernel, a Clock abstraction
that makes backtest and realtime runs share one loop, a bar-level historical
replay session, and the RunManifest reproducibility mechanism.

Status: alpha — the kernel, replay skeleton and manifest are in; the
strategy framework, factor library, risk gate and performance accounting
land in later releases.

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

## Install

Requires Python 3.11+.

```bash
pip install git+https://github.com/vanzeph/pulsar-core.git
```

## Usage sketch

```python
from datetime import date

from pulsar_core import EventKind, ReplaySession
from my_lake import MyMarketDataPort  # any MarketDataPort implementation


def on_bar(event) -> None:
    ...  # future tasks plug the strategy / risk / accounting in here


session = ReplaySession(
    port=MyMarketDataPort(),
    symbols=["600000", "000001"],
    start=date(2026, 1, 1),
    end=date(2026, 9, 30),
    seed=42,
)
session.subscribe(EventKind.MARKET, on_bar)
result = session.run()          # replays day by day on the kernel loop
result.manifest.write("run.json")  # reproducibility record
print(result.run_id, result.journal_digest)
```

## Reproducibility contract

- Same manifest inputs (config, seed, code version, data watermarks) rebuild
  the same `run_id` — it is the SHA-256 of their canonical JSON.
- Two runs with identical inputs produce identical event journals and
  digests; `EventBus.journal_digest` is the bit-level identity check.
- Data gaps relative to the trading calendar abort the run
  (`DataGapError`) — never silently skipped. Handler exceptions terminate
  the run with the journal of already-dispatched events preserved.

## Development

```bash
pip install -e ".[dev]"
pytest
```

The test suite includes the determinism check (two runs of the same replay
must match event by event) and the import-purity scan of every module.

## License

MIT

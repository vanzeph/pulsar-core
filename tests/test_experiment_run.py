"""Experiment run integration: TOML in, C2 intent pipeline out.

Runs experiments end to end over the deterministic scripted port from
``conftest`` and asserts the acceptance criteria of the research layer:

* a *new experiment is zero code* — the same runner code, a different
  TOML (factor subset / modeler), a different portfolio;
* targets flow through the existing Signal -> TargetPortfolio ->
  RiskGate -> OrderIntent pipeline (rejections prove the gate is live);
* monthly rebalancing only acts on first-trading-day dates.
"""

from __future__ import annotations

from datetime import date

import pytest
from pulsar_contracts import Side

from pulsar_core import (
    ReplaySession,
    RiskGate,
    SinglePositionCapRule,
    load_experiment,
    run_experiment,
)
from conftest import FillingExecutionPort, ScriptedClosesPort

SYMBOLS = ("UP", "MILD", "FLAT", "WOBBLY", "MILD_DN", "DN")
N_BARS = 160


def closes_for(symbol: str) -> list[float]:
    """Distinct deterministic paths: momentum and volatility rank apart."""
    if symbol == "UP":
        return [100.0 * 1.02**i for i in range(N_BARS)]
    if symbol == "MILD":
        return [100.0 * 1.005**i for i in range(N_BARS)]
    if symbol == "FLAT":
        return [100.0] * N_BARS
    if symbol == "WOBBLY":
        price = 100.0
        closes = []
        for index in range(N_BARS):
            price *= 1.03 if index % 2 == 0 else 0.97
            closes.append(round(price, 4))
        return closes
    if symbol == "MILD_DN":
        return [100.0 * 0.997**i for i in range(N_BARS)]
    return [100.0 * 0.98**i for i in range(N_BARS)]  # DN


def make_port() -> ScriptedClosesPort:
    return ScriptedClosesPort({symbol: closes_for(symbol) for symbol in SYMBOLS})


def make_venue(bus):
    return FillingExecutionPort(now=lambda: bus.now)


MOMENTUM_TOML = """
[experiment]
id = "momentum_top2"

[universe]
symbols = ["UP", "MILD", "FLAT", "WOBBLY", "MILD_DN", "DN"]

[factors]
names = ["momentum_20"]
preprocess = ["winsorize", "zscore"]

[model]
type = "equal_weight"

[portfolio]
method = "top_n"
top_n = 2
rebalance = "monthly"

[backtest]
start = 2026-10-01
end = 2026-12-31
seed = 3
"""

LOW_VOL_TOML = """
[experiment]
id = "low_vol_top1"

[universe]
symbols = ["UP", "MILD", "FLAT", "WOBBLY", "MILD_DN", "DN"]

[factors]
names = ["volatility_20"]
preprocess = ["winsorize", "zscore"]

[model]
type = "linear_score"
params = { weights = { volatility_20 = 1.0 } }

[portfolio]
method = "top_n"
top_n = 1
rebalance = "monthly"

[backtest]
start = 2026-10-01
end = 2026-12-31
seed = 3
"""


def window(port: ScriptedClosesPort) -> tuple[date, date]:
    days = port.calendar(date(2026, 6, 1), date(2027, 6, 1))
    return days[-60], days[-1]


class TestRunExperiment:
    def test_momentum_experiment_holds_the_top_two(self, tmp_path) -> None:
        port = make_port()
        start, end = window(port)
        text = MOMENTUM_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        (tmp_path / "momentum.toml").write_text(text, encoding="utf-8")
        config = load_experiment(tmp_path / "momentum.toml")

        result = run_experiment(config, port=port, venue=make_venue, initial_cash=500_000.0)

        assert result.experiment_id == "momentum_top2"
        assert result.manifest.config["experiment"]["id"] == "momentum_top2"
        assert result.manifest.config["session"]["symbols"] == sorted(SYMBOLS)
        # UP and MILD have the two highest trailing-20-bar returns
        held = {position.symbol for position in result.runtime.account.snapshot().positions}
        assert held == {"UP", "MILD"}
        buys = [s.intent for s in result.runtime.submissions if s.intent.side is Side.BUY]
        assert buys, "the first rebalance must buy"
        assert {intent.symbol for intent in buys} == {"UP", "MILD"}

    def test_new_experiment_zero_code_different_factor_and_model(self, tmp_path) -> None:
        port = make_port()
        start, end = window(port)
        for name, template in (("momentum.toml", MOMENTUM_TOML), ("low_vol.toml", LOW_VOL_TOML)):
            text = template.replace("start = 2026-10-01", f"start = {start}").replace(
                "end = 2026-12-31", f"end = {end}"
            )
            (tmp_path / name).write_text(text, encoding="utf-8")

        # identical runner code, two documents: two different experiments
        momentum = run_experiment(
            load_experiment(tmp_path / "momentum.toml"),
            port=port, venue=make_venue, initial_cash=500_000.0,
        )
        low_vol = run_experiment(
            load_experiment(tmp_path / "low_vol.toml"),
            port=port, venue=make_venue, initial_cash=500_000.0,
        )
        assert momentum.experiment_id != low_vol.experiment_id
        assert momentum.run_id != low_vol.run_id
        held_momentum = {p.symbol for p in momentum.runtime.account.snapshot().positions}
        held_low_vol = {p.symbol for p in low_vol.runtime.account.snapshot().positions}
        # momentum picks the risers; the volatility model picks the zero-vol name
        assert held_momentum == {"UP", "MILD"}
        assert held_low_vol == {"FLAT"}

    def test_monthly_rebalance_only_on_first_trading_days(self, tmp_path) -> None:
        port = make_port()
        start, end = window(port)
        text = MOMENTUM_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        (tmp_path / "momentum.toml").write_text(text, encoding="utf-8")

        result = run_experiment(
            load_experiment(tmp_path / "momentum.toml"),
            port=port, venue=make_venue, initial_cash=500_000.0,
        )
        trading_days = port.calendar(start, end)
        expected_firsts = {
            day
            for day in trading_days
            if not any(
                earlier.year == day.year and earlier.month == day.month
                for earlier in trading_days
                if earlier < day
            )
        }
        assert set(result.strategy.rebalance_days) == expected_firsts
        # every emitted intent keys off the run's manifest run id
        for submission in result.runtime.submissions:
            assert submission.intent.idempotency_key.run_id == result.run_id

    def test_rerun_reproduces_run_id_and_intents(self, tmp_path) -> None:
        port = make_port()
        start, end = window(port)
        text = MOMENTUM_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        (tmp_path / "momentum.toml").write_text(text, encoding="utf-8")
        config = load_experiment(tmp_path / "momentum.toml")

        first = run_experiment(config, port=port, venue=make_venue)
        second = run_experiment(config, port=port, venue=make_venue)
        assert first.run_id == second.run_id
        assert first.run.journal_digest == second.run.journal_digest
        assert [s.intent.model_dump() for s in first.runtime.submissions] == [
            s.intent.model_dump() for s in second.runtime.submissions
        ]

    def test_risk_gate_rejects_when_cap_is_tight(self, tmp_path) -> None:
        port = make_port()
        start, end = window(port)
        text = MOMENTUM_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        (tmp_path / "momentum.toml").write_text(text, encoding="utf-8")

        result = run_experiment(
            load_experiment(tmp_path / "momentum.toml"),
            port=port,
            venue=make_venue,
            initial_cash=500_000.0,
            gate=RiskGate((SinglePositionCapRule(0.05),)),  # weights are 0.5 each
        )
        # every drafted buy breaches the 5% single-position cap and is
        # rejected with its reason recorded; nothing ever gets through
        assert result.runtime.rejections, "the gate must see the drafts"
        assert all(
            rejection.rule == "single_position_cap"
            for rejection in result.runtime.rejections
        )
        assert result.runtime.account.snapshot().positions == ()

    def test_run_experiment_requires_expansion_for_sweep_configs(self, tmp_path) -> None:
        from pulsar_core.errors import PulsarCoreError

        text = MOMENTUM_TOML + '\n[[sweep.axis]]\npath = "portfolio.top_n"\nvalues = [1, 2]\n'
        (tmp_path / "sweep.toml").write_text(text, encoding="utf-8")
        config = load_experiment(tmp_path / "sweep.toml")
        with pytest.raises(PulsarCoreError, match="declares sweep axes"):
            run_experiment(config, port=make_port(), venue=make_venue)


class TestSessionAssemblyCompatibility:
    def test_experiment_run_over_plain_replay_session_components(self, tmp_path) -> None:
        # the runner is a thin assembly: a FactorModelStrategy also plugs
        # into a hand-wired ReplaySession like any other strategy
        from pulsar_core import BacktestClock, EventBus, FactorModelStrategy, StrategyRuntime
        from pulsar_core.runner import _day_start

        port = make_port()
        start, end = window(port)
        strategy = FactorModelStrategy({start: {"UP": 1.0}})
        bus = EventBus(BacktestClock(_day_start(start)))
        venue = FillingExecutionPort(now=lambda: bus.now)
        runtime = StrategyRuntime(
            bus=bus, port=venue, strategy=strategy, initial_cash=100_000.0
        )
        session = ReplaySession(
            port=port,
            symbols=["UP", "MILD"],
            start=start,
            end=end,
            seed=1,
            bus=bus,
            on_manifest=runtime.bind_manifest,
        )
        result = session.run()
        assert result.manifest.run_id == runtime.run_id
        held = {p.symbol for p in runtime.account.snapshot().positions}
        assert held == {"UP"}

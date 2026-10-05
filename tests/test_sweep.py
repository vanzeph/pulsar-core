"""Parameter sweep tests: one template, a run family of distinct runs.

Acceptance (task C3): the sweep produces a run family whose members
share the experiment id in their RunManifest while every run keeps its
own run id — controlled comparison, zero code per run.
"""

from __future__ import annotations

from datetime import date

import pytest

from pulsar_core import (
    UNIVERSE_REGISTRY,
    expand_sweep,
    load_experiment,
    register_universe,
    run_sweep,
)
from conftest import FillingExecutionPort, ScriptedClosesPort

SWEEP_TOML = """
[experiment]
id = "sweep_family_2026q4"
description = "model type x portfolio width"
status = "candidate"

[universe]
symbols = ["UP", "MILD", "FLAT", "DN"]

[factors]
names = ["momentum_20", "volatility_20"]
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
seed = 5

[[sweep.axis]]
path = "model.type"
values = ["equal_weight", "ic_weighted"]

[[sweep.axis]]
path = "portfolio.top_n"
values = [1, 2]
"""


def closes_for(symbol: str) -> list[float]:
    n = 150
    if symbol == "UP":
        return [100.0 * 1.01**i for i in range(n)]
    if symbol == "MILD":
        return [100.0 * 1.003**i for i in range(n)]
    if symbol == "FLAT":
        return [100.0] * n
    return [100.0 * 0.99**i for i in range(n)]


@pytest.fixture(scope="module")
def port() -> ScriptedClosesPort:
    return ScriptedClosesPort({s: closes_for(s) for s in ("UP", "MILD", "FLAT", "DN")})


@pytest.fixture(scope="module")
def window(port: ScriptedClosesPort) -> tuple[date, date]:
    days = port.calendar(date(2026, 6, 1), date(2027, 6, 1))
    return days[-50], days[-1]


@pytest.fixture()
def sweep_config(tmp_path, window):
    start, end = window
    text = SWEEP_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
        "end = 2026-12-31", f"end = {end}"
    )
    path = tmp_path / "sweep.toml"
    path.write_text(text, encoding="utf-8")
    return load_experiment(path)


def make_venue(bus):
    return FillingExecutionPort(now=lambda: bus.now)


class TestExpandSweep:
    def test_cartesian_product_of_axes(self, sweep_config) -> None:
        expansions = expand_sweep(sweep_config)
        assert len(expansions) == 4
        assignments = {
            (expansion.assignments["model.type"], expansion.assignments["portfolio.top_n"])
            for expansion in expansions
        }
        assert assignments == {
            ("equal_weight", 1),
            ("equal_weight", 2),
            ("ic_weighted", 1),
            ("ic_weighted", 2),
        }
        for expansion in expansions:
            assert expansion.index in range(4)

    def test_expanded_configs_carry_mutated_values(self, sweep_config) -> None:
        for expansion in expand_sweep(sweep_config):
            raw = expansion.config.config_snapshot()
            assert raw["model"]["type"] == expansion.assignments["model.type"]
            assert raw["portfolio"]["top_n"] == expansion.assignments["portfolio.top_n"]
            # the sweep section is stripped from concrete runs
            assert "sweep" not in raw
            assert expansion.config.sweep == ()
            # the family identity is preserved
            assert expansion.config.experiment_id == sweep_config.experiment_id

    def test_labels_render_assignments(self, sweep_config) -> None:
        labels = {expansion.label for expansion in expand_sweep(sweep_config)}
        assert "model.type=equal_weight,portfolio.top_n=1" in labels
        assert "model.type=ic_weighted,portfolio.top_n=2" in labels

    def test_no_axes_single_run(self, tmp_path, window) -> None:
        start, end = window
        text = SWEEP_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        text = text.split("[[sweep.axis]]")[0]
        path = tmp_path / "plain.toml"
        path.write_text(text, encoding="utf-8")
        expansions = expand_sweep(load_experiment(path))
        assert len(expansions) == 1
        assert expansions[0].assignments == {}


class TestRunSweep:
    def test_run_family_shares_experiment_id_and_splits_run_ids(
        self, sweep_config, port
    ) -> None:
        report = run_sweep(sweep_config, port=port, make_venue=make_venue)

        assert report.experiment_id == "sweep_family_2026q4"
        assert len(report.runs) == 4
        # every manifest carries the same experiment id ...
        experiment_ids = {
            result.manifest.config["experiment"]["id"] for result in report.runs
        }
        assert experiment_ids == {"sweep_family_2026q4"}
        # ... the sweep point is stamped and distinguishes the runs ...
        points = [
            result.manifest.config["sweep"]["point"] for result in report.runs
        ]
        assert len({tuple(sorted(point.items())) for point in points}) == 4
        # ... and the run ids are pairwise distinct
        assert len(set(report.run_ids)) == 4
        assert sorted(report.run_ids) == sorted(set(report.run_ids))

    def test_each_run_actually_trades(self, sweep_config, port) -> None:
        report = run_sweep(sweep_config, port=port, make_venue=make_venue)
        for result in report.runs:
            assert result.runtime.submissions, f"run {result.run_label} produced no intents"
            assert result.run.journal_digest

    def test_top_n_axis_changes_the_held_breadth(self, sweep_config, port) -> None:
        report = run_sweep(sweep_config, port=port, make_venue=make_venue)
        by_label = {result.run_label: result for result in report.runs}
        width_one = by_label["model.type=equal_weight,portfolio.top_n=1"]
        width_two = by_label["model.type=equal_weight,portfolio.top_n=2"]
        held_one = len(width_one.runtime.account.snapshot().positions)
        held_two = len(width_two.runtime.account.snapshot().positions)
        assert held_one == 1
        assert held_two == 2


class TestUniverseRegistry:
    def test_registered_universe_round_trip(self, tmp_path, window) -> None:
        if "mini_pool" not in UNIVERSE_REGISTRY:
            register_universe("mini_pool", ["UP", "DN"])
        start, end = window
        text = SWEEP_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        text = text.replace(
            '[universe]\nsymbols = ["UP", "MILD", "FLAT", "DN"]',
            '[universe]\nname = "mini_pool"',
        )
        path = tmp_path / "registered.toml"
        path.write_text(text, encoding="utf-8")
        config = load_experiment(path)
        assert config.symbols == ("DN", "UP")

    def test_dynamic_universe_callable_resolves_against_start(self, tmp_path, window) -> None:
        if "dated_pool" not in UNIVERSE_REGISTRY:
            register_universe(
                "dated_pool", lambda as_of: ["LATE"] if as_of >= date(2026, 11, 1) else ["EARLY"]
            )
        start, end = window
        text = SWEEP_TOML.replace("start = 2026-10-01", f"start = {start}").replace(
            "end = 2026-12-31", f"end = {end}"
        )
        text = text.replace(
            '[universe]\nsymbols = ["UP", "MILD", "FLAT", "DN"]',
            '[universe]\nname = "dated_pool"',
        )
        path = tmp_path / "dated.toml"
        path.write_text(text, encoding="utf-8")
        config = load_experiment(path)
        expected = ("LATE",) if start >= date(2026, 11, 1) else ("EARLY",)
        assert config.symbols == expected

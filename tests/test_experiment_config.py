"""Experiment TOML schema and loader tests: validation fails loudly.

Every negative case asserts the documented error surface: unknown
sections/keys, missing required values, wrong types, unregistered names,
and sweep axes that do not address the template.
"""

from __future__ import annotations

from datetime import date

import pytest

from pulsar_core import (
    UNIVERSE_REGISTRY,
    EqualWeightScorer,
    ExperimentConfigError,
    TopNConstructor,
    expand_sweep,
    load_experiment,
    register_universe,
)
from pulsar_core.modelers import IcWeightedScorer, LinearScoreScorer

BASE = """
[experiment]
id = "base_case"
status = "candidate"

[universe]
symbols = ["600000", "600009", "600016"]

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
start = 2026-01-01
end = 2026-06-30
costs = "a_share_default"
seed = 11
"""


def write(tmp_path, text: str, name: str = "experiment.toml"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture(scope="module", autouse=True)
def hs300() -> None:
    if "hs300" not in UNIVERSE_REGISTRY:
        register_universe(
            "hs300", ["600000", "600009", "600016", "600028", "600030"],
            description="test universe",
        )


class TestValidDocuments:
    def test_base_document_loads_and_resolves(self, tmp_path) -> None:
        config = load_experiment(write(tmp_path, BASE))
        assert config.experiment_id == "base_case"
        assert config.symbols == ("600000", "600009", "600016")
        assert config.factor_names == ("momentum_20", "volatility_20")
        assert len(config.preprocess) == 2
        assert isinstance(config.model, EqualWeightScorer)
        assert isinstance(config.portfolio, TopNConstructor)
        assert config.portfolio.top_n == 2
        assert config.rebalance == "monthly"
        assert (config.start, config.end) == (date(2026, 1, 1), date(2026, 6, 30))
        assert config.costs == "a_share_default"
        assert config.seed == 11
        assert config.sweep == ()

    def test_design_doc_shaped_document(self, tmp_path) -> None:
        text = """
[experiment]
id = "momentum_value_2026q4"
universe = "hs300"
status = "active"

[factors]
names = ["momentum_20", "volatility_20", "reversal_5"]
preprocess = ["winsorize", "zscore"]

[model]
type = "linear_ic"
params = { lookback = 500 }

[portfolio]
method = "top_n"
top_n = 30
rebalance = "monthly"

[backtest]
start = 2020-01-01
end = 2026-09-30
costs = "a_share_default"
"""
        config = load_experiment(write(tmp_path, text))
        assert config.symbols == tuple(sorted({"600000", "600009", "600016", "600028", "600030"}))
        assert isinstance(config.model, IcWeightedScorer)
        assert config.model.lookback == 500

    def test_registered_universe_section_form(self, tmp_path) -> None:
        text = BASE.replace(
            '[universe]\nsymbols = ["600000", "600009", "600016"]',
            '[universe]\nname = "hs300"',
        )
        config = load_experiment(write(tmp_path, text))
        assert config.symbols == tuple(sorted({"600000", "600009", "600016", "600028", "600030"}))

    def test_preprocess_step_table_with_params(self, tmp_path) -> None:
        text = BASE.replace(
            'preprocess = ["winsorize", "zscore"]',
            'preprocess = [{ step = "winsorize", quantile = 0.05 }, "zscore", { step = "fillna", method = "zero" }]',
        )
        config = load_experiment(write(tmp_path, text))
        assert config.preprocess[0].quantile == pytest.approx(0.05)
        assert config.preprocess[2].method == "zero"

    def test_linear_score_with_matching_weights(self, tmp_path) -> None:
        text = BASE.replace(
            '[model]\ntype = "equal_weight"',
            '[model]\ntype = "linear_score"\nparams = { weights = { momentum_20 = 0.7, volatility_20 = 0.3 } }',
        )
        config = load_experiment(write(tmp_path, text))
        assert isinstance(config.model, LinearScoreScorer)
        assert config.model.weights == {"momentum_20": 0.7, "volatility_20": 0.3}

    def test_config_snapshot_is_a_deep_copy_of_the_document(self, tmp_path) -> None:
        config = load_experiment(write(tmp_path, BASE))
        snapshot = config.config_snapshot()
        assert snapshot["experiment"]["id"] == "base_case"
        snapshot["portfolio"]["top_n"] = 99
        assert config.config_snapshot()["portfolio"]["top_n"] == 2


class TestUnknownKeysAndSections:
    def test_unknown_top_level_section(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="unknown section.*reporting"):
            load_experiment(write(tmp_path, BASE + "\n[reporting]\nformat = 'parquet'\n"))

    @pytest.mark.parametrize(
        "section,key,value",
        [
            ("experiment", "owner", '"zhang"'),
            ("universe", "as_of", "2026-01-01"),
            ("factors", "windows", "[20]"),
            ("model", "seed", "7"),
            ("portfolio", "weighting", '"score"'),
            ("backtest", "freq", '"1d"'),
        ],
    )
    def test_unknown_key_in_each_section(self, tmp_path, section, key, value) -> None:
        header = f"[{section}]"
        assert header in BASE
        text = BASE.replace(header, f"{header}\n{key} = {value}", 1)
        with pytest.raises(ExperimentConfigError, match=r"unknown key.*\[" + section + r"\]"):
            load_experiment(write(tmp_path, text))

    def test_sweep_axis_unknown_key(self, tmp_path) -> None:
        text = BASE + '\n[[sweep.axis]]\npath = "portfolio.top_n"\nvalues = [1, 2]\nname = "width"\n'
        with pytest.raises(ExperimentConfigError, match="sweep.axis"):
            load_experiment(write(tmp_path, text))


class TestRequiredValuesAndTypes:
    def test_missing_experiment_id(self, tmp_path) -> None:
        text = BASE.replace('id = "base_case"\n', "")
        with pytest.raises(ExperimentConfigError, match="experiment.id is required"):
            load_experiment(write(tmp_path, text))

    def test_universe_forms_are_exclusive(self, tmp_path) -> None:
        text = BASE.replace(
            'id = "base_case"', 'id = "base_case"\nuniverse = "hs300"', 1
        )
        with pytest.raises(ExperimentConfigError, match="declared twice"):
            load_experiment(write(tmp_path, text))

    def test_missing_universe(self, tmp_path) -> None:
        text = BASE.replace(
            '[universe]\nsymbols = ["600000", "600009", "600016"]\n\n', ""
        )
        with pytest.raises(ExperimentConfigError, match="no universe"):
            load_experiment(write(tmp_path, text))

    def test_universe_section_name_and_symbols_exclusive(self, tmp_path) -> None:
        text = BASE.replace(
            'symbols = ["600000", "600009", "600016"]',
            'name = "hs300"\nsymbols = ["600000"]',
        )
        with pytest.raises(ExperimentConfigError, match="both name and symbols"):
            load_experiment(write(tmp_path, text))

    def test_universe_symbols_must_be_unique_strings(self, tmp_path) -> None:
        text = BASE.replace(
            'symbols = ["600000", "600009", "600016"]',
            'symbols = ["600000", "600000"]',
        )
        with pytest.raises(ExperimentConfigError, match="duplicates"):
            load_experiment(write(tmp_path, text))

    def test_unknown_registered_universe(self, tmp_path) -> None:
        text = BASE.replace(
            '[universe]\nsymbols = ["600000", "600009", "600016"]',
            '[universe]\nname = "zz500"',
        )
        with pytest.raises(ExperimentConfigError, match="unknown universe 'zz500'"):
            load_experiment(write(tmp_path, text))

    def test_empty_or_unknown_factor_names(self, tmp_path) -> None:
        text = BASE.replace(
            'names = ["momentum_20", "volatility_20"]', "names = []"
        )
        with pytest.raises(ExperimentConfigError, match="non-empty list"):
            load_experiment(write(tmp_path, text))
        text = BASE.replace(
            'names = ["momentum_20", "volatility_20"]', 'names = ["momentum_20", "ep_ttm"]'
        )
        with pytest.raises(ExperimentConfigError, match="unknown factor 'ep_ttm'"):
            load_experiment(write(tmp_path, text))

    def test_duplicate_factor_name(self, tmp_path) -> None:
        text = BASE.replace(
            'names = ["momentum_20", "volatility_20"]', 'names = ["momentum_20", "momentum_20"]'
        )
        with pytest.raises(ExperimentConfigError, match="duplicate factor"):
            load_experiment(write(tmp_path, text))

    def test_unknown_preprocess_step(self, tmp_path) -> None:
        text = BASE.replace('preprocess = ["winsorize", "zscore"]', 'preprocess = ["winsorize", "pca"]')
        with pytest.raises(ExperimentConfigError, match="unknown preprocess step 'pca'"):
            load_experiment(write(tmp_path, text))

    def test_unknown_model_type(self, tmp_path) -> None:
        text = BASE.replace('type = "equal_weight"', 'type = "gbdt"')
        with pytest.raises(ExperimentConfigError, match="unknown model 'gbdt'"):
            load_experiment(write(tmp_path, text))

    def test_portfolio_validation(self, tmp_path) -> None:
        text = BASE.replace('rebalance = "monthly"', 'rebalance = "yearly"')
        with pytest.raises(ExperimentConfigError, match="rebalance"):
            load_experiment(write(tmp_path, text))
        text = BASE.replace("top_n = 2", "top_n = 0")
        with pytest.raises(ExperimentConfigError, match="top_n"):
            load_experiment(write(tmp_path, text))
        text = BASE.replace("top_n = 2", "top_n = true")
        with pytest.raises(ExperimentConfigError, match="top_n"):
            load_experiment(write(tmp_path, text))
        text = BASE.replace('method = "top_n"', 'method = "risk_parity"')
        with pytest.raises(ExperimentConfigError, match="unknown portfolio method"):
            load_experiment(write(tmp_path, text))

    def test_backtest_dates(self, tmp_path) -> None:
        text = BASE.replace("start = 2026-01-01", "start = 2026-12-31")
        with pytest.raises(ExperimentConfigError, match="must not be after"):
            load_experiment(write(tmp_path, text))
        text = BASE.replace("start = 2026-01-01", 'start = 2026-01-01T00:00:00')
        with pytest.raises(ExperimentConfigError, match="plain TOML date"):
            load_experiment(write(tmp_path, text))
        text = BASE.replace("start = 2026-01-01", "")
        with pytest.raises(ExperimentConfigError, match="backtest.start is required"):
            load_experiment(write(tmp_path, text))

    def test_seed_must_be_integer(self, tmp_path) -> None:
        text = BASE.replace("seed = 11", 'seed = "eleven"')
        with pytest.raises(ExperimentConfigError, match="seed"):
            load_experiment(write(tmp_path, text))

    def test_missing_file(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="cannot read"):
            load_experiment(tmp_path / "nope.toml")

    def test_invalid_toml(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="invalid TOML"):
            load_experiment(write(tmp_path, "[experiment\n"))


class TestSweepAxisValidation:
    def _sweep(self, tmp_path, path: str, values: str):
        text = BASE + f"\n[[sweep.axis]]\npath = \"{path}\"\nvalues = {values}\n"
        return write(tmp_path, text)

    def test_axis_path_must_address_template(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="does not address the template"):
            load_experiment(self._sweep(tmp_path, "portfolio.top_k", "[1, 2]"))

    def test_axis_values_type_must_match(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="has type str"):
            load_experiment(self._sweep(tmp_path, "portfolio.top_n", '["one", "two"]'))

    def test_axis_needs_values(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="non-empty values list"):
            load_experiment(self._sweep(tmp_path, "portfolio.top_n", "[]"))

    def test_duplicate_axis_path(self, tmp_path) -> None:
        text = (
            BASE
            + '\n[[sweep.axis]]\npath = "portfolio.top_n"\nvalues = [1, 2]\n'
            + '[[sweep.axis]]\npath = "portfolio.top_n"\nvalues = [3]\n'
        )
        with pytest.raises(ExperimentConfigError, match="duplicate sweep axis path"):
            load_experiment(write(tmp_path, text))

    def test_expansion_revalidates_mutated_documents(self, tmp_path) -> None:
        # the template uses equal_weight; sweeping onto an unregistered
        # model must fail at expansion, listing registered models
        config = load_experiment(
            self._sweep(tmp_path, "model.type", '["equal_weight", "bogus_model"]')
        )
        with pytest.raises(ExperimentConfigError, match="unknown model 'bogus_model'"):
            expand_sweep(config)

    def test_duplicate_expanded_runs_rejected(self, tmp_path) -> None:
        config = load_experiment(self._sweep(tmp_path, "portfolio.top_n", "[2, 2]"))
        with pytest.raises(ExperimentConfigError, match="duplicates an earlier run"):
            expand_sweep(config)

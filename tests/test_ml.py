"""ML modeler tests: device backend, torch scorers, pinned artifacts.

Acceptance (task ML1, Mac CPU):

* ``mlp_torch`` runs the full experiment chain — training -> scoring ->
  portfolio construction -> the standard RiskGate exit;
* pinned-artifact inference is bit-identical across loads, and a rerun of
  the same run id reloads the artifact without retraining;
* the artifact lands in ``runs/<run_id>/model_artifact/`` with sha256s,
  and the RunManifest carries the model-artifact record;
* device resolution falls back to CPU when CUDA is unavailable;
* ``model.type = "mlp_torch"`` sweeps into a run family.

These tests require torch (``pulsar-core[ml]``) and skip without it; the
torch-free behaviors (import purity, install guidance) live in
``test_ml_no_torch.py`` and run everywhere.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

torch = pytest.importorskip("torch")

from pulsar_core import (  # noqa: E402
    MODEL_REGISTRY,
    RiskGate,
    SinglePositionCapRule,
    load_experiment,
    load_pinned,
    model_artifact_dir,
    resolve_device,
    run_experiment,
    run_sweep,
    sha256_file,
    write_run_artifacts,
)
from pulsar_core.ml import (  # noqa: E402
    HASHES_FILENAME,
    TRAINING_CONFIG_FILENAME,
    WEIGHTS_FILENAME,
    enable_determinism,
    training_environment,
)
from conftest import FillingExecutionPort, ScriptedClosesPort  # noqa: E402
from pulsar_core.session import _day_start  # noqa: E402

SYMBOLS = ("UP", "MILD", "FLAT", "WOBBLY", "MILD_DN", "DN")
N_BARS = 160


def closes_for(symbol: str) -> list[float]:
    """Distinct deterministic paths so the panel ranks symbols apart."""
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


MLP_TOML = """
[experiment]
id = "mlp_drill"
status = "candidate"

[universe]
symbols = ["UP", "MILD", "FLAT", "WOBBLY", "MILD_DN", "DN"]

[factors]
names = ["momentum_20", "volatility_20", "reversal_5"]
preprocess = ["winsorize", "zscore"]

[model]
type = "mlp_torch"
params = { device = "auto", epochs = 3, lr = 0.02, hidden = [8], lookback = 30, horizon = 3, batch_size = 64, seed = 0 }

[portfolio]
method = "top_n"
top_n = 2
rebalance = "monthly"

[backtest]
start = 2026-10-01
end = 2026-12-31
seed = 3
"""

LSTM_TOML = """
[experiment]
id = "lstm_drill"
status = "candidate"

[universe]
symbols = ["UP", "MILD", "FLAT", "WOBBLY", "MILD_DN", "DN"]

[factors]
names = ["momentum_20", "volatility_20"]
preprocess = ["winsorize", "zscore"]

[model]
type = "lstm_torch"
params = { device = "cpu", epochs = 2, lr = 0.02, hidden = 6, window = 5, lookback = 25, horizon = 3, batch_size = 64, seed = 0 }

[portfolio]
method = "top_n"
top_n = 2
rebalance = "monthly"

[backtest]
start = 2026-10-01
end = 2026-12-31
seed = 3
"""


def write_config(tmp_path, text: str, name: str = "experiment.toml"):
    port = make_port()
    days = port.calendar(date(2026, 6, 1), date(2027, 6, 1))
    start, end = days[-60], days[-1]
    text = text.replace("start = 2026-10-01", f"start = {start}").replace(
        "end = 2026-12-31", f"end = {end}"
    )
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return load_experiment(path), port


def assert_state_dicts_equal(first, second) -> None:
    assert set(first) == set(second)
    for key in first:
        assert torch.equal(first[key], second[key]), key


class TestDeviceBackend:
    def test_auto_resolves_to_a_concrete_device(self) -> None:
        effective, note = resolve_device("auto")
        assert effective in ("cuda", "cpu")
        if torch.cuda.is_available():
            assert effective == "cuda"
        else:
            assert effective == "cpu"
            assert note == ""

    def test_forced_cpu_stays_cpu(self) -> None:
        assert resolve_device("cpu") == ("cpu", "")

    def test_forced_cuda_falls_back_visibly_without_cuda(self) -> None:
        effective, note = resolve_device("cuda")
        if torch.cuda.is_available():
            assert effective == "cuda"
        else:
            assert effective == "cpu"
            assert "fell back to cpu" in note

    def test_unknown_device_is_rejected(self) -> None:
        from pulsar_core.errors import PulsarCoreError

        with pytest.raises(PulsarCoreError, match="unknown device"):
            resolve_device("tpu")

    def test_determinism_mode_and_environment_record(self) -> None:
        record = enable_determinism(seed=11, device="cpu")
        assert record["seed"] == 11
        assert record["deterministic_algorithms"] == "warn_only"
        assert "torch" in record["seeded_rngs"]
        environment = training_environment("cpu", seed=11, determinism=record)
        assert environment["torch"] == torch.__version__
        assert environment["device"] == "cpu"
        assert environment["cuda_available"] == bool(torch.cuda.is_available())
        assert environment["determinism"]["seed"] == 11


class TestMlpTraining:
    def test_two_fresh_trainings_are_bit_identical(self, tmp_path) -> None:
        from pulsar_core import FactorEngine, TopNConstructor
        from pulsar_core.factors import FACTOR_REGISTRY
        from pulsar_core.preprocess import build_step

        config, port = write_config(tmp_path, MLP_TOML)
        factors = config.factors
        preprocess = [build_step("winsorize"), build_step("zscore")]
        from pulsar_contracts import AdjustMode, Freq
        from pulsar_core.session import _bars_from_frame

        # fetch with warmup history so the factors have data before the
        # decision date (exactly what run_experiment's warmup fetch does)
        full = port.calendar(date(2026, 6, 1), config.end)
        frame = port.fetch_bars(
            sorted(SYMBOLS), full[0], full[-1], Freq.DAILY, AdjustMode.FORWARD
        )
        bars = _bars_from_frame(frame, Freq.DAILY)
        bars_by_symbol: dict[str, list] = {}
        for bar in bars:
            bars_by_symbol.setdefault(bar.symbol, []).append(bar)

        def fresh_targets() -> dict:
            scorer = MODEL_REGISTRY.resolve("mlp_torch").create(
                {
                    "epochs": 3,
                    "lr": 0.02,
                    "hidden": [8],
                    "lookback": 30,
                    "horizon": 3,
                    "batch_size": 64,
                    "seed": 0,
                    "device": "cpu",
                },
                ["momentum_20", "volatility_20", "reversal_5"],
            )
            engine = FactorEngine(
                symbols=sorted(SYMBOLS),
                bars=bars_by_symbol,
                factors=factors,
                preprocess=preprocess,
                model=scorer,
                portfolio=TopNConstructor(top_n=2),
            )
            return engine.targets([config.start]), scorer

        first_targets, first_scorer = fresh_targets()
        second_targets, second_scorer = fresh_targets()
        assert first_targets == second_targets  # bit-identical scoring
        assert first_scorer.trained and second_scorer.trained
        assert_state_dicts_equal(
            first_scorer._state_dict(), second_scorer._state_dict()
        )

    def test_unknown_param_fails_loudly(self) -> None:
        from pulsar_core.errors import PulsarCoreError

        with pytest.raises(PulsarCoreError, match="parameters are"):
            MODEL_REGISTRY.resolve("mlp_torch").create({"n_layers": 2}, ["f"])


class TestMlpExperimentEndToEnd:
    def test_train_score_portfolio_risk_exit(self, tmp_path) -> None:
        config, port = write_config(tmp_path, MLP_TOML)
        runs_root = tmp_path / "runs"

        result = run_experiment(
            config, port=port, venue=make_venue, initial_cash=500_000.0,
            runs_root=runs_root,
        )

        # the run trained fresh and archived its artifact
        assert config.model.trained
        artifact_dir = model_artifact_dir(runs_root, result.run_id)
        assert (artifact_dir / WEIGHTS_FILENAME).is_file()
        assert (artifact_dir / TRAINING_CONFIG_FILENAME).is_file()
        assert (artifact_dir / HASHES_FILENAME).is_file()

        # the RunManifest carries the model-artifact section
        record = result.manifest.model_artifact
        assert record is not None
        assert record.model_type == "mlp_torch"
        assert record.origin == "trained"
        assert record.path == str(artifact_dir)
        assert record.weights_sha256 == sha256_file(artifact_dir / WEIGHTS_FILENAME)
        assert record.config_sha256 == sha256_file(
            artifact_dir / TRAINING_CONFIG_FILENAME
        )
        assert record.environment["torch"] == torch.__version__
        assert record.environment["device"] in ("cpu", "cuda")

        # training config is the full reproducibility record
        training_config = json.loads(
            (artifact_dir / TRAINING_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert training_config["model_type"] == "mlp_torch"
        assert training_config["factor_names"] == sorted(
            ["momentum_20", "volatility_20", "reversal_5"]
        )
        assert len(training_config["standardization"]["mean"]) == 3
        assert training_config["train_samples"] > 0

        # scoring -> portfolio -> intents flowed through the risk exit
        from pulsar_contracts import Side

        buys = [
            submission.intent
            for submission in result.runtime.submissions
            if submission.intent.side is Side.BUY
        ]
        assert buys, "the first rebalance must buy through the standard gate"

    def test_tight_risk_cap_rejects_every_draft(self, tmp_path) -> None:
        config, port = write_config(tmp_path, MLP_TOML)
        result = run_experiment(
            config,
            port=port,
            venue=make_venue,
            initial_cash=500_000.0,
            gate=RiskGate((SinglePositionCapRule(0.05),)),
        )
        assert result.runtime.rejections
        assert all(
            rejection.rule == "single_position_cap"
            for rejection in result.runtime.rejections
        )
        assert result.runtime.account.snapshot().positions == ()


class TestPinnedInference:
    def test_rerun_reuses_artifact_and_reproduces_bit_identically(self, tmp_path) -> None:
        runs_root = tmp_path / "runs"

        config, port = write_config(tmp_path, MLP_TOML)
        first = run_experiment(
            config, port=port, venue=make_venue, runs_root=runs_root
        )
        assert config.model.trained

        # rerun: same config -> same run id -> pinned artifact, no retrain
        # (our own bus so the dispatched journal stays reachable below)
        from pulsar_core import BacktestClock, EventBus

        config_again, _ = write_config(tmp_path, MLP_TOML, name="again.toml")
        rerun_bus = EventBus(BacktestClock(_day_start(config_again.start)))
        second = run_experiment(
            config_again,
            port=port,
            venue=lambda bus: FillingExecutionPort(now=lambda: bus.now),
            runs_root=runs_root,
            bus=rerun_bus,
        )
        assert second.run_id == first.run_id
        assert config_again.model.trained is False
        assert config_again.model.origin.startswith("pinned:")
        assert second.run.journal_digest == first.run.journal_digest
        assert [s.intent.model_dump() for s in second.runtime.submissions] == [
            s.intent.model_dump() for s in first.runtime.submissions
        ]
        record = second.manifest.model_artifact
        assert record is not None and record.origin == "pinned"
        assert record.weights_sha256 == first.manifest.model_artifact.weights_sha256

        # the archived run directory carries the model-artifact section too
        artifacts = write_run_artifacts(
            second.run,
            events=rerun_bus.journal,
            initial_cash=1_000_000.0,
            directory=runs_root / second.run_id,
        )
        archived = json.loads(
            (runs_root / second.run_id / "run_manifest.json").read_text(encoding="utf-8")
        )
        assert archived["model_artifact"]["weights_sha256"] == record.weights_sha256
        assert (runs_root / second.run_id / "model_artifact").is_dir()
        assert artifacts.run_id == second.run_id

    def test_two_pinned_loads_score_bit_identically(self, tmp_path) -> None:
        from pulsar_core import CrossSection

        config, port = write_config(tmp_path, MLP_TOML)
        runs_root = tmp_path / "runs"
        run_experiment(config, port=port, venue=make_venue, runs_root=runs_root)
        artifact_dir = model_artifact_dir(runs_root, "x")  # placeholder, replaced next

        # locate the actual artifact dir of the single run
        artifact_dir = next((tmp_path / "runs").glob("*/model_artifact"))
        section = CrossSection(
            as_of=date(2026, 10, 1),
            values={
                "momentum_20": {"UP": 1.4, "MILD": 0.9, "FLAT": -0.1, "DN": -1.7},
                "volatility_20": {"UP": 0.8, "MILD": -0.3, "FLAT": -1.2, "DN": 1.1},
                "reversal_5": {"UP": -0.4, "MILD": 0.2, "FLAT": 0.0, "DN": 0.6},
            },
        )

        scores = []
        for _ in range(2):
            state_dict, training_config = load_pinned(artifact_dir)
            scorer = MODEL_REGISTRY.resolve("mlp_torch").create({}, ["x"])
            scorer._adopt_pinned(state_dict, training_config)
            scores.append(scorer.score(section, None))
        assert scores[0] == scores[1]  # bit-for-bit (exact float equality)
        assert set(scores[0]) == {"UP", "MILD", "FLAT", "DN"}

    def test_tampered_weights_are_refused(self, tmp_path) -> None:
        from pulsar_core.errors import PulsarCoreError

        config, port = write_config(tmp_path, MLP_TOML)
        runs_root = tmp_path / "runs"
        run_experiment(config, port=port, venue=make_venue, runs_root=runs_root)
        artifact_dir = next((tmp_path / "runs").glob("*/model_artifact"))
        weights = artifact_dir / WEIGHTS_FILENAME
        original = weights.read_bytes()
        weights.write_bytes(original + b"\x00")
        with pytest.raises(PulsarCoreError, match="fails its pinned sha256"):
            load_pinned(artifact_dir)

    def test_missing_artifact_directory_is_refused(self, tmp_path) -> None:
        from pulsar_core.errors import PulsarCoreError

        with pytest.raises(PulsarCoreError, match="not found"):
            load_pinned(tmp_path / "nope")

    def test_explicit_artifact_param_scores_without_training(self, tmp_path) -> None:
        config, port = write_config(tmp_path, MLP_TOML)
        runs_root = tmp_path / "runs"
        run_experiment(config, port=port, venue=make_venue, runs_root=runs_root)
        artifact_dir = next((tmp_path / "runs").glob("*/model_artifact"))

        pinned_toml = MLP_TOML.replace(
            "params = { device = \"auto\", epochs = 3",
            f'params = {{ artifact = "{artifact_dir}", device = "auto", epochs = 3',
        )
        pinned_config, port = write_config(tmp_path, pinned_toml, name="pinned.toml")
        result = run_experiment(
            pinned_config, port=port, venue=make_venue, runs_root=tmp_path / "runs2"
        )
        assert pinned_config.model.trained is False
        assert pinned_config.model.origin.startswith("pinned:")
        assert result.runtime.submissions  # scored from the pinned weights


class TestLstmModeler:
    def test_lstm_experiment_runs_and_pinned_scores_match(self, tmp_path) -> None:
        runs_root = tmp_path / "runs"
        config, port = write_config(tmp_path, LSTM_TOML)
        result = run_experiment(
            config, port=port, venue=make_venue, runs_root=runs_root
        )
        assert config.model.trained
        artifact_dir = model_artifact_dir(runs_root, result.run_id)
        assert (artifact_dir / WEIGHTS_FILENAME).is_file()
        training_config = json.loads(
            (artifact_dir / TRAINING_CONFIG_FILENAME).read_text(encoding="utf-8")
        )
        assert training_config["model_type"] == "lstm_torch"
        assert training_config["architecture"]["kind"] == "lstm"
        assert training_config["architecture"]["window"] == 5

        # rerun reloads the pinned artifact (no retrain), same digest
        config_again, _ = write_config(tmp_path, LSTM_TOML, name="again.toml")
        second = run_experiment(
            config_again, port=port, venue=make_venue, runs_root=runs_root
        )
        assert second.run_id == result.run_id
        assert config_again.model.trained is False
        assert second.run.journal_digest == result.run.journal_digest


class TestSweepWithMlp:
    def test_sweep_expands_mlp_into_a_run_family(self, tmp_path) -> None:
        sweep_toml = MLP_TOML.replace(
            'params = { device = "auto", epochs = 3',
            'params = { device = "cpu", epochs = 3',
        ) + '\n[[sweep.axis]]\npath = "model.params.epochs"\nvalues = [2, 3]\n'
        config, port = write_config(tmp_path, sweep_toml, name="sweep.toml")
        report = run_sweep(
            config, port=port, make_venue=make_venue, runs_root=tmp_path / "runs"
        )
        assert len(report.runs) == 2
        assert len(set(report.run_ids)) == 2
        epochs_by_run = {}
        for run in report.runs:
            assert run.manifest.config["model"]["params"]["epochs"] in (2, 3)
            epochs_by_run[run.run_id] = run.manifest.config["model"]["params"]["epochs"]
            artifact_dir = model_artifact_dir(tmp_path / "runs", run.run_id)
            assert (artifact_dir / WEIGHTS_FILENAME).is_file(), run.run_id
            assert run.manifest.model_artifact is not None
        assert sorted(epochs_by_run.values()) == [2, 3]

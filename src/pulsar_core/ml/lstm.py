"""``lstm_torch`` — the sequence modeler over rolling factor windows.

Each sample is one symbol's ``window``-long sequence of preprocessed
factor rows ending at an evaluation date, labeled with that date's
``horizon``-bar forward return (z-scored cross-sectionally); an LSTM
reads the sequence and the final hidden state feeds a linear head
(滚动窗口因子序列 → 打分). Inference scores the current cross-section
from each symbol's trailing ``window`` of rows — assembled from the
same preprocessed panel the cross-section came from.

Experiment TOML::

    [model]
    type = "lstm_torch"
    params = { device = "auto", epochs = 30, lr = 0.01, hidden = 16,
               window = 10, lookback = 120, horizon = 5,
               batch_size = 256, seed = 0 }
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from ..errors import PulsarCoreError
from ..modelers import FactorHistoryView, ModelDefinition, ModelScorer
from .base import TorchModelScorer
from .dataset import sequence_samples

__all__ = ["LstmTorchScorer", "lstm_torch_definition"]

_ALLOWED_PARAMS: tuple[str, ...] = (
    "device",
    "epochs",
    "lr",
    "hidden",
    "seed",
    "lookback",
    "horizon",
    "window",
    "batch_size",
    "artifact",
    "retrain",
)

_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "epochs": 30,
    "lr": 0.01,
    "hidden": 16,
    "seed": 0,
    "lookback": 120,
    "horizon": 5,
    "window": 10,
    "batch_size": 256,
    "retrain": False,
}


def _checked(key: str, value: Any) -> Any:
    int_keys = {"epochs", "seed", "lookback", "horizon", "window", "batch_size", "hidden"}
    if key in int_keys:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PulsarCoreError(
                f"lstm_torch {key} must be an integer, got {value!r}"
            )
        return int(value)
    if key == "lr":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PulsarCoreError(f"lstm_torch lr must be a number, got {value!r}")
        return float(value)
    if key == "device":
        if not isinstance(value, str) or value not in ("auto", "cuda", "cpu"):
            raise PulsarCoreError(
                f"lstm_torch device must be auto/cuda/cpu, got {value!r}"
            )
        return value
    if key == "artifact":
        if value is not None and (not isinstance(value, str) or not value):
            raise PulsarCoreError(
                f"lstm_torch artifact must be a pinned artifact directory path, got {value!r}"
            )
        return value
    if key == "retrain":
        if not isinstance(value, bool):
            raise PulsarCoreError(f"lstm_torch retrain must be a boolean, got {value!r}")
        return value
    raise PulsarCoreError(f"lstm_torch has no parameter {key!r}")  # pragma: no cover


def _parse_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(_ALLOWED_PARAMS))
    if unknown:
        raise PulsarCoreError(
            f"model 'lstm_torch' parameters are {sorted(_ALLOWED_PARAMS)}, "
            f"got {unknown}"
        )
    resolved: dict[str, Any] = {key: value for key, value in _DEFAULTS.items()}
    resolved["artifact"] = None
    for key, value in params.items():
        resolved[key] = _checked(key, value)
    for key in ("epochs", "batch_size", "lookback", "horizon", "window"):
        if resolved[key] < 1:
            raise PulsarCoreError(f"lstm_torch {key} must be >= 1")
    if resolved["hidden"] < 1:
        raise PulsarCoreError("lstm_torch hidden must be >= 1")
    if resolved["lr"] <= 0.0:
        raise PulsarCoreError("lstm_torch lr must be > 0")
    return resolved


class LstmHeadNet:
    """LSTM + linear head, built lazily around torch's own module classes."""

    def __init__(self, *, input_dim: int, hidden: int) -> None:
        torch = _torch()
        self.lstm = torch.nn.LSTM(
            input_size=input_dim, hidden_size=hidden, batch_first=True
        )
        self.head = torch.nn.Linear(hidden, 1)

    def __call__(self, sequences: Any) -> Any:
        output, _hidden = self.lstm(sequences)
        return self.head(output[:, -1, :])

    def parameters(self) -> Any:
        return list(self.lstm.parameters()) + list(self.head.parameters())

    def state_dict(self) -> dict[str, Any]:
        return {
            "lstm": self.lstm.state_dict(),
            "head": self.head.state_dict(),
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        self.lstm.load_state_dict(dict(state_dict["lstm"]))
        self.head.load_state_dict(dict(state_dict["head"]))

    def to(self, device: Any) -> "LstmHeadNet":
        self.lstm.to(device)
        self.head.to(device)
        return self

    def train(self) -> None:
        self.lstm.train()
        self.head.train()

    def eval(self) -> None:
        self.lstm.eval()
        self.head.eval()


class LstmTorchScorer(TorchModelScorer):
    """Sequence modeler: rolling factor windows in, preference score out."""

    name = "lstm_torch"

    def __init__(self, params: Mapping[str, Any]) -> None:
        super().__init__(_parse_params(params))
        self.warmup_bars = (
            int(self.params["lookback"])
            + int(self.params["horizon"])
            + int(self.params["window"])
            + 1
        )

    # -- architecture ---------------------------------------------------------

    def _build_network(self, input_dim: int) -> Any:
        return LstmHeadNet(input_dim=input_dim, hidden=int(self.params["hidden"]))

    def _network_from_config(self, config: Mapping[str, Any]) -> Any:
        architecture = config.get("architecture") or {}
        hidden = int(architecture.get("hidden") or self.params["hidden"])
        input_dim = len(config.get("factor_names") or ())
        return LstmHeadNet(input_dim=input_dim, hidden=hidden)

    def _architecture(self) -> dict[str, Any]:
        return {
            "kind": "lstm",
            "hidden": int(self.params["hidden"]),
            "window": int(self.params["window"]),
            "input_dim": len(self._feature_names),
        }

    # -- samples & tensors -------------------------------------------------------

    def _collect_samples(
        self, history: FactorHistoryView, factors: Sequence[str]
    ) -> tuple[list[Any], "tuple[date, date] | None"]:
        samples, window = sequence_samples(
            history,
            factors=factors,
            lookback=int(self.params["lookback"]),
            horizon=int(self.params["horizon"]),
            window=int(self.params["window"]),
        )
        return list(samples), window

    def _features_to_tensor(self, samples: Sequence[Any]) -> Any:
        torch = _torch()
        return torch.tensor(
            [
                [[float(value) for value in row] for row in sequence]
                for sequence, _label in samples
            ],
            dtype=torch.float32,
        )

    def _rows_to_tensor(self, rows: Mapping[str, Mapping[str, float]]) -> Any:
        raise NotImplementedError("lstm_torch scores sequences, not flat rows")

    def _input_dim(self, features: Any) -> int:
        return int(features.shape[2])  # [n, window, d]

    def _read_outputs(self, outputs: Any) -> list[float]:
        return [float(value) for value in outputs.squeeze(-1).tolist()]

    # -- sequence-aware inference -------------------------------------------------

    def score(
        self, section: Any, history: FactorHistoryView
    ) -> dict[str, float]:
        from ..modelers import CrossSection

        assert isinstance(section, CrossSection)
        rows = section.complete_rows()
        if not rows:
            return {}
        factors = section.factors
        if not self._model:
            self._ensure_model(history, factors)
        if not self._model:
            count = len(factors)
            return {
                symbol: sum(row[factor] for factor in factors) / count
                for symbol, row in rows.items()
            }
        if tuple(factors) != self._feature_names:
            raise PulsarCoreError(
                f"model '{self.name}' was built for factors "
                f"{list(self._feature_names)} but scored on {list(factors)}"
            )
        sequences = self._inference_sequences(section, history, rows)
        if not sequences:
            return {}
        return self._forward_sequences(sequences)

    def _inference_sequences(
        self,
        section: Any,
        history: FactorHistoryView,
        rows: Mapping[str, Mapping[str, float]],
    ) -> dict[str, list[list[float]]]:
        """Each scorable symbol's trailing ``window`` factor rows.

        The window walks *backwards from the section's own values* over
        the history view's preprocessed panel; a symbol missing a row on
        any window date drops out of this scoring round (missing data is
        never guessed around). When the panel cannot supply a full window
        yet the round falls back to equal weight — the same young-history
        contract as the flat modelers.
        """
        factors = self._feature_names
        window = int(self.params["window"])
        common = list(history.common_dates())
        if len(common) < window:
            return {}
        window_dates = common[-window:]
        panel: dict[date, dict[str, dict[str, "float | None"]]] = {}
        for day in window_dates:
            columns: dict[str, dict[str, "float | None"]] = {}
            for factor in factors:
                merged = dict(history.preprocessed_values(factor, day))
                if day == section.as_of:
                    override = section.values.get(factor)
                    if override is not None:
                        merged.update(override)
                columns[factor] = merged
            panel[day] = columns
        symbols = sorted(rows)
        sequences: dict[str, list[list[float]]] = {}
        for symbol in symbols:
            steps: list[list[float]] = []
            complete = True
            for day in window_dates:
                row: list[float] = []
                for factor in factors:
                    value = panel[day][factor].get(symbol)
                    if value is None:
                        complete = False
                        break
                    row.append(float(value))
                if not complete:
                    break
                steps.append(row)
            if complete:
                sequences[symbol] = steps
        return sequences

    def _forward_sequences(
        self, sequences: Mapping[str, list[list[float]]]
    ) -> dict[str, float]:
        torch = _torch()
        symbols = sorted(sequences)
        tensor = torch.tensor(
            [
                [[float(value) for value in row] for row in sequences[symbol]]
                for symbol in symbols
            ],
            dtype=torch.float32,
        )
        mean = torch.tensor(self._mean, dtype=torch.float32)
        std = torch.tensor(self._std, dtype=torch.float32)
        tensor = (tensor - mean) / std
        with torch.no_grad():
            outputs = self._model(tensor.to(self._device))
        values = self._read_outputs(outputs)
        if len(values) != len(symbols):
            raise PulsarCoreError(
                f"model '{self.name}' returned {len(values)} scores for "
                f"{len(symbols)} symbols"
            )
        return dict(zip(symbols, values))


def _torch() -> Any:
    from .backend import require_torch

    return require_torch()


def _create_lstm_torch(
    params: Mapping[str, Any], factor_names: Sequence[str]
) -> ModelScorer:
    return LstmTorchScorer(params)


def lstm_torch_definition() -> ModelDefinition:
    """The registry entry for ``lstm_torch``."""
    return ModelDefinition(
        name="lstm_torch",
        create=_create_lstm_torch,
        description="LSTM over rolling factor-window sequences "
        "(torch; optional extra pulsar-core[ml])",
    )

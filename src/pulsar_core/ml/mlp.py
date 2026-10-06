"""``mlp_torch`` — the cross-sectional feed-forward modeler.

One hidden-stack MLP over one decision date's preprocessed factor row
per symbol (输入因子面板样本 → 每日打分). Training assembles, from the
history view, the trailing ``lookback`` evaluation dates' complete rows
against their ``horizon``-bar forward returns (z-scored per date), fits
with the shared deterministic loop, then scores every later
cross-section from the same weights — the artifact the run archives.

Experiment TOML::

    [model]
    type = "mlp_torch"
    params = { device = "auto", epochs = 50, lr = 0.01,
               hidden = [16], lookback = 120, horizon = 5,
               batch_size = 256, seed = 0 }

``device`` resolves per the backend policy (``auto`` → CUDA when
available, else CPU; forced CUDA falls back visibly). ``artifact``
points at a pinned artifact directory for inference-only runs, and
``retrain = true`` forces fresh training even when the runner found a
pinned artifact for the run id.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from ..errors import PulsarCoreError
from ..modelers import FactorHistoryView, ModelDefinition, ModelScorer
from .base import TorchModelScorer
from .dataset import flat_samples

__all__ = ["MlpTorchScorer", "mlp_torch_definition"]

_ALLOWED_PARAMS: tuple[str, ...] = (
    "device",
    "epochs",
    "lr",
    "hidden",
    "seed",
    "lookback",
    "horizon",
    "batch_size",
    "artifact",
    "retrain",
)

_DEFAULTS: dict[str, Any] = {
    "device": "auto",
    "epochs": 50,
    "lr": 0.01,
    "hidden": [16],
    "seed": 0,
    "lookback": 120,
    "horizon": 5,
    "batch_size": 256,
    "retrain": False,
}


def _parse_params(params: Mapping[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(params) - set(_ALLOWED_PARAMS))
    if unknown:
        raise PulsarCoreError(
            f"model 'mlp_torch' parameters are {sorted(_ALLOWED_PARAMS)}, "
            f"got {unknown}"
        )
    resolved = {key: value for key, value in _DEFAULTS.items()}
    resolved["artifact"] = None
    for key, value in params.items():
        resolved[key] = _checked(key, value)
    hidden = resolved["hidden"]
    assert isinstance(hidden, list)
    if any(size < 1 for size in hidden):
        raise PulsarCoreError("mlp_torch hidden layer sizes must be >= 1")
    for key in ("epochs", "batch_size", "lookback", "horizon"):
        if resolved[key] < 1:
            raise PulsarCoreError(f"mlp_torch {key} must be >= 1")
    if resolved["lr"] <= 0.0:
        raise PulsarCoreError("mlp_torch lr must be > 0")
    return resolved


def _checked(key: str, value: Any) -> Any:
    int_keys = {"epochs", "seed", "lookback", "horizon", "batch_size"}
    if key in int_keys:
        if isinstance(value, bool) or not isinstance(value, int):
            raise PulsarCoreError(f"mlp_torch {key} must be an integer, got {value!r}")
        return int(value)
    if key == "lr":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise PulsarCoreError(f"mlp_torch lr must be a number, got {value!r}")
        return float(value)
    if key == "hidden":
        if not isinstance(value, list) or not value:
            raise PulsarCoreError(
                f"mlp_torch hidden must be a non-empty list of sizes, got {value!r}"
            )
        sizes: list[int] = []
        for size in value:
            if isinstance(size, bool) or not isinstance(size, int):
                raise PulsarCoreError(
                    f"mlp_torch hidden sizes must be integers, got {size!r}"
                )
            sizes.append(int(size))
        return sizes
    if key == "device":
        if not isinstance(value, str) or value not in ("auto", "cuda", "cpu"):
            raise PulsarCoreError(
                f"mlp_torch device must be auto/cuda/cpu, got {value!r}"
            )
        return value
    if key == "artifact":
        if value is not None and (not isinstance(value, str) or not value):
            raise PulsarCoreError(
                f"mlp_torch artifact must be a pinned artifact directory path, got {value!r}"
            )
        return value
    if key == "retrain":
        if not isinstance(value, bool):
            raise PulsarCoreError(f"mlp_torch retrain must be a boolean, got {value!r}")
        return value
    raise PulsarCoreError(f"mlp_torch has no parameter {key!r}")  # pragma: no cover


class MlpTorchScorer(TorchModelScorer):
    """Cross-sectional MLP: factor row in, preference score out."""

    name = "mlp_torch"

    def __init__(self, params: Mapping[str, Any]) -> None:
        super().__init__(_parse_params(params))
        self.warmup_bars = (
            int(self.params["lookback"]) + int(self.params["horizon"]) + 1
        )

    # -- architecture ---------------------------------------------------------

    def _build_network(self, input_dim: int) -> Any:
        torch = _torch()
        sizes = list(self.params["hidden"])
        layers: list[Any] = []
        previous = input_dim
        for size in sizes:
            layers.extend((torch.nn.Linear(previous, int(size)), torch.nn.ReLU()))
            previous = int(size)
        layers.append(torch.nn.Linear(previous, 1))
        return torch.nn.Sequential(*layers)

    def _network_from_config(self, config: Mapping[str, Any]) -> Any:
        architecture = config.get("architecture") or {}
        hidden = list(architecture.get("hidden") or self.params["hidden"])
        input_dim = len(config.get("factor_names") or ())
        saved = list(self.params["hidden"])
        self.params["hidden"] = [int(size) for size in hidden]
        try:
            return self._build_network(input_dim)
        finally:
            self.params["hidden"] = saved

    def _architecture(self) -> dict[str, Any]:
        return {
            "kind": "mlp",
            "hidden": [int(size) for size in self.params["hidden"]],
            "input_dim": len(self._feature_names),
        }

    # -- samples & tensors ------------------------------------------------------

    def _collect_samples(
        self, history: FactorHistoryView, factors: Sequence[str]
    ) -> tuple[list[Any], "tuple[date, date] | None"]:
        samples, window = flat_samples(
            history,
            factors=factors,
            lookback=int(self.params["lookback"]),
            horizon=int(self.params["horizon"]),
        )
        return list(samples), window

    def _features_to_tensor(self, samples: Sequence[Any]) -> Any:
        torch = _torch()
        return torch.tensor(
            [[float(value) for value in features] for features, _label in samples],
            dtype=torch.float32,
        )

    def _rows_to_tensor(self, rows: Mapping[str, Mapping[str, float]]) -> Any:
        torch = _torch()
        return torch.tensor(
            [[float(row[factor]) for factor in self._feature_names] for row in rows.values()],
            dtype=torch.float32,
        )

    def _read_outputs(self, outputs: Any) -> list[float]:
        return [float(value) for value in outputs.squeeze(-1).tolist()]


def _torch() -> Any:
    from .backend import require_torch

    return require_torch()


def _create_mlp_torch(
    params: Mapping[str, Any], factor_names: Sequence[str]
) -> ModelScorer:
    return MlpTorchScorer(params)


def mlp_torch_definition() -> ModelDefinition:
    """The registry entry for ``mlp_torch``."""
    return ModelDefinition(
        name="mlp_torch",
        create=_create_mlp_torch,
        description="cross-sectional MLP over the preprocessed factor panel "
        "(torch; optional extra pulsar-core[ml])",
    )

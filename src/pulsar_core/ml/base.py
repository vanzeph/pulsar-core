"""The torch scorer base: artifact lifecycle around one lazy-trained model.

:class:`TorchModelScorer` implements the ML half of the modeler contract
(core-engine design): *train once per run* on the factor panel the
history view serves (clipped at the first decision date — no look-ahead),
then score every cross-section from those pinned weights; persist the
weights + training config + sha256 as a versioned artifact the runner
archives under the run directory, and reload them hash-verified on reruns
so a rerun never retrains (unless the config explicitly sets
``retrain = true``).

Young-history fallback mirrors the IC-weighted modeler: when the panel
cannot supply training samples yet, scoring falls back to the equal-weight
mean of the section — deterministic and documented, never an error
mid-run.

Subclasses provide the architecture (:meth:`_build_network`), the sample
builder and the inference forward; torch is imported exclusively inside
methods, so importing this module on a torch-free install is harmless.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping, Sequence

from ..errors import PulsarCoreError
from ..modelers import CrossSection, FactorHistoryView, ModelScorer
from .artifacts import save_model_artifact
from .backend import enable_determinism, resolve_device, training_environment

__all__ = ["TorchModelScorer", "train_network"]


class TorchModelScorer(ModelScorer):
    """Base of the registered torch modelers (``mlp_torch`` / ``lstm_torch``)."""

    name = "torch_model"

    def __init__(self, params: Mapping[str, Any]) -> None:
        #: Resolved, validated constructor parameters (JSON-native).
        self.params: dict[str, Any] = dict(params)
        self._model: Any = None
        self._device: str = ""
        self._feature_names: tuple[str, ...] = ()
        self._mean: list[float] = []
        self._std: list[float] = []
        self._environment: dict[str, Any] = {}
        self._origin: str = ""
        self._train_window: "tuple[date, date] | None" = None
        self._sample_count: int = 0

    # -- ModelScorer ------------------------------------------------------

    def score(
        self, section: CrossSection, history: FactorHistoryView
    ) -> dict[str, float]:
        rows = section.complete_rows()
        if not rows:
            return {}
        factors = section.factors
        if not self._model:
            self._ensure_model(history, factors)
        if not self._model:
            # young history: deterministic equal-weight fallback
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
        return self._forward(rows)

    # -- artifact lifecycle -------------------------------------------------

    @property
    def trained(self) -> bool:
        """Whether this scorer currently holds a freshly trained model."""
        return bool(self._origin == "trained")

    @property
    def origin(self) -> str:
        """Provenance of the current model: ``trained`` / ``pinned:<dir>`` / ``""``."""
        return self._origin

    @property
    def force_retrain(self) -> bool:
        """Whether the config explicitly asked to retrain over pinned reuse."""
        return bool(self.params.get("retrain", False))

    @property
    def environment(self) -> dict[str, Any]:
        """The recorded training environment of the loaded/pinned model."""
        return dict(self._environment)

    @property
    def artifact_ready(self) -> bool:
        """Whether :meth:`save_artifact` would persist a usable model."""
        return bool(self._model)

    def load_pinned(self, artifact_dir: str | Any) -> None:
        """Load (hash-verified) a pinned artifact as this scorer's model.

        Inference-only: no training happens on a pinned load. The
        artifact's architecture, factor order and standardization stats
        rebuild the exact network the weights were trained into.
        """
        from .artifacts import load_pinned

        state_dict, config = load_pinned(artifact_dir)
        self._adopt_pinned(state_dict, config)
        self._origin = f"pinned:{artifact_dir}"

    def save_artifact(self, artifact_dir: str | Any) -> dict[str, str]:
        """Persist weights + training config + sha256 under ``artifact_dir``."""
        if not self._model:
            raise PulsarCoreError(
                f"model '{self.name}' has no trained or pinned weights to archive"
            )
        return save_model_artifact(
            artifact_dir,
            model_type=self.name,
            state_dict=self._state_dict(),
            training_config=self._training_config(),
        )

    # -- hooks for subclasses ------------------------------------------------

    def _build_network(self, input_dim: int) -> Any:
        """Construct the torch module for ``input_dim`` features."""
        raise NotImplementedError

    def _collect_samples(
        self, history: FactorHistoryView, factors: Sequence[str]
    ) -> tuple[list[Any], "tuple[date, date] | None"]:
        """Training samples from the history view (feature rows / sequences)."""
        raise NotImplementedError

    def _features_to_tensor(self, samples: Sequence[Any]) -> Any:
        """Stack sample features into one input tensor ``[n, ...]``."""
        raise NotImplementedError

    def _input_dim(self, features: Any) -> int:
        """The network's feature width of a stacked sample tensor.

        Flat panels are ``[n, d]`` (dim 1); sequence panels are
        ``[n, window, d]`` (dim 2). Subclasses with sequence samples
        override this.
        """
        return int(features.shape[1])

    def _rows_to_tensor(self, rows: Mapping[str, Mapping[str, float]]) -> Any:
        """Stack inference rows into one input tensor ``[m, ...]``."""
        raise NotImplementedError

    def _read_outputs(self, outputs: Any) -> list[float]:
        """Pull one python float per row out of the network output."""
        raise NotImplementedError

    def _architecture(self) -> dict[str, Any]:
        """Architecture facts recorded in the artifact config."""
        raise NotImplementedError

    def _network_from_config(self, config: Mapping[str, Any]) -> Any:
        """Rebuild the network an artifact's config describes."""
        raise NotImplementedError

    def _state_dict(self) -> dict[str, Any]:
        assert self._model is not None  # guarded by callers
        state_dict: dict[str, Any] = dict(self._model.state_dict())
        return state_dict

    # -- internals -------------------------------------------------------------

    def _ensure_model(
        self, history: FactorHistoryView, factors: Sequence[str]
    ) -> None:
        # fail with install guidance before any ML work happens
        _require_torch()
        artifact = self.params.get("artifact")
        if artifact and not self.force_retrain:
            self.load_pinned(artifact)
            return
        self._train(history, tuple(factors))

    def _train(self, history: FactorHistoryView, factors: tuple[str, ...]) -> None:
        torch = _require_torch()
        samples, window = self._collect_samples(history, factors)
        if not samples:
            return  # fallback path in score()
        requested = str(self.params.get("device", "auto"))
        device, note = resolve_device(requested)
        seed = int(self.params.get("seed", 0))
        determinism = enable_determinism(seed, device)
        features = self._features_to_tensor(samples)
        labels = torch.tensor(
            [float(label) for _, label in samples], dtype=torch.float64
        ).to(torch.float32)
        # per-feature statistics over every leading (sample/sequence) dim,
        # so flat [n, d] panels and sequence [n, w, d] panels standardize alike
        leading = tuple(range(features.dim() - 1))
        mean = features.mean(dim=leading)
        std = features.std(dim=leading, unbiased=False)
        std = torch.where(std > 0, std, torch.ones_like(std))
        features = (features - mean) / std
        network = self._build_network(self._input_dim(features)).to(device)
        train_network(
            network,
            features.to(device),
            labels.to(device),
            epochs=int(self.params.get("epochs", 50)),
            lr=float(self.params.get("lr", 0.01)),
            batch_size=int(self.params.get("batch_size", 256)),
            seed=seed,
            device=device,
        )
        network.eval()
        self._model = network
        self._device = device
        self._feature_names = factors
        self._mean = [float(value) for value in mean.tolist()]
        self._std = [float(value) for value in std.tolist()]
        self._environment = training_environment(
            device, seed=seed, determinism=determinism
        )
        if note:
            self._environment["device_note"] = note
        self._origin = "trained"
        self._train_window = window
        self._sample_count = len(samples)

    def _forward(self, rows: Mapping[str, Mapping[str, float]]) -> dict[str, float]:
        torch = _require_torch()
        symbols = sorted(rows)
        tensor = self._rows_to_tensor({symbol: rows[symbol] for symbol in symbols})
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

    def _adopt_pinned(self, state_dict: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        torch = _require_torch()
        if config.get("model_type") != self.name:
            raise PulsarCoreError(
                f"artifact holds a {config.get('model_type')!r} model but this "
                f"scorer is '{self.name}'"
            )
        factors = tuple(str(name) for name in config.get("factor_names", ()))
        if not factors:
            raise PulsarCoreError("pinned artifact config carries no factor names")
        network = self._network_from_config(config)
        network.load_state_dict(dict(state_dict))
        requested = str(self.params.get("device", config.get("params", {}).get("device", "auto")))
        device, _note = resolve_device(requested)
        network = network.to(device)
        network.eval()
        self._model = network
        self._device = device
        self._feature_names = factors
        standardization = config.get("standardization") or {}
        self._mean = [float(value) for value in standardization.get("mean", [])]
        self._std = [float(value) for value in standardization.get("std", [])]
        environment = config.get("environment")
        self._environment = (
            dict(environment) if isinstance(environment, Mapping) else {}
        )
        self._train_window = None
        self._sample_count = 0

    def _training_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model_type": self.name,
            "params": dict(self.params),
            "factor_names": list(self._feature_names),
            "standardization": {"mean": self._mean, "std": self._std},
            "environment": dict(self._environment),
        }
        config["architecture"] = self._architecture()
        if self._train_window is not None:
            config["train_window"] = {
                "first": self._train_window[0].isoformat(),
                "last": self._train_window[1].isoformat(),
            }
        config["train_samples"] = self._sample_count
        return config


def train_network(
    network: Any,
    features: Any,
    labels: Any,
    *,
    epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    device: str,
) -> None:
    """Deterministic supervised loop: Adam, seeded batches, fixed order.

    The batch permutation comes from one CPU :class:`torch.Generator`
    seeded with ``seed`` — the same data, parameters and device replay the
    same update sequence (strictly bit-identical on CPU; best-effort on
    CUDA per the design's determinism contract).
    """
    torch = _require_torch()
    if epochs < 1 or batch_size < 1:
        raise PulsarCoreError("epochs and batch_size must be >= 1")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    count = features.shape[0]
    network.train()
    for _epoch in range(epochs):
        permutation = torch.randperm(count, generator=generator).to(device)
        for start in range(0, count, batch_size):
            batch = permutation[start : start + batch_size]
            optimizer.zero_grad()
            predictions = network(features[batch]).squeeze(-1)
            loss = torch.nn.functional.mse_loss(predictions, labels[batch])
            loss.backward()
            optimizer.step()


def _require_torch() -> Any:
    from .backend import require_torch

    return require_torch()

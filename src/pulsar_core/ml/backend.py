"""Torch backend access: lazy import, device resolution, determinism.

PyTorch is an *optional* extra (``pip install pulsar-core[ml]``): the
default install carries zero torch, so ``import pulsar_core`` and every
non-ML code path stay torch-free. This module is the single place that
touches ``torch`` — always inside functions, never at module import time
(same discipline as the pyarrow lazy imports in :mod:`pulsar_core.artifacts`).

Device policy (core-engine design, 本地运行约束): ``auto`` picks CUDA
when available and falls back to CPU otherwise; ``cuda`` / ``cpu`` force a
device, with an explicit request for an unavailable ``cuda`` falling back
to CPU (recorded, never a mid-run crash). Determinism: training enables
deterministic algorithms in warn-only mode
(``torch.use_deterministic_algorithms(True, warn_only=True)``) and seeds every RNG
the training loop touches; the training environment (torch / CUDA
versions, device, seed) is recorded into the model artifact and the
RunManifest so a rerun can prove which stack produced the weights.
"""

from __future__ import annotations

import os
import random
from typing import Any, Mapping

from ..errors import PulsarCoreError

__all__ = [
    "ML_EXTRA_INSTALL_HINT",
    "require_torch",
    "resolve_device",
    "enable_determinism",
    "training_environment",
]

#: The user-facing fix for a missing torch install.
ML_EXTRA_INSTALL_HINT = "pip install 'pulsar-core[ml]'"

#: Devices a model ``params.device`` may name.
_KNOWN_DEVICES: tuple[str, ...] = ("auto", "cuda", "cpu")


def require_torch() -> Any:
    """Import and return the ``torch`` module, or fail with install guidance.

    ML modelers call this at the moment they actually need torch (model
    construction, training, inference) — never at registration time, so an
    experiment TOML referencing ``mlp_torch`` on a torch-free install fails
    with a clear pointer to the extra instead of breaking the import.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - exercised via subprocess test
        raise PulsarCoreError(
            "the torch ML backend is not installed; install the optional "
            f"extra with: {ML_EXTRA_INSTALL_HINT} "
            "(Windows CUDA wheels: pip install torch --index-url "
            "https://download.pytorch.org/whl/cu124)"
        ) from exc
    return torch


def resolve_device(requested: "str | None" = None) -> tuple[str, str]:
    """Resolve the effective device for a requested device name.

    Returns ``(effective, note)``: ``effective`` is one of ``"cuda"`` /
    ``"cpu"``; ``note`` documents what happened (empty when the request
    was honored as asked). ``auto`` prefers CUDA and falls back to CPU;
    ``cuda`` on a machine without CUDA falls back to CPU *visibly* — the
    run continues on CPU and the note lands in the training environment
    record, never a silent surprise.
    """
    torch = require_torch()
    name = (requested or "auto").strip().lower()
    if name not in _KNOWN_DEVICES:
        raise PulsarCoreError(
            f"unknown device {requested!r}; expected one of {list(_KNOWN_DEVICES)}"
        )
    cuda_available = bool(torch.cuda.is_available())
    if name == "cuda" and not cuda_available:
        return "cpu", "cuda requested but torch.cuda.is_available() is False; fell back to cpu"
    if name == "auto":
        return ("cuda", "") if cuda_available else ("cpu", "")
    return name, ""


def enable_determinism(seed: int, device: str) -> dict[str, Any]:
    """Put the torch stack into deterministic mode for one training run.

    Applies ``torch.use_deterministic_algorithms(True, warn_only=True)`` — the
    design contract: CPU training is *strictly* deterministic, GPU is
    best-effort (no hard failure when an op lacks a deterministic kernel,
    the warning surfaces instead). Seeds torch (plus the CUDA RNGs when
    present), the Python RNG and numpy when it is importable, and sets
    ``CUBLAS_WORKSPACE_CONFIG`` so cuBLAS can run deterministically on
    CUDA >= 10.2. Returns the facts recorded alongside the artifact.
    """
    torch = require_torch()
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise PulsarCoreError(f"seed must be an integer, got {seed!r}")
    torch.use_deterministic_algorithms(True, warn_only=True)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(seed)
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    seeded_rngs = ["python_random", "torch"]
    try:
        import numpy

        numpy.random.seed(seed)
        seeded_rngs.append("numpy")
    except ImportError:  # pragma: no cover - numpy rides in with pandas
        pass
    return {
        "deterministic_algorithms": "warn_only",
        "seed": seed,
        "seeded_rngs": seeded_rngs,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def training_environment(
    device: str, *, seed: int, determinism: "Mapping[str, Any] | None" = None
) -> dict[str, Any]:
    """The training-side environment record for artifacts and manifests.

    Everything that shapes the numeric output of a training run and can
    differ between machines: torch version, CUDA runtime version, CUDA
    availability, the effective device and the determinism settings. The
    record rides with the model artifact and is stamped into the
    RunManifest's model-artifact section (as provenance — it does not fork
    the run id).
    """
    torch = require_torch()
    environment: dict[str, Any] = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": device,
        "seed": seed,
    }
    if determinism:
        environment["determinism"] = dict(determinism)
    if device == "cuda" and torch.cuda.is_available():
        environment["gpu_name"] = torch.cuda.get_device_name(0)
    return environment

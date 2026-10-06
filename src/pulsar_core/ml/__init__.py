"""pulsar-core ML modelers: torch behind an optional extra.

This subpackage adds the ML half of the modeler registry (core-engine
design, 模型形态): the cross-sectional ``mlp_torch`` and the sequence
``lstm_torch`` scorers, their versioned training artifacts (weights +
config + sha256, pinned hash-verified inference) and the device /
determinism / environment-record backend they share.

Importing :mod:`pulsar_core.ml` (and therefore ``import pulsar_core``)
never imports torch: registration into :data:`~pulsar_core.modelers.MODEL_REGISTRY`
is metadata-only and every torch import sits inside the methods that
execute ML work. A torch-free install fails only when an ML modeler is
actually used, with guidance to install ``pulsar-core[ml]``.
"""

from __future__ import annotations

from ..modelers import MODEL_REGISTRY
from .artifacts import (
    ARTIFACT_DIRNAME,
    ARTIFACT_FILENAMES,
    TRAINING_CONFIG_FILENAME,
    HASHES_FILENAME,
    WEIGHTS_FILENAME,
    load_pinned,
    model_artifact_dir,
    save_model_artifact,
    sha256_file,
)
from .backend import (
    ML_EXTRA_INSTALL_HINT,
    enable_determinism,
    require_torch,
    resolve_device,
    training_environment,
)
from .base import TorchModelScorer, train_network
from .lstm import LstmTorchScorer, lstm_torch_definition
from .mlp import MlpTorchScorer, mlp_torch_definition

__all__ = [
    # backend
    "ML_EXTRA_INSTALL_HINT",
    "require_torch",
    "resolve_device",
    "enable_determinism",
    "training_environment",
    # scorers
    "TorchModelScorer",
    "MlpTorchScorer",
    "LstmTorchScorer",
    # artifacts
    "ARTIFACT_DIRNAME",
    "WEIGHTS_FILENAME",
    "TRAINING_CONFIG_FILENAME",
    "HASHES_FILENAME",
    "ARTIFACT_FILENAMES",
    "sha256_file",
    "model_artifact_dir",
    "save_model_artifact",
    "load_pinned",
    # training helper
    "train_network",
    # registration
    "register_torch_modelers",
]


#: Module-level singletons so repeated registration passes the registry's
#: identical-item idempotency check (fresh objects would be a name clash).
_MLP_DEFINITION = mlp_torch_definition()
_LSTM_DEFINITION = lstm_torch_definition()


def register_torch_modelers() -> None:
    """Register ``mlp_torch`` and ``lstm_torch`` in the modeler registry.

    Idempotent: the same definition objects re-register as no-ops (the
    registry tolerates exact re-registration of an identical item).
    """
    for definition in (_MLP_DEFINITION, _LSTM_DEFINITION):
        MODEL_REGISTRY.register(definition, replace=True)


register_torch_modelers()

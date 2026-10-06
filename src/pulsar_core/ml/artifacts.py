"""Versioned model artifacts: weights + training config + sha256, pinned.

A trained ML modeler is a *run asset* (core-engine design, 本地运行约
束): its weights, the exact training configuration and the training
environment are archived together under the run directory ::

    runs/<run_id>/model_artifact/
        weights.pt             torch state_dict of the trained model
        training_config.json   model type, params, factor names, feature
                               standardization, environment record
        artifact.json          file -> sha256 map (the tamper check)

:func:`load_pinned` reads an artifact back *verifying every hash first* —
inference only ever runs from a pinned, hash-checked artifact, so a
rerun of the same run id loads the same bytes and scores bit-identically
(the strict determinism promise; GPU is best-effort per the same design).
Training itself never runs on a pinned load unless the experiment config
explicitly asks for a ``retrain``.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from ..errors import PulsarCoreError

__all__ = [
    "ARTIFACT_DIRNAME",
    "WEIGHTS_FILENAME",
    "TRAINING_CONFIG_FILENAME",
    "HASHES_FILENAME",
    "ARTIFACT_FILENAMES",
    "sha256_file",
    "model_artifact_dir",
    "save_model_artifact",
    "load_pinned",
    "read_pinned_manifest",
]

#: Directory name of the model artifact inside a run directory.
ARTIFACT_DIRNAME = "model_artifact"

#: Canonical file names of the artifact contract. The hash-manifest name
#: is deliberately distinct from the run-manifest contract in
#: :mod:`pulsar_core.artifacts` (``run_manifest.json``).
WEIGHTS_FILENAME = "weights.pt"
TRAINING_CONFIG_FILENAME = "training_config.json"
HASHES_FILENAME = "artifact.json"

ARTIFACT_FILENAMES: tuple[str, ...] = (WEIGHTS_FILENAME, TRAINING_CONFIG_FILENAME)


def sha256_file(path: str | Path) -> str:
    """The sha256 hex digest of one file's bytes."""
    digest = sha256()
    with open(Path(path), "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_artifact_dir(runs_root: str | Path, run_id: str) -> Path:
    """``<runs_root>/<run_id>/model_artifact`` — the artifact location."""
    return Path(runs_root) / run_id / ARTIFACT_DIRNAME


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def save_model_artifact(
    artifact_dir: str | Path,
    *,
    model_type: str,
    state_dict: Mapping[str, Any],
    training_config: Mapping[str, Any],
) -> dict[str, str]:
    """Write one complete artifact directory; return the file->sha256 map.

    ``state_dict`` is a torch state dict (passed through untouched — the
    torch import lives in the caller); ``training_config`` must be
    JSON-native. Writing is all-or-nothing in practice: the hash manifest
    lands last, and :func:`load_pinned` refuses directories whose manifest
    or hashes do not check out.
    """
    import torch

    out = Path(artifact_dir)
    out.mkdir(parents=True, exist_ok=True)
    config = dict(training_config)
    config["model_type"] = model_type
    weights_path = out / WEIGHTS_FILENAME
    config_path = out / TRAINING_CONFIG_FILENAME
    torch.save(dict(state_dict), weights_path)
    config_path.write_text(_canonical_json(config) + "\n", encoding="utf-8")
    hashes = {
        WEIGHTS_FILENAME: sha256_file(weights_path),
        TRAINING_CONFIG_FILENAME: sha256_file(config_path),
    }
    (out / HASHES_FILENAME).write_text(
        _canonical_json({"files": hashes, "model_type": model_type}) + "\n",
        encoding="utf-8",
    )
    return hashes


def load_pinned(artifact_dir: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a pinned artifact: verify hashes, return ``(state_dict, config)``.

    Fails loudly (never scores from an unverified artifact) when the
    directory is incomplete, the hash manifest is missing, any file's
    sha256 does not match, or the config's recorded model type disagrees
    with the artifact manifest.
    """
    torch = _require_torch_for_load()
    root = Path(artifact_dir)
    if not root.is_dir():
        raise PulsarCoreError(f"model artifact directory not found: {root}")
    manifest_path = root / HASHES_FILENAME
    if not manifest_path.is_file():
        raise PulsarCoreError(
            f"model artifact at {root} has no {HASHES_FILENAME}; refusing to "
            "load an unverified artifact"
        )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PulsarCoreError(f"model artifact manifest unreadable ({root}): {exc}") from exc
    recorded: dict[str, str] = {
        str(name): str(digest) for name, digest in (manifest.get("files") or {}).items()
    }
    for name in ARTIFACT_FILENAMES:
        path = root / name
        if not path.is_file():
            raise PulsarCoreError(f"model artifact at {root} misses {name}")
        digest = recorded.get(name)
        if not digest:
            raise PulsarCoreError(
                f"model artifact manifest at {root} has no hash for {name}"
            )
        actual = sha256_file(path)
        if actual != digest:
            raise PulsarCoreError(
                f"model artifact file {name} fails its pinned sha256 "
                f"(recorded {digest}, found {actual}); the artifact was "
                "modified or corrupted — refusing inference"
            )
    try:
        config = json.loads((root / TRAINING_CONFIG_FILENAME).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PulsarCoreError(f"model artifact config unreadable ({root}): {exc}") from exc
    if config.get("model_type") != manifest.get("model_type"):
        raise PulsarCoreError(
            f"model artifact at {root} mixes config of {config.get('model_type')!r} "
            f"with manifest of {manifest.get('model_type')!r}"
        )
    state_dict = torch.load(
        root / WEIGHTS_FILENAME, map_location="cpu", weights_only=True
    )
    if not isinstance(state_dict, dict):
        raise PulsarCoreError(
            f"model artifact weights at {root} did not deserialize into a state dict"
        )
    return dict(state_dict), dict(config)


def _require_torch_for_load() -> Any:
    from .backend import require_torch

    return require_torch()


def read_pinned_manifest(artifact_dir: str | Path) -> dict[str, Any]:
    """Parse (unverified) an artifact's hash manifest — provenance reads.

    Callers that only need the recorded hashes (e.g. stamping a reused
    artifact into the RunManifest after :func:`load_pinned` already
    verified every hash) read the manifest without re-hashing; anything
    that *executes* the model must go through :func:`load_pinned`.
    """
    path = Path(artifact_dir) / HASHES_FILENAME
    if not path.is_file():
        raise PulsarCoreError(f"model artifact manifest not found: {path}")
    try:
        parsed = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PulsarCoreError(f"model artifact manifest unreadable ({path}): {exc}") from exc
    return dict(parsed)

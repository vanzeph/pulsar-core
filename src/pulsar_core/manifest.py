"""RunManifest: the reproducibility record of one run.

A manifest archives everything that can change the outcome of a run:
configuration snapshot, data-lake partition watermarks (data version), code
version (git commit), random seed and — in later tasks — risk parameters.
The run id is derived from the SHA-256 of the canonical form of those
inputs, so rebuilding a manifest from the same inputs yields the same run
id, and rerunning the same manifest on the same code version must produce a
bit-identical result (equity, fills, metrics in later tasks; here: the event
journal digest).

The manifest carries no wall-clock timestamps on purpose: generation is
deterministic so that "same inputs" implies "same manifest, byte for byte".
"""

from __future__ import annotations

import json
import os
import subprocess
from datetime import date, datetime
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable, Mapping

from pydantic import Field

from pulsar_contracts import Bar, ContractModel

from .lifecycle import LifecycleRecord

__all__ = [
    "RunManifest",
    "ModelArtifactRecord",
    "load_manifest",
    "code_version",
    "bars_watermark",
]

_SCHEMA_VERSION = 1
_RUN_ID_HEX_CHARS = 16

#: Run modes defined by the architecture baseline.
MODES = ("research", "paper", "live")


def _canonical_json(value: Any) -> str:
    """Deterministic JSON: sorted keys, no whitespace, no NaN, ISO dates."""

    def _encode(obj: Any) -> str:
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"non-JSON-native value in manifest inputs: {obj!r}")

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        default=_encode,
    )


def code_version() -> str:
    """Best-effort identifier of the currently running pulsar-core code.

    Resolution order: the ``PULSAR_CODE_VERSION`` environment override, the
    git HEAD commit of the source tree containing this package, the
    installed distribution version, then ``"unknown"``. Purely local — this
    function never touches the network.
    """
    override = os.environ.get("PULSAR_CODE_VERSION")
    if override:
        return override
    package_dir = Path(__file__).resolve()
    for candidate in package_dir.parents:
        if (candidate / ".git").exists():
            try:
                result = subprocess.run(  # noqa: S603 - fixed argv, local repo
                    ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=10,
                )
            except (OSError, subprocess.SubprocessError):
                break
            commit = result.stdout.strip()
            if commit:
                return commit
            break
    try:
        return version("pulsar-core")
    except PackageNotFoundError:  # pragma: no cover - source checkout without install
        return "unknown"


#: Alias so :meth:`RunManifest.build` can call the resolver even though its
#: parameter shadows the public name inside the function body.
_default_code_version = code_version


class ModelArtifactRecord(ContractModel):
    """Provenance of one run's versioned ML model artifact (模型工件段).

    Recorded when a torch modeler trained during the run (``origin`` =
    ``trained``) or scored from a pinned artifact (``origin`` =
    ``pinned``). Like the rest of the audit overlay it annotates the run:
    the hashes pin *which bytes* produced the scores, they do not fork
    the run id (a deterministic rerun rebuilds the same weights and the
    same run id before this section exists).
    """

    model_type: str = Field(min_length=1)
    path: str = Field(min_length=1)
    weights_sha256: str = Field(min_length=1)
    config_sha256: str = Field(min_length=1)
    origin: str = Field(min_length=1)
    environment: dict[str, Any] = Field(default_factory=dict)


def bars_watermark(bars: Iterable[Bar]) -> dict[str, str]:
    """Data-lake watermark per bar partition, computed from fetched bars.

    Returns ``{"bars/<freq>/<symbol>": <latest bar ts, ISO>}`` keyed in
    sorted order. This is the data-version side of the manifest: reruns must
    see the same latest timestamp per partition or they are not the same
    run.
    """
    latest: dict[str, str] = {}
    for bar in bars:
        key = f"bars/{bar.freq.value}/{bar.symbol}"
        current = latest.get(key)
        stamp = bar.ts.isoformat()
        if current is None or stamp > current:
            latest[key] = stamp
    return {key: latest[key] for key in sorted(latest)}


class RunManifest(ContractModel):
    """Immutable record of one run's reproducibility inputs.

    Two fields are a deliberate *audit overlay*, outside the run-id
    identity: ``lifecycle`` (上下线 actions observed by this run — e.g. a
    mid-run retire — with reason, operator and timestamp) and
    ``config_commit`` (the git commit of the experiment configuration the
    assembly used, so "which version went live" stays traceable). They
    annotate governance provenance; they never fork run ids, and a retire
    record can therefore be appended to a running session's manifest
    after its run id was already derived.

    ``model_artifact`` joins that overlay for ML runs (模型工件段): the
    pinned weights/config hashes of the modeler that produced the run's
    scores. It appears after training resolved, so it likewise cannot
    participate in the run-id derivation.
    """

    schema_version: int = Field(default=_SCHEMA_VERSION)
    run_id: str = Field(min_length=8)
    mode: str
    config: dict[str, Any] = Field(default_factory=dict)
    seed: int
    code_version: str = Field(min_length=1)
    data_watermarks: dict[str, str] = Field(default_factory=dict)
    config_commit: "str | None" = None
    lifecycle: list[LifecycleRecord] = Field(default_factory=list)
    model_artifact: "ModelArtifactRecord | None" = None

    # -- construction -------------------------------------------------------

    @classmethod
    def build(
        cls,
        *,
        mode: str,
        seed: int,
        config: Mapping[str, Any] | None = None,
        code_version: str | None = None,
        data_watermarks: Mapping[str, str] | None = None,
        config_commit: "str | None" = None,
        lifecycle: "Iterable[LifecycleRecord] | None" = None,
    ) -> "RunManifest":
        """Derive a manifest (and its run id) from the run's inputs.

        The run id is the first :data:`_RUN_ID_HEX_CHARS` hex characters of
        the SHA-256 over the canonical JSON of all identity fields, so the
        same inputs always rebuild the same manifest with the same run id.
        ``config_commit`` and ``lifecycle`` are audit provenance and are
        stored but excluded from the derivation (see the class docstring).
        """
        if mode not in MODES:
            raise ValueError(f"unknown run mode {mode!r}; expected one of {MODES}")
        inputs: dict[str, Any] = {
            "schema_version": _SCHEMA_VERSION,
            "mode": mode,
            "config": dict(config) if config else {},
            "seed": seed,
            "code_version": code_version
            if code_version is not None
            else _default_code_version(),
            "data_watermarks": dict(sorted((data_watermarks or {}).items())),
        }
        # Fail fast on non-JSON-native config values, then normalize them to
        # their JSON form (dates -> ISO strings) so a built manifest and its
        # loaded round-trip compare equal field by field.
        try:
            inputs["config"] = json.loads(_canonical_json(inputs["config"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"manifest inputs must be JSON-native: {exc}") from exc
        return cls(
            run_id=cls._derive_run_id(inputs),
            config_commit=config_commit,
            lifecycle=list(lifecycle) if lifecycle else [],
            **inputs,
        )

    @staticmethod
    def _derive_run_id(inputs: Mapping[str, Any]) -> str:
        canonical = _canonical_json(dict(inputs))
        return sha256(canonical.encode("utf-8")).hexdigest()[:_RUN_ID_HEX_CHARS]

    # -- identity -----------------------------------------------------------

    def identity_json(self) -> str:
        """Canonical JSON of the inputs that determine the run id.

        Excludes ``run_id`` itself; two manifests with equal
        ``identity_json`` are the same run.
        """
        inputs = {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "config": self.config,
            "seed": self.seed,
            "code_version": self.code_version,
            "data_watermarks": dict(sorted(self.data_watermarks.items())),
        }
        return _canonical_json(inputs)

    def matches_code(self, current_version: str) -> bool:
        """Whether this manifest was produced by ``current_version`` code.

        The reproducibility promise holds for a manifest rerun on its own
        code version; callers should check this before rerunning.
        """
        return self.code_version == current_version

    # -- lifecycle audit overlay ------------------------------------------------

    def record_lifecycle(self, record: LifecycleRecord) -> None:
        """Append one lifecycle audit record to this manifest, in place.

        The one sanctioned late mutation of a manifest: a running session
        that receives a retire (下线) stamps the action into the manifest
        it was built from, so the archived document answers "why did this
        run stop trading". The record joins the audit overlay only — the
        run id and identity stay untouched by construction.
        """
        self.lifecycle.append(record)

    def record_model_artifact(self, record: ModelArtifactRecord) -> None:
        """Stamp this run's model artifact provenance into the manifest.

        The ML counterpart of the lifecycle overlay: the runner records
        the artifact it trained (or the pinned one it scored from) after
        the run's outcome is fixed, so the archived manifest answers
        "which exact model bytes produced these scores" without forking
        the run id. Setting it twice is a bug and raises.
        """
        if self.model_artifact is not None:
            raise ValueError(
                "this manifest already carries a model artifact record"
            )
        # ContractModel is frozen; this is the sanctioned late-stamp escape
        # hatch (mirroring how lifecycle records join their list field).
        object.__setattr__(self, "model_artifact", record)

    # -- persistence --------------------------------------------------------

    def to_json(self) -> str:
        """Deterministic JSON document of the whole manifest."""
        return _canonical_json(self.model_dump(mode="json"))

    def write(self, path: str | Path) -> Path:
        """Write the manifest document to ``path`` (parent dirs created)."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.to_json() + "\n", encoding="utf-8")
        return target


def load_manifest(path: str | Path) -> RunManifest:
    """Load a manifest from its JSON document.

    Raises ``pydantic.ValidationError`` on malformed documents, so a
    hand-edited or truncated manifest fails loudly instead of replaying
    against guessed inputs.
    """
    manifest: RunManifest = RunManifest.model_validate_json(
        Path(path).read_text(encoding="utf-8")
    )
    return manifest

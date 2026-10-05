"""Experiment configuration lifecycle: candidate -> active -> retired.

Core-engine design (模型配置生命周期（上下线）): experiment configurations
carry a status field enforced by the assembly layer. The registry is the
``experiments`` directory itself — git versioning provides audit and
rollback, no database is introduced.

Status semantics (the authoritative table):

===========  =============================  ===============================
status       meaning                       runnable modes
===========  =============================  ===============================
candidate    under research                research
active       in production                 research, paper, live
retired      taken offline                 none (read-only post-mortem)
===========  =============================  ===============================

* 上线 (candidate -> active): after the experiment passes engineering
  acceptance in Research and a human confirms, the status is changed.
  Paper / Live assembly rejects every non-active configuration outright.
* 下线 (active -> retired): running sessions immediately stop producing
  new order intents; existing positions are left to the strategy's own
  exit rules (the engine never force-sells). The action and its reason
  are written into the current run's RunManifest.
* Every assembly records the git commit of the configuration it used, so
  "which version went live" stays traceable.

This module owns the state machine and its audit record
(:class:`LifecycleRecord`); the runtime-side retire checkpoint that stops
new intents lives on :class:`~pulsar_core.runtime.StrategyRuntime` and the
manifest-side archival on :class:`~pulsar_core.manifest.RunManifest`.
"""

from __future__ import annotations

import subprocess
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from pydantic import Field, field_validator

from pulsar_contracts import ContractModel, SHANGHAI_TZ

from .errors import LifecycleError

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    from .experiment import ExperimentConfig

__all__ = [
    "RUN_MODES",
    "ExperimentStatus",
    "STATUS_ALLOWED_MODES",
    "LifecycleAction",
    "LifecycleRecord",
    "validate_assembly",
    "experiment_commit",
    "activate_experiment",
    "retire_experiment",
]

#: Run modes defined by the architecture baseline (mirrors
#: :data:`pulsar_core.manifest.MODES`; kept here so this module carries no
#: import cycle with the manifest — a test pins the two together).
RUN_MODES: tuple[str, ...] = ("research", "paper", "live")


class ExperimentStatus:
    """Namespace of the three lifecycle statuses (plain string values).

    Kept as constants rather than an enum so TOML-loaded values compare,
    serialize and round-trip as plain strings everywhere (config
    snapshots, manifests) without enum-coercion surprises.
    """

    CANDIDATE = "candidate"
    ACTIVE = "active"
    RETIRED = "retired"

    ALL: tuple[str, ...] = (CANDIDATE, ACTIVE, RETIRED)


#: Which run modes each status may be assembled in (the design's table).
STATUS_ALLOWED_MODES: dict[str, tuple[str, ...]] = {
    ExperimentStatus.CANDIDATE: ("research",),
    ExperimentStatus.ACTIVE: ("research", "paper", "live"),
    ExperimentStatus.RETIRED: (),  # read-only post-mortem: no runnable modes
}

#: The two human-driven transitions of the state machine.
LifecycleAction = Literal["activate", "retire"]

_HEX_DIGITS = frozenset("0123456789abcdef")


def _is_sha(value: str) -> bool:
    """A git commit sha: 7..40 lowercase hex characters."""
    return 7 <= len(value) <= 40 and all(char in _HEX_DIGITS for char in value)


def _now() -> datetime:
    """Wall-clock now in the system's canonical timezone (Asia/Shanghai)."""
    return datetime.now(SHANGHAI_TZ)


class LifecycleRecord(ContractModel):
    """One lifecycle transition, archived into the run's RunManifest.

    Fields mirror the design's 下线 semantics — the status change, the
    reason, the operator, the timestamp — plus the git commit of the
    experiment configuration the action was taken against (provenance:
    "which version went live"). ``ts`` accepts a :class:`~datetime.datetime`
    (normalized to its ISO form) or a ready ISO string.
    """

    action: LifecycleAction
    from_status: str = Field(min_length=1)
    to_status: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    operator: str = Field(min_length=1)
    ts: str = Field(min_length=1)
    config_commit: str | None = Field(default=None, pattern=r"^[0-9a-f]{7,40}$")

    @field_validator("ts", mode="before")
    @classmethod
    def _ts_accepts_datetime(cls, value: object) -> object:
        if isinstance(value, datetime):
            return value.isoformat()
        return value


# -- assembly validation ------------------------------------------------------


def validate_assembly(mode: str, experiment: "ExperimentConfig") -> None:
    """Enforce the status/mode matrix at assembly time.

    Given the run ``mode`` (research | paper | live) and an experiment
    configuration, reject the assembly unless the configuration's status
    is allowed to run in that mode. Paper and Live therefore only ever
    assemble ``active`` experiments; ``candidate`` runs in Research only;
    ``retired`` runs nowhere (read-only post-mortem replays archives, it
    does not re-run). The error message carries status, mode and reason.
    """
    mode_text = str(mode)
    if mode_text not in RUN_MODES:
        raise LifecycleError(
            f"unknown run mode {mode!r}; expected one of {RUN_MODES}",
            mode=mode_text,
        )
    status = experiment.status
    allowed = experiment.allowed_modes
    if mode_text in allowed:
        return
    if status == ExperimentStatus.RETIRED:
        reason = (
            "the experiment is retired and admits no runnable mode "
            "(read-only post-mortem only)"
        )
    elif status == ExperimentStatus.CANDIDATE:
        reason = (
            "the experiment is still a candidate (research only); promote it "
            "to active — after engineering acceptance and human confirmation — "
            "before assembling paper/live"
        )
    else:  # pragma: no cover - active allows every mode, the branch is unreachable
        reason = "the experiment's status does not allow this mode"
    raise LifecycleError(
        f"assembly rejected: experiment {experiment.experiment_id!r} has status "
        f"{status!r}, which may not run in mode {mode_text!r} "
        f"(allowed modes: {list(allowed) or 'none'}); {reason}",
        status=status,
        mode=mode_text,
    )


# -- registry (experiments directory + git) ------------------------------------


def experiment_commit(path: str | Path, *, commit: str | None = None) -> str:
    """Resolve the git commit of an experiment configuration file.

    Two supported routes (the registry is the experiments directory under
    git version control):

    * ``commit`` passed explicitly — validated as a 7..40 hex sha — wins;
    * otherwise the git repository containing ``path`` is queried for its
      ``HEAD`` commit (walking up from the file's directory, exactly like
      :func:`~pulsar_core.manifest.code_version`).

    Returns ``"unknown"`` when the file lives outside any git repository
    or git is unavailable — mirroring the ``code_version`` fallback, never
    a network call.
    """
    if commit is not None:
        stripped = commit.strip().lower()
        if not _is_sha(stripped):
            raise LifecycleError(
                f"explicit config commit {commit!r} is not a git sha "
                f"(7..40 hex characters)"
            )
        return stripped
    file_path = Path(path)
    directory = file_path.parent if file_path.is_file() else file_path
    for candidate in [directory, *directory.parents]:
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
            resolved = result.stdout.strip()
            if resolved:
                return resolved
            break
    return "unknown"


def _perform_transition(
    path: str | Path,
    *,
    action: LifecycleAction,
    expected_from: str,
    to_status: str,
    reason: str,
    operator: str,
    ts: datetime | None,
    commit: str | None,
) -> LifecycleRecord:
    """Shared machinery of :func:`activate_experiment` / :func:`retire_experiment`."""
    from .experiment import load_experiment  # local import: experiment imports this module

    if not isinstance(reason, str) or not reason.strip():
        raise LifecycleError(f"{action} requires a non-empty reason")
    if not isinstance(operator, str) or not operator.strip():
        raise LifecycleError(f"{action} requires a non-empty operator (who acted)")
    file_path = Path(path)
    config = load_experiment(file_path)
    if config.status != expected_from:
        raise LifecycleError(
            f"illegal transition {action}: experiment {config.experiment_id!r} is "
            f"{config.status!r}, expected {expected_from!r}; legal transitions are "
            f"candidate->active (activate) and active->retired (retire)",
            status=config.status,
        )
    stamp = ts if ts is not None else _now()
    config_commit = experiment_commit(file_path, commit=commit)
    text = file_path.read_text(encoding="utf-8")
    file_path.write_text(
        _rewrite_status(text, to_status), encoding="utf-8"
    )
    return LifecycleRecord(
        action=action,
        from_status=config.status,
        to_status=to_status,
        reason=reason.strip(),
        operator=operator.strip(),
        ts=stamp,
        config_commit=None if config_commit == "unknown" else config_commit,
    )


def activate_experiment(
    path: str | Path,
    *,
    reason: str,
    operator: str,
    confirmed: bool = False,
    ts: datetime | None = None,
    commit: str | None = None,
) -> LifecycleRecord:
    """上线: promote a candidate experiment to active (candidate -> active).

    Promotion requires explicit human confirmation (``confirmed=True``) —
    the design allows it only after the experiment reached engineering
    acceptance in Research. The registry file's ``status`` is rewritten in
    place (everything else byte-preserved; git history provides the audit
    trail) and the resulting :class:`LifecycleRecord` is what assembly
    layers archive into the RunManifest.
    """
    if not confirmed:
        raise LifecycleError(
            "activation requires explicit human confirmation: pass "
            "confirmed=True after reviewing the experiment's research "
            "acceptance evidence"
        )
    return _perform_transition(
        path,
        action="activate",
        expected_from=ExperimentStatus.CANDIDATE,
        to_status=ExperimentStatus.ACTIVE,
        reason=reason,
        operator=operator,
        ts=ts,
        commit=commit,
    )


def retire_experiment(
    path: str | Path,
    *,
    reason: str,
    operator: str,
    ts: datetime | None = None,
    commit: str | None = None,
) -> LifecycleRecord:
    """下线: take an active experiment offline (active -> retired).

    Rewrites the registry file's ``status`` to ``retired`` and returns the
    audit record (reason, operator, timestamp, config commit). Stopping
    the *running sessions* of the experiment is a separate, immediate
    concern handled by :meth:`StrategyRuntime.retire
    <pulsar_core.runtime.StrategyRuntime.retire>` — existing positions are
    deliberately not force-sold here; they follow the strategy's own exit
    rules.
    """
    return _perform_transition(
        path,
        action="retire",
        expected_from=ExperimentStatus.ACTIVE,
        to_status=ExperimentStatus.RETIRED,
        reason=reason,
        operator=operator,
        ts=ts,
        commit=commit,
    )


# -- TOML surgery ---------------------------------------------------------------


def _match_status_line(line: str) -> "tuple[str, str] | None":
    """Match one ``status = "..."`` line; return ``(indent, suffix)``.

    ``suffix`` is whatever follows the closing quote (whitespace and an
    optional trailing comment — preserved verbatim on rewrite). Returns
    ``None`` for every other line. Hand-parsed on purpose: the module
    stays inside the package's stdlib-basics import allowlist.
    """
    body = line.rstrip("\n")
    stripped = body.lstrip(" \t")
    indent = body[: len(body) - len(stripped)]
    key, sep, tail = stripped.partition("=")
    if not sep or key.strip() != "status":
        return None
    tail = tail.strip(" \t")
    if not tail.startswith('"'):
        return None
    closing = tail.find('"', 1)
    if closing <= 0:
        return None  # no closing quote: not a plain string value
    suffix = tail[closing + 1 :]
    cleaned = suffix.strip(" \t")
    if cleaned and not cleaned.startswith("#"):
        return None  # garbage after the value: leave the line alone
    return indent, suffix


def _rewrite_status(text: str, new_status: str) -> str:
    """Replace the ``status`` value inside the ``[experiment]`` table.

    Line-surgical on purpose: the registry is a git-versioned directory,
    so the rewrite must touch exactly one line and leave the rest of the
    document byte-identical (minimal diffs are the audit trail). Works on
    plain ``status = "..."`` lines — the only form the loader accepts —
    preserving indentation and any trailing comment.
    """
    lines = text.splitlines(keepends=True)
    section: int | None = None
    replaced = False
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            if section is not None:
                break  # left the [experiment] table without finding status
            if stripped == "[experiment]":
                section = index
            continue
        if section is None:
            continue
        match = _match_status_line(line)
        if match is None:
            continue
        indent, suffix = match
        newline = "\n" if line.endswith("\n") else ""
        lines[index] = f'{indent}status = "{new_status}"{suffix}{newline}'
        replaced = True
        break
    if not replaced:
        raise LifecycleError(
            "cannot rewrite experiment.status: no `status = \"...\"` line found "
            "inside the [experiment] table (the loader requires one; was the "
            "file edited by hand?)"
        )
    return "".join(lines)

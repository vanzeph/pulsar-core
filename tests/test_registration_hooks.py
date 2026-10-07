"""External registration hooks (自定义代码组装): the public surface the
``pulsar-app`` store assembles custom code against.

``register_factor`` existed from day one; this suite pins the two pieces
task STORE1 added to make the *external* registration contract complete:

* :func:`pulsar_core.register_model` — the modeler twin of
  ``register_factor`` (``MODEL_REGISTRY.register`` was reachable, but the
  factor surface had a named wrapper and the model surface did not);
* :meth:`Registry.discard` — the inverse of ``register`` so an assembly
  layer can *preview* a candidate module (execute it, diff the registry,
  undo exactly its additions) without leaving process state behind.
"""

from __future__ import annotations

import pytest

from pulsar_core import (
    FACTOR_REGISTRY,
    MODEL_REGISTRY,
    FactorDefinition,
    ModelDefinition,
    ModelScorer,
    PulsarCoreError,
    Registry,
    register_factor,
    register_model,
)


class _Scorer(ModelScorer):
    name = "hook_test_scorer"

    def score(self, section, history):  # pragma: no cover - never invoked
        return {}


def _factor(name: str) -> FactorDefinition:
    def compute(bars):  # pragma: no cover - never invoked
        return None

    return FactorDefinition(
        name=name, label=name, compute=compute, min_bars=1
    )


def _model(name: str) -> ModelDefinition:
    def create(params, factor_names):  # pragma: no cover - never invoked
        return _Scorer()

    return ModelDefinition(name=name, create=create, description="hook test")


@pytest.fixture(params=[FACTOR_REGISTRY, MODEL_REGISTRY])
def registry(request: pytest.FixtureRequest) -> Registry[object]:
    """Both research registries implement the same hook contract."""
    return request.param  # type: ignore[no-any-return]


def test_register_model_registers_into_model_registry() -> None:
    definition = _model("hook_register_ok")
    register_model(definition)
    try:
        assert MODEL_REGISTRY.resolve("hook_register_ok") is definition
        assert "hook_register_ok" in MODEL_REGISTRY.names()
    finally:
        MODEL_REGISTRY.discard("hook_register_ok")


def test_register_model_rejects_duplicates_loudly() -> None:
    register_model(_model("hook_dup"))
    try:
        with pytest.raises(PulsarCoreError, match="already registered"):
            register_model(_model("hook_dup"))
    finally:
        MODEL_REGISTRY.discard("hook_dup")


def test_discard_removes_exactly_one_registration(registry: Registry[object]) -> None:
    kept = _factor("hook_keep") if registry is FACTOR_REGISTRY else _model("hook_keep")  # type: ignore[arg-type]
    gone = _factor("hook_gone") if registry is FACTOR_REGISTRY else _model("hook_gone")  # type: ignore[arg-type]
    registry.register(kept)  # type: ignore[arg-type]
    registry.register(gone)  # type: ignore[arg-type]
    registry.discard("hook_gone")
    assert "hook_gone" not in registry
    assert registry.resolve("hook_keep") is kept
    registry.discard("hook_keep")  # leave the registry as we found it


def test_discard_fails_loudly_on_unknown_names(registry: Registry[object]) -> None:
    with pytest.raises(PulsarCoreError, match="unknown"):
        registry.discard("hook_never_registered")


def test_discard_removes_aliases_with_the_canonical_name() -> None:
    item = _model("hook_aliased")
    registry = Registry[object](kind="test", name_of=lambda thing: thing.name)  # type: ignore[attr-defined]
    registry.register(item, aliases=("hook_alias",))  # type: ignore[arg-type]
    registry.discard("hook_alias")
    assert "hook_aliased" not in registry
    assert "hook_alias" not in registry
    assert len(registry) == 0


def test_preview_pattern_snapshot_diff_discard_restores_state() -> None:
    """The exact pattern the pulsar-app store loader uses at put time."""
    before = set(MODEL_REGISTRY.names())
    register_model(_model("hook_preview"))
    added = set(MODEL_REGISTRY.names()) - before
    assert added == {"hook_preview"}
    for name in sorted(added):
        MODEL_REGISTRY.discard(name)
    assert set(MODEL_REGISTRY.names()) == before


def test_register_factor_still_round_trips() -> None:
    factor = _factor("hook_factor_ok")
    register_factor(factor)
    try:
        assert FACTOR_REGISTRY.resolve("hook_factor_ok") is factor
    finally:
        FACTOR_REGISTRY.discard("hook_factor_ok")

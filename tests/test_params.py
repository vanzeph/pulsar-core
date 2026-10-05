"""Params: declaration, binding, overrides, manifest export."""

from __future__ import annotations

import pytest

from pulsar_core import Params, PulsarCoreError


class TestDeclaration:
    def test_declares_ordered_defaults(self) -> None:
        params = Params(fast=5, slow=20, tag="dual")
        assert params.names() == ("fast", "slow", "tag")
        assert params.defaults() == {"fast": 5, "slow": 20, "tag": "dual"}
        assert len(params) == 3
        assert "fast" in params and "missing" not in params
        assert list(params) == ["fast", "slow", "tag"]

    def test_empty_declaration_is_valid(self) -> None:
        assert Params().names() == ()

    def test_non_scalar_default_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="bool/int/float/str"):
            Params(window=[5, 20])


class TestBinding:
    def test_bind_without_overrides_uses_defaults(self) -> None:
        bound = Params(fast=5, slow=20).bind()
        assert bound.fast == 5
        assert bound.slow == 20

    def test_bind_applies_overrides(self) -> None:
        bound = Params(fast=5, slow=20).bind({"fast": 3})
        assert bound.fast == 3
        assert bound.slow == 20

    def test_unknown_override_fails_loudly(self) -> None:
        with pytest.raises(PulsarCoreError, match="unknown parameter 'faster'"):
            Params(fast=5).bind({"faster": 3})

    @pytest.mark.parametrize(
        ("default", "override"),
        [
            (5, "5"),  # str for int
            (5, 5.5),  # lossy float for int
            ("dual", 7),  # int for str
            (5, True),  # bool for int
            (True, 1),  # int for bool
        ],
    )
    def test_type_mismatch_is_rejected(self, default: object, override: object) -> None:
        with pytest.raises(PulsarCoreError, match="expects"):
            Params(value=default).bind({"value": override})

    def test_int_widens_to_declared_float(self) -> None:
        bound = Params(threshold=1.0).bind({"threshold": 2})
        assert bound.threshold == 2.0
        assert isinstance(bound.threshold, float)

    def test_missing_name_raises_attribute_error(self) -> None:
        bound = Params(fast=5).bind()
        with pytest.raises(AttributeError, match="no such parameter"):
            _ = bound.missing

    def test_equality_and_export(self) -> None:
        base = Params(fast=5, slow=20)
        assert base.bind({"fast": 3}) == Params(fast=5, slow=20).bind({"fast": 3})
        assert base.bind() != base.bind({"fast": 3})
        exported = base.bind({"fast": 3}).to_dict()
        assert exported == {"fast": 3, "slow": 20}
        # JSON-native: directly serializable for the manifest
        import json

        assert json.loads(json.dumps(exported)) == exported

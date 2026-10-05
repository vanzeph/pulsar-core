"""Strategy parameter declaration and binding.

Design policy (core-engine design): parameters and strategy code are
separated. A strategy declares its parameters once as class-level metadata
(``params = Params(fast=5, slow=20)``); runs supply overrides through
configuration, and the effective parameter set flows into the run's
``RunManifest`` so every run is reproducible and parameter sweeps never
touch strategy code.

Supported value types are the JSON-native scalars (``bool``, ``int``,
``float``, ``str``) — enough for a manifest to archive them verbatim.
"""

from __future__ import annotations

from typing import Any, Iterator, Mapping, cast

from pydantic import Field

from pulsar_contracts import ContractModel

from .errors import PulsarCoreError

__all__ = ["Param", "Params", "BoundParams"]

#: Scalar types a parameter may carry, mapped to a checker per type.
_ALLOWED_TYPES: tuple[type, ...] = (bool, int, float, str)


class Param(ContractModel):
    """One declared parameter: name, default value, optional description."""

    name: str = Field(min_length=1)
    default: bool | int | float | str
    description: str = ""

    @property
    def value_type(self) -> type:
        """The declared type, taken from the default value."""
        return type(self.default)


class Params:
    """An ordered declaration of strategy parameters.

    Declared with keyword syntax so a strategy class reads like its own
    documentation::

        class DualMA(StrategyBase):
            params = Params(fast=5, slow=20)

    The declaration is immutable; :meth:`bind` produces the per-run
    effective set (defaults overridden by configuration).
    """

    def __init__(self, **defaults: Any) -> None:
        self._params: dict[str, Param] = {}
        for name, default in defaults.items():
            if not isinstance(default, _ALLOWED_TYPES):
                raise TypeError(
                    f"parameter {name!r} default must be bool/int/float/str, "
                    f"got {type(default).__name__}"
                )
            self._params[name] = Param(name=name, default=default)

    # -- declaration surface ------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return name in self._params

    def __iter__(self) -> Iterator[str]:
        return iter(self._params)

    def __len__(self) -> int:
        return len(self._params)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        body = ", ".join(
            f"{name}={param.default!r}" for name, param in self._params.items()
        )
        return f"Params({body})"

    def names(self) -> tuple[str, ...]:
        """Declared parameter names in declaration order."""
        return tuple(self._params)

    def defaults(self) -> dict[str, Any]:
        """The default values as a JSON-native dict (manifest-ready)."""
        return {name: param.default for name, param in self._params.items()}

    # -- binding ------------------------------------------------------------

    def bind(self, overrides: Mapping[str, Any] | None = None) -> "BoundParams":
        """Resolve the effective parameters for one run.

        ``overrides`` come from run configuration; unknown names and
        type-incompatible values fail loudly — a typo in a config must never
        silently fall back to a default. ``int`` values may widen to a
        declared ``float``; every other type mismatch is rejected.
        """
        effective: dict[str, bool | int | float | str] = dict(self.defaults())
        for name, value in (overrides or {}).items():
            param = self._params.get(name)
            if param is None:
                raise PulsarCoreError(
                    f"unknown parameter {name!r}; declared: {list(self._params)}"
                )
            effective[name] = _coerce(param, value)
        return BoundParams(effective)


def _coerce(param: Param, value: Any) -> bool | int | float | str:
    """Validate ``value`` against ``param``'s declared type."""
    declared = param.value_type
    if isinstance(value, bool):
        if declared is bool:
            return value
        raise PulsarCoreError(
            f"parameter {param.name!r} expects {declared.__name__}, got bool"
        )
    if declared is bool:
        raise PulsarCoreError(
            f"parameter {param.name!r} expects bool, got {type(value).__name__}"
        )
    if declared is float:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    elif isinstance(value, declared) and not isinstance(value, bool):
        # narrowed by the runtime isinstance check against ``declared``
        return cast("bool | int | float | str", value)
    raise PulsarCoreError(
        f"parameter {param.name!r} expects {declared.__name__}, "
        f"got {type(value).__name__}"
    )


class BoundParams:
    """The effective parameter set of one strategy instance.

    Attribute access reads the resolved values; the whole set exports to a
    JSON-native dict for the run manifest.
    """

    def __init__(self, values: Mapping[str, bool | int | float | str]) -> None:
        self._values: dict[str, bool | int | float | str] = dict(values)

    def __getattr__(self, name: str) -> Any:
        values = object.__getattribute__(self, "_values")
        if name.startswith("_"):
            raise AttributeError(name)
        if name not in values:
            raise AttributeError(f"no such parameter {name!r}")
        return values[name]

    def __getitem__(self, name: str) -> bool | int | float | str:
        return self._values[name]

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BoundParams):
            return self._values == other._values
        return NotImplemented

    def __hash__(self) -> int:  # pragma: no cover - completeness
        return hash(tuple(sorted(self._values.items(), key=lambda kv: kv[0])))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        body = ", ".join(f"{k}={v!r}" for k, v in self._values.items())
        return f"BoundParams({body})"

    def to_dict(self) -> dict[str, bool | int | float | str]:
        """JSON-native copy of the effective parameters (manifest-ready)."""
        return dict(self._values)

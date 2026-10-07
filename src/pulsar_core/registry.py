"""Generic name-keyed registries for the research layer.

The layered-configuration design (core-engine design, 因子库与实验配置) splits
the world in two: *code things* — factors, modelers, preprocess steps,
portfolio constructors, universes — are implemented once and registered
under a stable name; *experiments* are pure TOML that reference those
names. This module is the registration half of that contract: a
:class:`Registry` maps names (plus optional aliases) to instances, fails
loudly on unknown names and duplicate registrations, and iterates its
contents in a deterministic (sorted) order so manifests and error
messages never depend on registration order.
"""

from __future__ import annotations

from typing import Callable, Generic, Iterator, Sequence, TypeVar

from .errors import PulsarCoreError

__all__ = ["Registry"]

T = TypeVar("T")


class Registry(Generic[T]):
    """A name-keyed registry of research-layer building blocks.

    ``name_of`` extracts the canonical key of an item (usually a ``name``
    attribute). Aliases resolve to the same item — e.g. the IC-weighted
    modeler is reachable both as ``ic_weighted`` and by its design-doc
    name ``linear_ic`` — but :meth:`names` lists canonical names only.
    """

    def __init__(self, *, kind: str, name_of: Callable[[T], str]) -> None:
        self._kind = kind
        self._name_of = name_of
        self._items: dict[str, T] = {}
        self._aliases: dict[str, str] = {}

    # -- registration --------------------------------------------------------

    def register(
        self, item: T, *, aliases: Sequence[str] = (), replace: bool = False
    ) -> None:
        """Register ``item`` under its canonical name plus ``aliases``.

        Registration is idempotent only for the exact same item; clashing
        names raise unless ``replace`` is set (tests and assembly layers
        may deliberately override a built-in).
        """
        key = self._name_of(item)
        if not key:
            raise PulsarCoreError(f"{self._kind} name must be non-empty")
        names = [key, *aliases]
        for alias in names:
            existing = self._items.get(alias)
            if existing is not None and (existing is not item or not replace):
                if existing is item and replace:
                    continue
                raise PulsarCoreError(
                    f"{self._kind} name {alias!r} is already registered"
                )
        for alias in names:
            self._items[alias] = item
        for alias in aliases:
            self._aliases[alias] = key

    # -- removal ----------------------------------------------------------------

    def discard(self, name: str) -> None:
        """Remove the registration behind ``name`` (canonical or alias).

        The inverse of :meth:`register`, added for the assembly layer's
        *preview* passes (自定义代码组装): tooling may execute a candidate
        module to discover what it registers, then undo exactly those
        registrations so the process state is left as it was found.
        Removing one item never touches other items; unknown names fail
        loudly, mirroring :meth:`resolve`.
        """
        canonical = self.canonical_name(name)  # fails loudly when unknown
        del self._items[canonical]
        for alias, target in list(self._aliases.items()):
            if target == canonical:
                del self._aliases[alias]
                self._items.pop(alias, None)

    # -- lookup ---------------------------------------------------------------

    def resolve(self, name: str) -> T:
        """Return the item registered under ``name`` (canonical or alias)."""
        item = self._items.get(name)
        if item is None:
            raise PulsarCoreError(
                f"unknown {self._kind} {name!r}; registered: {self.names()}"
            )
        return item

    def canonical_name(self, name: str) -> str:
        """The canonical name behind ``name`` (resolving aliases)."""
        self.resolve(name)  # fail loudly on unknown names
        return self._aliases.get(name, name)

    def names(self) -> tuple[str, ...]:
        """Canonical names only, sorted for deterministic output."""
        canonical = {self._name_of(item) for item in set(self._items.values())}
        return tuple(sorted(canonical))

    def items(self) -> tuple[tuple[str, T], ...]:
        """Canonical (name, item) pairs sorted by name."""
        canonical = {self._name_of(item): item for item in self._items.values()}
        return tuple(sorted(canonical.items()))

    # -- container surface ------------------------------------------------------

    def __contains__(self, name: object) -> bool:
        return isinstance(name, str) and name in self._items

    def __len__(self) -> int:
        return len(self.names())

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

"""Portfolio construction: modeler scores to target weights.

The last research-side step of the factor pipeline (core-engine design,
组合构建: TopN · 加权) turns one decision date's scores into target
weights. Constructors are registered code — like factors and modelers —
and experiments select one by the ``[portfolio] method`` name.

Construction invariants:

* only scored symbols can receive weight (missing data never buys);
* weights are non-negative and sum to at most 1.0 of equity;
* tie-breaking and iteration are deterministic (score descending, then
  symbol ascending), so identical inputs construct identical targets.

The output feeds the C2 intent pipeline as *target weights* — the
strategy declares them, the engine diffs, risk-checks and emits intents
(:mod:`pulsar_core.pipeline`, :mod:`pulsar_core.runner`).
"""

from __future__ import annotations

from typing import Callable, Mapping

from .errors import PulsarCoreError
from .registry import Registry

__all__ = [
    "PortfolioConstructor",
    "TopNConstructor",
    "PORTFOLIO_REGISTRY",
]


class PortfolioConstructor:
    """Base class of one registered portfolio construction method."""

    name: str = "portfolio_method"

    def construct(self, scores: Mapping[str, float]) -> dict[str, float]:
        """Turn ``{symbol: score}`` into ``{symbol: target weight}``."""
        raise NotImplementedError


class TopNConstructor(PortfolioConstructor):
    """Equal weight over the ``top_n`` highest-scoring symbols.

    Fewer scored symbols than ``top_n`` spreads weight over what exists
    (a young or gappy universe is not an error).
    """

    name = "top_n"

    def __init__(self, top_n: int) -> None:
        if isinstance(top_n, bool) or not isinstance(top_n, int) or top_n < 1:
            raise PulsarCoreError(f"top_n must be a positive integer, got {top_n!r}")
        self.top_n = top_n

    def construct(self, scores: Mapping[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        selected = ranked[: self.top_n]
        weight = 1.0 / len(selected)
        return {symbol: weight for symbol, _score in selected}


class PortfolioMethodDefinition:
    """Registry entry: how to build one constructor from ``[portfolio]``."""

    def __init__(
        self,
        *,
        name: str,
        create: Callable[[Mapping[str, object]], PortfolioConstructor],
        description: str = "",
    ) -> None:
        self.name = name
        self.create = create
        self.description = description


def _create_top_n(params: Mapping[str, object]) -> PortfolioConstructor:
    top_n = params.get("top_n")
    if isinstance(top_n, bool) or not isinstance(top_n, int):
        raise PulsarCoreError(
            f"portfolio method 'top_n' requires integer top_n, got {top_n!r}"
        )
    return TopNConstructor(top_n)


#: The default registry experiments resolve ``[portfolio] method`` in.
PORTFOLIO_REGISTRY: Registry[PortfolioMethodDefinition] = Registry(
    kind="portfolio method", name_of=lambda method: method.name
)
PORTFOLIO_REGISTRY.register(
    PortfolioMethodDefinition(
        name="top_n",
        create=_create_top_n,
        description="equal weight over the top-N scored symbols",
    )
)

"""Signals and target portfolios: what strategies want, not how to get it.

Pipeline semantics (core-engine design):

1. ``Signal`` — the target tendency a strategy declares for one symbol:
   a weight of equity, an absolute share count, or a bare "size me"
   intention handed to the portfolio builder.
2. ``PortfolioBuilder`` — integrates the signals of one decision point
   into a ``TargetPortfolio``: equal weight, weighted, or any custom
   implementation.
3. Downstream (rebalance + risk + intent) turns the target portfolio into
   sized order intents; strategies never size or place orders themselves.

Semantics of a *target*: it is the desired end state for the symbols it
mentions. Symbols it does not mention are left alone this decision point —
a strategy exits a name by explicitly declaring a zero target.
"""

from __future__ import annotations

from typing import Sequence

from pydantic import Field, model_validator

from pulsar_contracts import ContractModel

from .errors import PulsarCoreError

__all__ = [
    "Signal",
    "TargetLeg",
    "TargetPortfolio",
    "PortfolioBuilder",
    "PassThroughBuilder",
    "EqualWeightBuilder",
]


class Signal(ContractModel):
    """One declared target tendency for one symbol.

    Exactly one sizing mode is active:

    * ``weight`` — fraction of current equity allocated to the symbol
      (``0.0`` is an explicit exit);
    * ``shares`` — absolute target share count (``0`` is an explicit exit);
    * both ``None`` — a bare "long interest" handed to the builder for
      sizing (equal weight and top-N builders size these).
    """

    symbol: str = Field(min_length=1)
    weight: float | None = Field(default=None, ge=0)
    shares: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_exactly_one_mode(self) -> "Signal":
        if self.weight is not None and self.shares is not None:
            raise ValueError(
                f"signal for {self.symbol!r} cannot carry both weight and shares"
            )
        return self

    @property
    def is_bare(self) -> bool:
        """Whether the signal carries no sizing and asks the builder to size it."""
        return self.weight is None and self.shares is None


class TargetLeg(ContractModel):
    """One resolved target: exactly one of weight or absolute shares."""

    symbol: str = Field(min_length=1)
    weight: float | None = Field(default=None, ge=0)
    shares: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def _validate_exactly_one_mode(self) -> "TargetLeg":
        if (self.weight is None) == (self.shares is None):
            raise ValueError(
                f"target leg for {self.symbol!r} must carry exactly one of "
                f"weight / shares"
            )
        return self


class TargetPortfolio(ContractModel):
    """The resolved target set of one decision point.

    The portfolio only constrains the symbols it mentions; the rebalance
    step computes differences against current holdings and in-flight
    orders for exactly those symbols.
    """

    targets: dict[str, TargetLeg] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_leg_symbols(self) -> "TargetPortfolio":
        for symbol, leg in self.targets.items():
            if leg.symbol != symbol:
                raise ValueError(
                    f"target map key {symbol!r} does not match leg symbol {leg.symbol!r}"
                )
        return self

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def symbols(self) -> tuple[str, ...]:
        """Mentioned symbols in sorted order (deterministic iteration)."""
        return tuple(sorted(self.targets))


class PortfolioBuilder:
    """Base class / protocol anchor for portfolio construction.

    A builder turns the signals declared at one decision point into a
    :class:`TargetPortfolio`. Implementations must be deterministic and
    must raise :class:`~pulsar_core.errors.PulsarCoreError` on inputs they
    cannot size — never silently drop or guess a signal.
    """

    def build(self, signals: Sequence[Signal]) -> TargetPortfolio:
        raise NotImplementedError


class PassThroughBuilder(PortfolioBuilder):
    """Uses declared sizings verbatim; refuses bare (unsized) signals."""

    def build(self, signals: Sequence[Signal]) -> TargetPortfolio:
        legs: dict[str, TargetLeg] = {}
        for signal in signals:
            if signal.is_bare:
                raise PulsarCoreError(
                    f"PassThroughBuilder cannot size bare signal for {signal.symbol!r}"
                )
            leg = (
                TargetLeg(symbol=signal.symbol, weight=signal.weight)
                if signal.weight is not None
                else TargetLeg(symbol=signal.symbol, shares=signal.shares)
            )
            existing = legs.get(signal.symbol)
            if existing is not None and existing != leg:
                raise PulsarCoreError(
                    f"conflicting targets declared for {signal.symbol!r}"
                )
            legs[signal.symbol] = leg
        return TargetPortfolio(targets=legs)


class EqualWeightBuilder(PortfolioBuilder):
    """Sizes every bare signal with an equal fraction of equity.

    Explicitly sized signals (weight / shares) pass through untouched, so a
    strategy can mix "equal weight my selection" with pinned names. Bare
    signals share ``1 / n`` of equity among themselves; whether the total
    exposure is acceptable is a risk-chain concern, not a builder one.
    """

    def build(self, signals: Sequence[Signal]) -> TargetPortfolio:
        bare = [signal for signal in signals if signal.is_bare]
        share = 1.0 / len(bare) if bare else 0.0
        legs: dict[str, TargetLeg] = {}
        for signal in signals:
            if signal.is_bare:
                leg = TargetLeg(symbol=signal.symbol, weight=share)
            elif signal.weight is not None:
                leg = TargetLeg(symbol=signal.symbol, weight=signal.weight)
            else:
                leg = TargetLeg(symbol=signal.symbol, shares=signal.shares)
            existing = legs.get(signal.symbol)
            if existing is not None and existing != leg:
                raise PulsarCoreError(
                    f"conflicting targets declared for {signal.symbol!r}"
                )
            legs[signal.symbol] = leg
        return TargetPortfolio(targets=legs)

"""Decision-side trading account: cash, positions and marked equity.

This is the book the intent pipeline reads when it sizes targets and when
risk rules evaluate exposure. It is updated exclusively from fill reports
streaming back through the kernel loop ("成交回报回流账务，驱动下一轮决策"),
plus the last seen prices of each bar/snapshot for marking to market.

Full performance accounting (equity curve, metrics, fee attribution) is a
later task; here we keep exactly what decisions need: cash, per-symbol
quantities with A-share T+1 availability, average costs and last prices.
"""

from __future__ import annotations

from datetime import date

from pydantic import Field

from pulsar_contracts import ContractModel, Fill, Side

__all__ = ["PositionView", "PortfolioView", "TradingAccount"]


class PositionView(ContractModel):
    """Immutable snapshot of one held position."""

    symbol: str = Field(min_length=1)
    quantity: int = Field(ge=0)
    available_quantity: int = Field(ge=0)
    avg_cost: float | None = Field(default=None, gt=0)


class PortfolioView(ContractModel):
    """Immutable account snapshot handed to strategies and risk rules."""

    cash: float
    equity: float
    positions: tuple[PositionView, ...] = ()  # sorted by symbol

    def position(self, symbol: str) -> PositionView | None:
        """The snapshot of ``symbol``, or ``None`` when nothing is held."""
        for view in self.positions:
            if view.symbol == symbol:
                return view
        return None


class _Position:
    """Mutable internal position record (never exposed to strategies)."""

    __slots__ = ("quantity", "available", "avg_cost")

    def __init__(self, quantity: int, available: int, avg_cost: float | None) -> None:
        self.quantity = quantity
        self.available = available
        self.avg_cost = avg_cost


class TradingAccount:
    """Cash + positions + last prices, updated from fills and bars.

    The account is deliberately *not* handed to strategies: contexts expose
    :class:`PortfolioView` snapshots only, so strategy code can read the
    book but never mutate it.
    """

    def __init__(self, *, cash: float) -> None:
        if cash <= 0:
            raise ValueError(f"initial cash must be positive, got {cash}")
        self._cash = float(cash)
        self._positions: dict[str, _Position] = {}
        self._last_prices: dict[str, float] = {}

    # -- reads --------------------------------------------------------------

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def prices(self) -> dict[str, float]:
        """A copy of the last seen price per symbol."""
        return dict(self._last_prices)

    def last_price(self, symbol: str) -> float | None:
        return self._last_prices.get(symbol)

    @property
    def equity(self) -> float:
        """Cash plus the market value of all positions at last prices."""
        value = self._cash
        for symbol, position in self._positions.items():
            price = self._last_prices.get(symbol)
            if price is not None:
                value += position.quantity * price
        return value

    def position(self, symbol: str) -> PositionView | None:
        record = self._positions.get(symbol)
        if record is None:
            return None
        return PositionView(
            symbol=symbol,
            quantity=record.quantity,
            available_quantity=record.available,
            avg_cost=record.avg_cost,
        )

    def positions(self) -> tuple[PositionView, ...]:
        """Position snapshots sorted by symbol (deterministic order)."""
        views: list[PositionView] = []
        for symbol in sorted(self._positions):
            view = self.position(symbol)
            if view is not None:  # held symbols always resolve
                views.append(view)
        return tuple(views)

    def snapshot(self) -> PortfolioView:
        """An immutable view of the whole account at last prices."""
        return PortfolioView(
            cash=self._cash, equity=self.equity, positions=self.positions()
        )

    # -- writes (kernel loop only) -------------------------------------------

    def mark_price(self, symbol: str, price: float) -> None:
        """Record the latest price of ``symbol`` (bar close / last price)."""
        if price <= 0:
            raise ValueError(f"price must be positive, got {price}")
        self._last_prices[symbol] = float(price)

    def on_new_trading_day(self, day: date) -> None:
        """Release T+1 availability: shares bought earlier become sellable.

        Called by the runtime when the loop first sees an event of a new
        trading day; shares bought during that same day stay locked until
        the next one, per A-share T+1 settlement.
        """
        del day  # the date itself is not needed; the parameter keeps intent explicit
        for record in self._positions.values():
            record.available = record.quantity

    def apply_fill(self, fill: Fill) -> None:
        """Book one confirmed fill: cash moves now, positions follow suit.

        Fees are charged in the direction the venue reports them:
        commission and transfer fee on both sides, stamp duty on sells.
        """
        symbol = fill.symbol
        record = self._positions.setdefault(symbol, _Position(0, 0, None))
        proceeds = fill.price * fill.quantity
        fees = fill.commission + fill.transfer_fee + fill.stamp_duty
        if fill.side is Side.BUY:
            total_cost = proceeds + fees
            new_quantity = record.quantity + fill.quantity
            if new_quantity > 0:
                record.avg_cost = (
                    record.quantity * (record.avg_cost or fill.price) + total_cost
                ) / new_quantity
            self._cash -= total_cost
            record.quantity = new_quantity
            # bought shares stay locked until the next trading day (T+1)
        else:
            self._cash += proceeds - fees
            record.quantity -= fill.quantity
            record.available = max(0, record.available - fill.quantity)
            if record.quantity <= 0:
                del self._positions[symbol]

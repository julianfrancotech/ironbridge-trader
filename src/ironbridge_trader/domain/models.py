"""Core domain objects.

Frozen vs. mutable is a deliberate choice per class, not a default --
the same split ironbridge-capital's own domain models use: Bar and Fill
are facts that already happened (immutable, hashable). Order
and Position are state that evolves over time (mutable, updated in place
as fills arrive).

This app trades once per *bar* (one row of OHLCV data, default daily),
not per tick -- see docs/platform-boundaries.md for why. A tick has no
open/high/low/close of its own to compute returns or indicators from; a
bar does.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from enum import Enum, auto


class Side(Enum):
    BUY = auto()
    SELL = auto()


class OrderStatus(Enum):
    NEW = auto()
    FILLED = auto()
    REJECTED = auto()


class Action(Enum):
    BUY = auto()
    SELL = auto()
    HOLD = auto()


@dataclass(frozen=True, slots=True)
class Bar:
    """One OHLCV bar for a symbol. Immutable -- a historical fact."""

    symbol: str
    timestamp: datetime
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int


@dataclass(frozen=True, slots=True)
class Fill:
    """A completed execution. Immutable.

    `price` is the actual execution price -- for PaperBroker, already
    including the spread/slippage haircut (see adapters/paper_broker.py);
    for a real broker, whatever it actually filled at. `commission` is
    kept separate, matching how a real brokerage statement shows them:
    one number for what you paid *for the shares*, another for the fee
    on top. Defaults to 0 so every existing caller/test that predates
    cost modeling keeps working unchanged.
    """

    order_id: str
    symbol: str
    side: Side
    price: Decimal
    quantity: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    commission: Decimal = Decimal(0)


@dataclass(slots=True)
class Order:
    """A paper order. Mutable -- status changes as the broker responds."""

    order_id: str
    symbol: str
    side: Side
    quantity: int
    # The bar this order was decided on: its close is what the paper
    # broker fills at, and its timestamp is what the Fill is stamped
    # with. Both live on the order (set once, at construction) rather
    # than as mutable state on the broker or "now" at fill time -- the
    # latter would be wrong during a backtest, where "now" is the wall
    # clock the backtest happened to run at, not the simulated date.
    reference_price: Decimal
    as_of: datetime
    status: OrderStatus = OrderStatus.NEW
    fill: Fill | None = None

    def apply_fill(self, fill: Fill) -> None:
        self.fill = fill
        self.status = OrderStatus.FILLED

    @classmethod
    def market_order(
        cls, order_id: str, symbol: str, side: Side, quantity: int,
        reference_price: Decimal, as_of: datetime,
    ) -> Order:
        return cls(
            order_id=order_id, symbol=symbol, side=side, quantity=quantity,
            reference_price=reference_price, as_of=as_of,
        )


@dataclass(slots=True)
class Position:
    """Running position in one symbol. Mutable -- updated on every fill."""

    symbol: str
    quantity: int = 0
    average_price: Decimal = Decimal(0)

    def apply_fill(self, fill: Fill) -> None:
        signed_qty = fill.quantity if fill.side is Side.BUY else -fill.quantity
        new_quantity = self.quantity + signed_qty

        if new_quantity == 0:
            self.average_price = Decimal(0)
        elif self.quantity == 0 or (self.quantity > 0) == (signed_qty > 0):
            total_cost = self.average_price * abs(self.quantity) + fill.price * fill.quantity
            self.average_price = total_cost / abs(new_quantity)
        self.quantity = new_quantity


@dataclass(frozen=True, slots=True)
class QuantSignal:
    """The predictive model's opinion for one symbol at one point in time.

    prob_up is a calibrated-ish probability in [0, 1] that the next bar's
    close will be higher than the current close. It is deliberately just
    a number + the feature snapshot that produced it -- no BUY/SELL verdict
    baked in. Turning a probability into an action is a policy decision,
    made downstream by the decision-making service, not by the model itself.
    """

    symbol: str
    as_of: datetime
    prob_up: float
    model_version: str
    features: dict[str, float]


@dataclass(frozen=True, slots=True)
class Decision:
    """The decision-maker's final, structured verdict for one symbol at one bar.

    This is the record that makes the system's "flow" inspectable: every
    trade (or non-trade) traces back to one of these, which itself
    carries the quant signal it was grounded in and a natural-language
    rationale. rationale is never parsed as data downstream -- only shown.
    """

    symbol: str
    as_of: datetime
    action: Action
    confidence: float
    rationale: str
    quant_signal: QuantSignal
    agent_kind: str  # "claude" or "quant_threshold" -- which decision-maker produced this

"""Pre-trade risk checks -- the deterministic gate every proposed order
passes through *after* a Decision is made and *before* it ever reaches
the broker, regardless of whether the Claude agent or the threshold
fallback produced that Decision. This is the one place in the system
that can veto a trade with a hard rule; nothing upstream of it (model,
agent) has the authority to skip it. Stateless: handed the current
position and equity on every call rather than keeping its own running
tally, so the caller stays the single source of truth for position state.

Position limit is a fraction of account equity (notional value), not
a fixed unit count. A fixed count doesn't scale with price, so it
silently stops meaning the same thing as a symbol's price moves --
confirmed the hard way: at this app's default risk_fraction/
stop_loss_fraction/account_equity, the position sizer's own math
wants more units than a `max_position_size=100`-style cap allowed for
a meaningful stretch of AAPL's price history, vetoing roughly 70% of
its directional decisions for a reason that had nothing to do with
the signal (see scripts/risk_veto_report.py). A cap expressed as a
fraction of equity, like risk_fraction and margin_rate already are,
doesn't have that problem -- it means the same thing at any price.
"""

from __future__ import annotations

from decimal import Decimal

from ironbridge_trader.domain.exceptions import (
    InsufficientMarginError,
    PositionLimitExceededError,
)
from ironbridge_trader.domain.models import Order, Position, Side


class RiskManager:
    def __init__(self, max_position_fraction: Decimal, margin_rate: Decimal) -> None:
        self._max_position_fraction = max_position_fraction
        self._margin_rate = margin_rate

    def check_order(
        self,
        order: Order,
        current_position: Position,
        available_equity: Decimal,
    ) -> None:
        signed_qty = order.quantity if order.side is Side.BUY else -order.quantity
        projected_quantity = current_position.quantity + signed_qty
        projected_notional = abs(projected_quantity) * order.reference_price
        max_notional = available_equity * self._max_position_fraction

        if projected_notional > max_notional:
            raise PositionLimitExceededError(
                symbol=order.symbol,
                projected_notional=projected_notional,
                max_notional=max_notional,
            )

        required_margin = order.reference_price * order.quantity * self._margin_rate
        if required_margin > available_equity:
            raise InsufficientMarginError(
                order_id=order.order_id, required=required_margin, available=available_equity
            )

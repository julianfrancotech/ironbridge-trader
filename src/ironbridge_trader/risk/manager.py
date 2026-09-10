"""Pre-trade risk checks -- the deterministic gate every proposed order
passes through *after* a Decision is made and *before* it ever reaches
the broker, regardless of whether the Claude agent or the threshold
fallback produced that Decision. This is the one place in the system
that can veto a trade with a hard rule; nothing upstream of it (model,
agent) has the authority to skip it. Stateless: handed the current
position and equity on every call rather than keeping its own running
tally, so the caller stays the single source of truth for position state.
"""

from __future__ import annotations

from decimal import Decimal

from ironbridge_trader.domain.exceptions import (
    InsufficientMarginError,
    PositionLimitExceededError,
)
from ironbridge_trader.domain.models import Order, Position, Side


class RiskManager:
    def __init__(self, max_position_size: int, margin_rate: Decimal) -> None:
        self._max_position_size = max_position_size
        self._margin_rate = margin_rate

    def check_order(
        self,
        order: Order,
        current_position: Position,
        available_equity: Decimal,
    ) -> None:
        signed_qty = order.quantity if order.side is Side.BUY else -order.quantity
        projected_quantity = current_position.quantity + signed_qty

        if abs(projected_quantity) > self._max_position_size:
            raise PositionLimitExceededError(
                symbol=order.symbol,
                projected_quantity=projected_quantity,
                max_quantity=self._max_position_size,
            )

        required_margin = order.reference_price * order.quantity * self._margin_rate
        if required_margin > available_equity:
            raise InsufficientMarginError(
                order_id=order.order_id, required=required_margin, available=available_equity
            )

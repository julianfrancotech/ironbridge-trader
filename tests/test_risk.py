from datetime import UTC, datetime
from decimal import Decimal

import pytest

from ironbridge_trader.domain.exceptions import (
    InsufficientMarginError,
    PositionLimitExceededError,
)
from ironbridge_trader.domain.models import Order, Position, Side
from ironbridge_trader.risk.manager import RiskManager


def make_order(side: Side, quantity: int, price: str) -> Order:
    return Order.market_order(
        order_id="o1", symbol="TEST", side=side, quantity=quantity,
        reference_price=Decimal(price), as_of=datetime.now(UTC),
    )


def test_check_order_passes_within_limits():
    # order notional = 10 * 50 = 500; cap = 1000 * 0.6 = 600 -- under the cap
    risk = RiskManager(max_position_fraction=Decimal("0.6"), margin_rate=Decimal("0.1"))
    order = make_order(Side.BUY, 10, "50")
    risk.check_order(order, Position(symbol="TEST"), available_equity=Decimal(1000))  # no raise


def test_check_order_blocks_position_limit_breach():
    # order notional = 10 * 50 = 500; cap = 1000 * 0.1 = 100 -- over the cap
    risk = RiskManager(max_position_fraction=Decimal("0.1"), margin_rate=Decimal("0.1"))
    order = make_order(Side.BUY, 10, "50")
    with pytest.raises(PositionLimitExceededError):
        risk.check_order(order, Position(symbol="TEST"), available_equity=Decimal(1000))


def test_check_order_blocks_insufficient_margin():
    # max_position_fraction set generously (200%) to isolate the margin
    # check specifically -- order notional 500 comfortably clears the
    # 800 position cap, but margin_rate=1.0 still requires the full 500
    # against only 400 available.
    risk = RiskManager(max_position_fraction=Decimal("2.0"), margin_rate=Decimal("1.0"))
    order = make_order(Side.BUY, 10, "50")
    with pytest.raises(InsufficientMarginError):
        risk.check_order(order, Position(symbol="TEST"), available_equity=Decimal(400))


def test_check_order_accounts_for_existing_position():
    # existing 10 + new 10 = 20 units * $50 = 1000 notional; cap = 1000 * 0.5 = 500
    risk = RiskManager(max_position_fraction=Decimal("0.5"), margin_rate=Decimal("0.1"))
    order = make_order(Side.BUY, 10, "50")
    existing = Position(symbol="TEST", quantity=10, average_price=Decimal(45))
    with pytest.raises(PositionLimitExceededError):
        risk.check_order(order, existing, available_equity=Decimal(1000))

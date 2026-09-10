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
    risk = RiskManager(max_position_size=100, margin_rate=Decimal("0.1"))
    order = make_order(Side.BUY, 10, "50")
    risk.check_order(order, Position(symbol="TEST"), available_equity=Decimal(1000))  # no raise


def test_check_order_blocks_position_limit_breach():
    risk = RiskManager(max_position_size=5, margin_rate=Decimal("0.1"))
    order = make_order(Side.BUY, 10, "50")
    with pytest.raises(PositionLimitExceededError):
        risk.check_order(order, Position(symbol="TEST"), available_equity=Decimal(1000000))


def test_check_order_blocks_insufficient_margin():
    risk = RiskManager(max_position_size=1000, margin_rate=Decimal("1.0"))
    order = make_order(Side.BUY, 10, "50")
    with pytest.raises(InsufficientMarginError):
        risk.check_order(order, Position(symbol="TEST"), available_equity=Decimal(10))


def test_check_order_accounts_for_existing_position():
    risk = RiskManager(max_position_size=15, margin_rate=Decimal("0.1"))
    order = make_order(Side.BUY, 10, "50")
    existing = Position(symbol="TEST", quantity=10, average_price=Decimal(45))
    with pytest.raises(PositionLimitExceededError):
        risk.check_order(order, existing, available_equity=Decimal(1000000))

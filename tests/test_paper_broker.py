from datetime import UTC, datetime
from decimal import Decimal

from ironbridge_trader.adapters.paper_broker import PaperBroker
from ironbridge_trader.domain.models import Order, Side


def _order(side: Side, reference_price: str = "100") -> Order:
    return Order.market_order(
        order_id="TEST-abc123", symbol="TEST", side=side, quantity=10,
        reference_price=Decimal(reference_price), as_of=datetime(2024, 6, 1, tzinfo=UTC),
    )


def test_buy_fills_higher_than_the_reference_price():
    broker = PaperBroker(transaction_cost_bps=Decimal(10), commission_per_trade=Decimal(0))

    fill = broker.place_order(_order(Side.BUY))

    assert fill.price == Decimal(100) * (Decimal(1) + Decimal(10) / Decimal(10_000))
    assert fill.price > Decimal(100)


def test_sell_fills_lower_than_the_reference_price():
    broker = PaperBroker(transaction_cost_bps=Decimal(10), commission_per_trade=Decimal(0))

    fill = broker.place_order(_order(Side.SELL))

    assert fill.price == Decimal(100) * (Decimal(1) - Decimal(10) / Decimal(10_000))
    assert fill.price < Decimal(100)


def test_zero_cost_bps_fills_at_the_exact_reference_price():
    broker = PaperBroker(transaction_cost_bps=Decimal(0), commission_per_trade=Decimal(0))

    fill = broker.place_order(_order(Side.BUY))

    assert fill.price == Decimal(100)


def test_commission_is_attached_to_every_fill_regardless_of_side():
    broker = PaperBroker(transaction_cost_bps=Decimal(0), commission_per_trade=Decimal("1.50"))

    buy_fill = broker.place_order(_order(Side.BUY))
    sell_fill = broker.place_order(_order(Side.SELL))

    assert buy_fill.commission == Decimal("1.50")
    assert sell_fill.commission == Decimal("1.50")


def test_higher_cost_bps_widens_the_haircut():
    broker = PaperBroker(transaction_cost_bps=Decimal(100), commission_per_trade=Decimal(0))  # 1%

    fill = broker.place_order(_order(Side.BUY))

    assert fill.price == Decimal(101)

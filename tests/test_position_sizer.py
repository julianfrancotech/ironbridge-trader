from decimal import Decimal

from ironbridge_trader.risk.position_sizer import PositionSizer


def test_size_scales_with_equity():
    sizer = PositionSizer(risk_fraction=Decimal("0.01"), stop_loss_fraction=Decimal("0.05"))
    # risk_amount = 100,000 * 0.01 = 1,000; risk_per_unit = 50 * 0.05 = 2.5 -> 400 units
    assert sizer.size(reference_price=Decimal(50), available_equity=Decimal(100_000)) == 400
    # double the equity, double the size
    assert sizer.size(reference_price=Decimal(50), available_equity=Decimal(200_000)) == 800


def test_size_shrinks_as_price_rises():
    sizer = PositionSizer(risk_fraction=Decimal("0.01"), stop_loss_fraction=Decimal("0.05"))
    cheap = sizer.size(reference_price=Decimal(10), available_equity=Decimal(100_000))
    expensive = sizer.size(reference_price=Decimal(1000), available_equity=Decimal(100_000))
    assert cheap > expensive > 0


def test_size_is_zero_when_one_unit_exceeds_the_risk_budget():
    # A conservative retail-scale risk budget genuinely cannot afford even
    # one whole unit of a very expensive asset -- this must return 0, not
    # round up to 1 and silently take on more risk than requested.
    sizer = PositionSizer(risk_fraction=Decimal("0.01"), stop_loss_fraction=Decimal("0.05"))
    size = sizer.size(reference_price=Decimal(110_000), available_equity=Decimal(100_000))
    assert size == 0


def test_size_is_zero_for_nonpositive_price():
    sizer = PositionSizer(risk_fraction=Decimal("0.01"), stop_loss_fraction=Decimal("0.05"))
    assert sizer.size(reference_price=Decimal(0), available_equity=Decimal(100_000)) == 0

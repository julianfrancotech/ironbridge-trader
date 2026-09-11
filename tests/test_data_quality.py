from datetime import UTC, datetime
from decimal import Decimal

import pytest

from ironbridge_trader.domain.exceptions import MarketDataError
from ironbridge_trader.domain.models import Bar
from ironbridge_trader.features.data_quality import validate_bars


def _bar(close: str, day: int = 1) -> Bar:
    price = Decimal(close)
    return Bar(
        symbol="TEST", timestamp=datetime(2024, 1, day, tzinfo=UTC),
        open=price, high=price, low=price, close=price, volume=1000,
    )


def test_accepts_a_normal_sequence_of_moves():
    bars = [_bar("100", 1), _bar("102", 2), _bar("99", 3)]
    validate_bars("TEST", bars, previous_close=None)  # must not raise


def test_accepts_moves_against_a_previous_close():
    bars = [_bar("105", 2)]
    validate_bars("TEST", bars, previous_close=Decimal(100))  # 5% -- fine


def test_rejects_an_implausible_jump_within_the_new_batch():
    bars = [_bar("100", 1), _bar("500", 2)]  # 5x in one bar
    with pytest.raises(MarketDataError):
        validate_bars("TEST", bars, previous_close=None)


def test_rejects_an_implausible_jump_against_the_previous_close():
    bars = [_bar("1000", 2)]  # previous close was 100 -- a 10x scaling-error shape
    with pytest.raises(MarketDataError):
        validate_bars("TEST", bars, previous_close=Decimal(100))


def test_a_zero_previous_close_does_not_crash_the_check():
    # Division-by-zero guard -- shouldn't happen in real data, but the
    # check must not itself blow up if it ever does.
    bars = [_bar("50", 2)]
    validate_bars("TEST", bars, previous_close=Decimal(0))

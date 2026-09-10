from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from ironbridge_trader.domain.models import Bar


def make_bars(symbol: str = "TEST", n: int = 120, start_price: float = 100.0, seed: int = 7) -> list[Bar]:
    rng = random.Random(seed)
    bars = []
    price = start_price
    start = datetime(2024, 1, 1, tzinfo=UTC)
    for i in range(n):
        price = max(1.0, price + rng.uniform(-2, 2))
        high = price + rng.uniform(0, 1)
        low = price - rng.uniform(0, 1)
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=start + timedelta(days=i),
                open=Decimal(str(round(price, 2))),
                high=Decimal(str(round(high, 2))),
                low=Decimal(str(round(low, 2))),
                close=Decimal(str(round(price, 2))),
                volume=rng.randint(1000, 5000),
            )
        )
    return bars


@pytest.fixture
def bars() -> list[Bar]:
    return make_bars()

"""Sanity-checks on freshly-fetched bars, before they're trusted enough
to store or decide on. This guards against a different failure mode
than resilience.py's retry_with_backoff: a dropped connection is
transient and worth retrying, but a decimal-point/scaling glitch or a
bad print from a data provider is wrong data returned successfully --
retrying it would just get the same wrong answer back. The only sound
response is to refuse to store it and surface the failure loudly.
"""

from __future__ import annotations

from decimal import Decimal
from itertools import pairwise

from ironbridge_trader.domain.exceptions import MarketDataError
from ironbridge_trader.domain.models import Bar

# A close moving further than this from the prior close (including the
# boundary against the last bar already stored) is treated as
# implausible data rather than a real market move. Generous on purpose:
# real single-day moves on this app's default watchlist (large-cap
# equities, one major crypto pair) have never come close to this, even
# around earnings or major news -- the point is to catch a scaling
# error or a bad print, not to second-guess genuine volatility.
MAX_DAILY_MOVE_FRACTION = Decimal("0.5")


def validate_bars(symbol: str, bars: list[Bar], previous_close: Decimal | None) -> None:
    """Raises MarketDataError on the first close-to-close move (bars
    against each other, and the first bar against `previous_close` when
    given) that exceeds MAX_DAILY_MOVE_FRACTION. Doesn't modify `bars`
    or decide what happens next -- that's the caller's call (here:
    don't store them).
    """
    closes = ([previous_close] if previous_close is not None else []) + [b.close for b in bars]
    for prior, current in pairwise(closes):
        if prior == 0:
            continue
        if abs(current / prior - 1) > MAX_DAILY_MOVE_FRACTION:
            raise MarketDataError(
                f"{symbol}: implausible move from {prior} to {current} "
                f"(> {float(MAX_DAILY_MOVE_FRACTION):.0%} in one bar) -- looks like bad data, not stored"
            )

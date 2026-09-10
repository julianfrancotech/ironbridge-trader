"""Market data adapter: Alpaca as an alternative to yfinance
(adapters/market_data.py). Same fetch_bars(symbol, years, interval, start)
contract -- same MarketDataError on no data, same "empty result is fine
when start was given" incremental-fetch behavior -- so
orchestration/fetch_market_data.py can use either provider without
knowing which one it got. See Settings.data_provider.

Stock and crypto need two different Alpaca clients and two different
symbol formats: this app's watchlist follows yfinance's convention
("BTC-USD"), Alpaca's crypto pairs use a slash ("BTC/USD"). This module
routes on symbol shape (a "-" means crypto) rather than exposing that
split to callers -- the rest of the app just sees a list[Bar], same as
the yfinance adapter.

Only daily bars are supported, matching this app's one baked-in
bar_interval assumption (see config.py, docs/platform-boundaries.md).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import requests
from alpaca.common.exceptions import APIError
from alpaca.data.enums import Adjustment
from alpaca.data.historical.crypto import CryptoHistoricalDataClient
from alpaca.data.historical.stock import StockHistoricalDataClient
from alpaca.data.requests import CryptoBarsRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame

from ironbridge_trader.domain.exceptions import MarketDataError
from ironbridge_trader.domain.models import Bar
from ironbridge_trader.resilience import retry_with_backoff

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


class _TransientAlpacaError(Exception):
    """Marker used only to route an Alpaca APIError through retry_with_backoff --
    never raised or caught outside this module.
    """


def _reraise_for_retry(exc: APIError) -> None:
    # Alpaca's own rate-limit/server errors (429, 5xx) are exactly as
    # transient as a dropped connection and worth the same backoff.
    # Anything else (a bad request, an unknown symbol, bad credentials)
    # is permanent -- retrying it would just fail again identically.
    if exc.status_code in _TRANSIENT_STATUS_CODES:
        raise _TransientAlpacaError(str(exc)) from exc
    raise exc


def _is_crypto(symbol: str) -> bool:
    return "-" in symbol  # this app's watchlist convention, e.g. "BTC-USD"


def _alpaca_crypto_symbol(symbol: str) -> str:
    return symbol.replace("-", "/")  # Alpaca's crypto pair format, e.g. "BTC/USD"


_TRANSIENT_ERRORS = (
    _TransientAlpacaError, requests.exceptions.RequestException, ConnectionError, TimeoutError,
)


@retry_with_backoff(max_attempts=3, backoff_seconds=1.0, retry_on=_TRANSIENT_ERRORS)
def _download_bars(
    symbol: str, years: int, start: datetime | None, api_key: str, secret_key: str
) -> list:
    window_start = start if start is not None else datetime.now(UTC) - timedelta(days=365 * years)
    try:
        if _is_crypto(symbol):
            alpaca_symbol = _alpaca_crypto_symbol(symbol)
            client = CryptoHistoricalDataClient(api_key, secret_key)
            request = CryptoBarsRequest(
                symbol_or_symbols=alpaca_symbol, timeframe=TimeFrame.Day, start=window_start,
            )
            barset = client.get_crypto_bars(request)
        else:
            client = StockHistoricalDataClient(api_key, secret_key)
            request = StockBarsRequest(
                symbol_or_symbols=symbol, timeframe=TimeFrame.Day, start=window_start,
                adjustment=Adjustment.ALL,  # matches yfinance's auto_adjust=True
            )
            barset = client.get_stock_bars(request)
            alpaca_symbol = symbol
    except APIError as exc:
        _reraise_for_retry(exc)
        raise  # unreachable -- _reraise_for_retry always raises; satisfies type checkers

    return barset.data.get(alpaca_symbol, [])


def fetch_bars(
    symbol: str, years: int, interval: str = "1d", start: datetime | None = None,
    *, api_key: str, secret_key: str,
) -> list[Bar]:
    if interval != "1d":
        raise MarketDataError(f"alpaca adapter only supports daily bars, got interval={interval!r}")

    raw_bars = _download_bars(symbol, years, start, api_key, secret_key)
    if not raw_bars:
        if start is not None:
            # Incremental fetch found nothing newer than `start` -- e.g.
            # re-running before a new bar exists yet. Not an error, same
            # contract as adapters/market_data.py::fetch_bars.
            return []
        raise MarketDataError(f"Alpaca returned no data for {symbol}")

    return [
        Bar(
            symbol=symbol,
            timestamp=b.timestamp.astimezone(UTC),
            open=Decimal(str(b.open)), high=Decimal(str(b.high)),
            low=Decimal(str(b.low)), close=Decimal(str(b.close)),
            volume=int(b.volume),
        )
        for b in raw_bars
    ]

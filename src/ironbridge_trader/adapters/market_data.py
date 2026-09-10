"""Market data adapter: the boundary to the outside price source (Yahoo
Finance, free, no API key). This plays the same role a dedicated
exchange-connectivity adapter would -- isolating the third-party
integration behind one module -- but without an explicit connection
state machine. A CQG-style adapter needs one because that kind of
connection is a persistent, singleton websocket session
(DISCONNECTED -> CONNECTED -> LOGGED_ON -> ...); this adapter makes
stateless, independent HTTP requests, one per ingestion run, with
nothing to track between them. Modeling state that doesn't exist would
be the overengineering the project brief warns against.

Ingestion only: this module never talks to the orchestration layer
directly. scripts/fetch_data.py calls it and writes the result into
storage/db.py, and everything downstream (features, model, engine,
services, dashboard) reads bars back out of that database -- so the
rest of the app doesn't care whether bars came from Yahoo Finance, a
CSV, or a real broker feed tomorrow. `Database` itself already exposes
`get_bars(symbol, lookback)` matching protocols.MarketDataSource, so it
is used directly as the market-data adapter at read time.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pandas as pd
import requests
import yfinance as yf

from ironbridge_trader.domain.exceptions import MarketDataError
from ironbridge_trader.domain.models import Bar
from ironbridge_trader.resilience import retry_with_backoff

# Only network-transient failures are retried -- a symbol yfinance has
# genuinely never heard of should fail once, immediately, not stall the
# whole ingestion run for three backed-off attempts before failing anyway.
_TRANSIENT_ERRORS = (requests.exceptions.RequestException, ConnectionError, TimeoutError)


@retry_with_backoff(max_attempts=3, backoff_seconds=1.0, retry_on=_TRANSIENT_ERRORS)
def _download_history(symbol: str, years: int, interval: str, start: datetime | None) -> pd.DataFrame:
    ticker = yf.Ticker(symbol)
    if start is not None:
        # Incremental fetch: ask only for bars since the last one already
        # stored, instead of redownloading the full `years` window every
        # run. Re-requesting `start`'s own date is deliberate (not
        # start + 1 day) -- upsert_bars is an idempotent INSERT OR
        # REPLACE, so the only risk of an off-by-one here is skipping a
        # day, never duplicating one.
        return ticker.history(start=start.date(), interval=interval, auto_adjust=True)
    return ticker.history(period=f"{years}y", interval=interval, auto_adjust=True)


def fetch_bars(symbol: str, years: int, interval: str = "1d", start: datetime | None = None) -> list[Bar]:
    df = _download_history(symbol, years, interval, start)
    if df.empty:
        if start is not None:
            # Incremental fetch found nothing newer than `start` -- e.g.
            # re-running before a new bar exists yet. Not an error.
            return []
        raise MarketDataError(f"yfinance returned no data for {symbol}")

    bars: list[Bar] = []
    for ts, row in df.iterrows():
        bars.append(
            Bar(
                symbol=symbol,
                timestamp=ts.to_pydatetime().astimezone(UTC),
                open=Decimal(str(round(row["Open"], 4))),
                high=Decimal(str(round(row["High"], 4))),
                low=Decimal(str(round(row["Low"], 4))),
                close=Decimal(str(round(row["Close"], 4))),
                volume=int(row["Volume"]),
            )
        )
    return bars

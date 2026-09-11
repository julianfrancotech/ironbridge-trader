"""Workflow: pull fresh bars for the whole watchlist and persist them.

This is the orchestration layer's job in this app -- the same role a
production service's orchestration workflow files play: wire the
concrete adapters/services for one business workflow and run it, so
scripts/*.py (the CLI entrypoint) and, if this app ever grows one, a
scheduler or API layer can both call the same reusable function
instead of duplicating the wiring. build_fetch_bars is this workflow's
version of run_decision_cycle.py::build_decision_maker: pick the
concrete adapter, with the same log-and-fall-back-to-the-free-path
behavior when Alpaca is selected but its credentials aren't set.

Symbols are fetched with bounded concurrency (concurrency.run_bounded,
the same helper engine/trading_engine.py uses) -- one REST call per
symbol is the same shape as the decision cycle's per-symbol Claude
calls, and benefits from the same fix. Each call also asks for bars
only since that symbol's last stored one (db.latest_bar) rather than
redownloading the full history_years window every run -- see
adapters/market_data.py::fetch_bars. Every freshly-fetched batch is
sanity-checked (features/data_quality.py) before it's stored -- a bad
print or a scaling glitch from the data provider is wrong data
returned *successfully*, so nothing upstream would otherwise catch it.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date

from ironbridge_trader.adapters import alpaca_market_data, market_data
from ironbridge_trader.concurrency import run_bounded
from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Bar
from ironbridge_trader.features.data_quality import validate_bars
from ironbridge_trader.storage.db import Database

logger = logging.getLogger(__name__)

FetchBarsFn = Callable[..., list[Bar]]


@dataclass(frozen=True, slots=True)
class FetchResult:
    symbol: str
    bar_count: int
    first_date: date
    last_date: date


def build_fetch_bars(settings: Settings) -> FetchBarsFn:
    if settings.data_provider == "alpaca":
        if settings.alpaca_api_key and settings.alpaca_secret_key:
            logger.info("using Alpaca for market data (data_provider=alpaca)")
            return functools.partial(
                alpaca_market_data.fetch_bars,
                api_key=settings.alpaca_api_key, secret_key=settings.alpaca_secret_key,
            )
        logger.warning(
            "data_provider=alpaca but ALPACA_API_KEY/ALPACA_SECRET_KEY aren't set in .env -- "
            "falling back to yfinance"
        )
    return market_data.fetch_bars


def run(db: Database, settings: Settings) -> list[FetchResult]:
    fetch_bars = build_fetch_bars(settings)

    def fetch_one(symbol: str) -> FetchResult:
        previous_bar = db.latest_bar(symbol)
        last_ts = previous_bar.timestamp if previous_bar is not None else None
        bars = fetch_bars(
            symbol, years=settings.history_years, interval=settings.bar_interval, start=last_ts
        )
        if not bars:
            # Nothing newer than what's already stored (e.g. re-run
            # before a new bar exists) -- nothing to upsert.
            assert last_ts is not None  # only possible when start was passed, i.e. last_ts is set
            return FetchResult(symbol=symbol, bar_count=0, first_date=last_ts.date(), last_date=last_ts.date())

        previous_close = previous_bar.close if previous_bar is not None else None
        validate_bars(symbol, bars, previous_close)
        db.upsert_bars(bars)
        return FetchResult(
            symbol=symbol, bar_count=len(bars),
            first_date=bars[0].timestamp.date(), last_date=bars[-1].timestamp.date(),
        )

    futures = run_bounded(settings.symbols, fetch_one, settings.max_concurrent_symbols)
    return [future.result() for future in futures]

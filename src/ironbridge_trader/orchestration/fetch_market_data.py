"""Workflow: pull fresh bars for the whole watchlist and persist them.

This is the orchestration layer's job in this app -- the same role a
production service's orchestration workflow files play: wire the
concrete adapters/services for one business workflow and run it, so
scripts/*.py (the CLI entrypoint) and, if this app ever grows one, a
scheduler or API layer can both call the same reusable function
instead of duplicating the wiring.

Symbols are fetched with bounded concurrency (concurrency.run_bounded,
the same helper engine/trading_engine.py uses) -- one REST call per
symbol is the same shape as the decision cycle's per-symbol Claude
calls, and benefits from the same fix. Each call also asks for bars
only since that symbol's last stored one (db.latest_bar_ts) rather
than redownloading the full history_years window every run -- see
adapters/market_data.py::fetch_bars.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ironbridge_trader.adapters.market_data import fetch_bars
from ironbridge_trader.concurrency import run_bounded
from ironbridge_trader.config import Settings
from ironbridge_trader.storage.db import Database


@dataclass(frozen=True, slots=True)
class FetchResult:
    symbol: str
    bar_count: int
    first_date: date
    last_date: date


def run(db: Database, settings: Settings) -> list[FetchResult]:
    def fetch_one(symbol: str) -> FetchResult:
        last_ts = db.latest_bar_ts(symbol)
        bars = fetch_bars(
            symbol, years=settings.history_years, interval=settings.bar_interval, start=last_ts
        )
        if not bars:
            # Nothing newer than what's already stored (e.g. re-run
            # before a new bar exists) -- nothing to upsert.
            assert last_ts is not None  # only possible when start was passed, i.e. last_ts is set
            return FetchResult(symbol=symbol, bar_count=0, first_date=last_ts.date(), last_date=last_ts.date())

        db.upsert_bars(bars)
        return FetchResult(
            symbol=symbol, bar_count=len(bars),
            first_date=bars[0].timestamp.date(), last_date=bars[-1].timestamp.date(),
        )

    futures = run_bounded(settings.symbols, fetch_one, settings.max_concurrent_symbols)
    return [future.result() for future in futures]

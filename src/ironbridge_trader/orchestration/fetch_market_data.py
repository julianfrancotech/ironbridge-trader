"""Workflow: pull fresh bars for the whole watchlist and persist them.

This is the orchestration layer's job in this app -- the same role a
production service's orchestration workflow files play: wire the
concrete adapters/services for one business workflow and run it, so
scripts/*.py (the CLI entrypoint) and, if this app ever grows one, a
scheduler or API layer can both call the same reusable function
instead of duplicating the wiring.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from ironbridge_trader.adapters.market_data import fetch_bars
from ironbridge_trader.config import Settings
from ironbridge_trader.storage.db import Database


@dataclass(frozen=True, slots=True)
class FetchResult:
    symbol: str
    bar_count: int
    first_date: date
    last_date: date


def run(db: Database, settings: Settings) -> list[FetchResult]:
    results = []
    for symbol in settings.symbols:
        bars = fetch_bars(symbol, years=settings.history_years, interval=settings.bar_interval)
        db.upsert_bars(bars)
        results.append(
            FetchResult(
                symbol=symbol, bar_count=len(bars),
                first_date=bars[0].timestamp.date(), last_date=bars[-1].timestamp.date(),
            )
        )
    return results

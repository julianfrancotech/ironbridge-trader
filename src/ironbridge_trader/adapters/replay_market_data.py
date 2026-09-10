"""Market data adapter for backtesting only. Wraps a symbol's full stored
history but only reveals bars up to a moving cursor, so a simulated day
can never see a bar that "hasn't happened yet." That's the single most
important thing to get right in a backtest -- leaking future bars into
a decision about the past produces impossibly good, meaningless results.
Satisfies the same protocols.MarketDataSource as Database, so
scripts/backtest.py can hand this straight to the engine unmodified.
"""

from __future__ import annotations

from ironbridge_trader.domain.models import Bar
from ironbridge_trader.storage.db import Database


class ReplayMarketData:
    def __init__(self, source_db: Database, symbols: list[str], start_index: int) -> None:
        self._all_bars: dict[str, list[Bar]] = {s: source_db.get_bars(s) for s in symbols}
        self._cursor: dict[str, int] = {s: min(start_index, len(self._all_bars[s]) - 1) for s in symbols}

    def get_bars(self, symbol: str, lookback: int) -> list[Bar]:
        visible = self._all_bars[symbol][: self._cursor[symbol] + 1]
        return visible[-lookback:] if lookback else visible

    def advance(self, symbol: str) -> bool:
        """Move symbol's cursor forward one bar. False when history is exhausted."""
        if self._cursor[symbol] + 1 >= len(self._all_bars[symbol]):
            return False
        self._cursor[symbol] += 1
        return True

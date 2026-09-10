"""CLI entrypoint for orchestration.run_backtest. See that module's
docstring for the two honesty caveats about what this backtest can and
can't tell you.

    python scripts/backtest.py
"""

from __future__ import annotations

import logging

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import run_backtest
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db = Database(SETTINGS.db_path)
    result = run_backtest.run(db, SETTINGS)
    logger.info(
        "backtest complete: %d simulated fills across %d symbols -> %s",
        result.total_fills, result.symbols_replayed, result.backtest_db_path,
    )
    logger.info("open the dashboard's Backtest tab to see the equity curve and decision log")


if __name__ == "__main__":
    main()

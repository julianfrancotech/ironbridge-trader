"""CLI entrypoint for orchestration.fetch_market_data. Run first, and
re-run periodically to pull fresh bars before retraining or trading.

    python scripts/fetch_data.py
"""

from __future__ import annotations

import logging

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import fetch_market_data
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db = Database(SETTINGS.db_path)
    for result in fetch_market_data.run(db, SETTINGS):
        logger.info(
            "stored %d bars for %s (%s -> %s)",
            result.bar_count, result.symbol, result.first_date, result.last_date,
        )


if __name__ == "__main__":
    main()

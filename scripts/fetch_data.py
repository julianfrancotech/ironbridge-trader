"""CLI entrypoint for orchestration.fetch_market_data. Run first, and
re-run periodically to pull fresh bars before retraining or trading.

    python scripts/fetch_data.py

Records a heartbeat (storage/db.py's run_heartbeats table) on every
invocation, success or failure -- the dashboard's "Data fetch" status
reads this, same reasoning as run_paper_trader.py's heartbeat.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import fetch_market_data
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db = Database(SETTINGS.db_path)
    try:
        results = fetch_market_data.run(db, SETTINGS)
    except Exception as exc:
        db.record_heartbeat("fetch_data", datetime.now(UTC), "error", str(exc))
        raise

    db.record_heartbeat("fetch_data", datetime.now(UTC), "ok", f"{len(results)} symbol(s)")
    for result in results:
        logger.info(
            "stored %d bars for %s (%s -> %s)",
            result.bar_count, result.symbol, result.first_date, result.last_date,
        )


if __name__ == "__main__":
    main()

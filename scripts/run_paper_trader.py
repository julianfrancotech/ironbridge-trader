"""CLI entrypoint for orchestration.run_decision_cycle: one live
paper-trading cycle across the watchlist. No real money or brokerage is
involved -- see adapters/paper_broker.py. Meant to run once per bar
(once a day, for the default daily-bar config) via cron or by hand.

    python scripts/run_paper_trader.py
"""

from __future__ import annotations

import logging

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import run_decision_cycle
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db = Database(SETTINGS.db_path)
    decisions = run_decision_cycle.run(db, SETTINGS)
    for d in decisions:
        logger.info(
            "%s -> %s (confidence %.2f, via %s): %s",
            d.symbol, d.action.name, d.confidence, d.agent_kind, d.rationale,
        )


if __name__ == "__main__":
    main()

"""CLI entrypoint for orchestration.run_decision_cycle: one live
paper-trading cycle across the watchlist. No real money or brokerage is
involved -- see adapters/paper_broker.py. Meant to run once per bar
(once a day, for the default daily-bar config) via cron or by hand.

    python scripts/run_paper_trader.py

Records a heartbeat (storage/db.py's run_heartbeats table) on every
invocation, success or failure -- the dashboard's "Decision cycle"
status reads this. Without it, a cron job that silently stops running
(a closed laptop, a crashed process) would be invisible: nothing else
in this app checks whether today's cycle actually happened.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import run_decision_cycle
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db = Database(SETTINGS.db_path)
    try:
        decisions = run_decision_cycle.run(db, SETTINGS)
    except Exception as exc:
        db.record_heartbeat("run_paper_trader", datetime.now(UTC), "error", str(exc))
        raise

    db.record_heartbeat("run_paper_trader", datetime.now(UTC), "ok", f"{len(decisions)} new decision(s)")
    for d in decisions:
        logger.info(
            "%s -> %s (confidence %.2f, via %s): %s",
            d.symbol, d.action.name, d.confidence, d.agent_kind, d.rationale,
        )


if __name__ == "__main__":
    main()

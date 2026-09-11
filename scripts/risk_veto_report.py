"""CLI entrypoint for analytics.risk_veto: how often does
risk/manager.py veto a trade the decision-maker wanted, and why? Built
to answer a question raised by scripts/multi_regime_backtest.py's
results -- whether max_position_size, then a fixed unit cap, was
binding often enough against a dollar-risk-based position sizer to be
contaminating those results independent of whether the underlying
signal has real edge (it was -- see risk/manager.py's
max_position_fraction). Worth re-running after any change to
risk/position-sizing settings, not just that one past incident.

    python scripts/risk_veto_report.py
"""

from __future__ import annotations

import logging

from ironbridge_trader.analytics.risk_veto import capture_risk_vetoes
from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import run_backtest
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    source_db = Database(SETTINGS.db_path)
    with capture_risk_vetoes() as vetoes:
        result = run_backtest.run(source_db, SETTINGS)

    backtest_db = Database(result.backtest_db_path)
    directional = [d for d in backtest_db.all_decisions() if d["action"] != "HOLD"]

    logger.info("=== Risk veto report ===")
    logger.info(
        "%d directional decisions (BUY/SELL), %d filled, %d vetoed by risk/manager.py",
        len(directional), result.total_fills, vetoes.total,
    )
    if directional:
        logger.info("veto rate: %.1f%% of directional decisions", 100 * vetoes.total / len(directional))

    logger.info("--- by reason ---")
    for reason, count in vetoes.by_reason().most_common():
        logger.info("  %s: %d", reason, count)
    logger.info("--- by symbol ---")
    for symbol, count in vetoes.by_symbol().most_common():
        logger.info("  %s: %d", symbol, count)


if __name__ == "__main__":
    main()

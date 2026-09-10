"""CLI entrypoint for orchestration.retrain_model. Re-run any time after
scripts/fetch_data.py pulls fresh bars.

    python scripts/train_model.py
"""

from __future__ import annotations

import logging

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import retrain_model
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    db = Database(SETTINGS.db_path)
    result = retrain_model.run(db, SETTINGS)
    logger.info(
        "trained %s on %d rows, held out %d rows -> accuracy=%.3f auc=%.3f",
        result.predictor.version, result.train_rows, result.eval_rows,
        result.eval_accuracy, result.eval_auc,
    )
    logger.info(
        "eval rows are chronologically AFTER train rows (walk-forward split) -- "
        "these numbers are the honest, no-lookahead estimate of accuracy."
    )


if __name__ == "__main__":
    main()

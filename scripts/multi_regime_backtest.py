"""CLI entrypoint for analytics.multi_regime: run the full backtest
(orchestration.run_backtest, same as scripts/backtest.py) and then
slice its equity curve into calendar-length windows, reporting each
one's own return/Sharpe/drawdown against the buy-and-hold benchmark
and analytics.validation_bar's thresholds. See
docs/validation-plan.md for why this exists and what it can't tell
you (no permutation test here -- see that doc's item D).

    python scripts/multi_regime_backtest.py
"""

from __future__ import annotations

import logging

from ironbridge_trader.analytics.multi_regime import evaluate_regime_windows
from ironbridge_trader.config import SETTINGS
from ironbridge_trader.orchestration import run_backtest
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

NUM_WINDOWS = 4  # a diagnostic parameter, not a policy dial -- edit freely, see analytics/multi_regime.py


def main() -> None:
    source_db = Database(SETTINGS.db_path)
    result = run_backtest.run(source_db, SETTINGS)
    logger.info(
        "backtest complete: %d simulated fills across %d symbols -> %s",
        result.total_fills, result.symbols_replayed, result.backtest_db_path,
    )

    backtest_db = Database(result.backtest_db_path)
    windows = evaluate_regime_windows(
        backtest_db, list(SETTINGS.symbols), SETTINGS.account_equity, num_windows=NUM_WINDOWS
    )
    if not windows:
        logger.warning("not enough history to split into %d windows", NUM_WINDOWS)
        return

    logger.info("=== Multi-regime backtest report (%d windows) ===", len(windows))
    all_pass = True
    for w in windows:
        verdict = "PASS" if (w.beats_return and w.beats_sharpe and w.within_drawdown_budget) else "FAIL"
        all_pass = all_pass and verdict == "PASS"
        logger.info(
            "Window %d [%s -> %s]: %s",
            w.window_index + 1, w.start.date(), w.end.date(), verdict,
        )
        logger.info(
            "  return:   strategy %+.1f%%  vs benchmark %+.1f%%  %s",
            w.strategy_return * 100, w.benchmark_return * 100,
            "OK" if w.beats_return else "FAIL",
        )
        logger.info(
            "  sharpe:   strategy %s  vs benchmark %s  %s",
            f"{w.strategy_sharpe:.2f}" if w.strategy_sharpe is not None else "n/a",
            f"{w.benchmark_sharpe:.2f}" if w.benchmark_sharpe is not None else "n/a",
            "OK" if w.beats_sharpe else "FAIL",
        )
        logger.info(
            "  max dd:   strategy %.1f%%  vs benchmark %.1f%%  %s",
            w.strategy_max_drawdown * 100, w.benchmark_max_drawdown * 100,
            "OK" if w.within_drawdown_budget else "FAIL",
        )

    logger.info(
        "=== %s: %s ===",
        "ALL WINDOWS PASS" if all_pass else "AT LEAST ONE WINDOW FAILS",
        "still not the sealed evaluation in docs/validation-plan.md -- this is a development-time "
        "diagnostic, not a go/no-go decision by itself",
    )


if __name__ == "__main__":
    main()

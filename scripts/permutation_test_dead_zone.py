"""CLI entrypoint for analytics.permutation_test: is the AUC gap found
by scripts/compare_label_dead_zone.py (baseline ~0.513 vs. ~0.533 with
the dead zone) distinguishable from what a model with no real
predictive relationship would produce by chance, or is it within what
noise alone would explain given the eval set's size? See
docs/validation-plan.md's criterion D and
analytics/permutation_test.py's docstring for exactly what this does
and doesn't test (a model-level check, not yet the full
backtest-return version that document describes).

Uses the dead-zone dataset's own natural train/eval split (not the
cross-model-aligned split scripts/compare_label_dead_zone.py builds --
this validates the dead-zone model's OWN held-out AUC on its own
terms, which is what needs validating).

    python scripts/permutation_test_dead_zone.py
"""

from __future__ import annotations

import logging

from sklearn.metrics import roc_auc_score

from ironbridge_trader.analytics.permutation_test import run_permutation_test
from ironbridge_trader.analytics.validation_bar import PERMUTATION_TEST_MAX_P_VALUE
from ironbridge_trader.config import SETTINGS
from ironbridge_trader.features.engineering import FEATURE_NAMES
from ironbridge_trader.ml.training import train_eval_split
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MIN_MOVE_FRACTION = 0.005  # must match scripts/compare_label_dead_zone.py to validate the same result


def _auc(model, eval_) -> float:
    proba = model.predict_proba(eval_[FEATURE_NAMES])[:, 1]
    return float(roc_auc_score(eval_["label"], proba))


def main() -> None:
    db = Database(SETTINGS.db_path)
    train, eval_ = train_eval_split(db, SETTINGS, min_move_fraction=MIN_MOVE_FRACTION)

    logger.info(
        "running permutation test: %d shuffled retrains, %d train rows, %d eval rows...",
        100, len(train), len(eval_),
    )
    result = run_permutation_test(train, eval_, _auc)

    logger.info("=== Permutation test: dead-zone model's AUC ===")
    logger.info("real model AUC:      %.4f", result.real_score)
    logger.info(
        "shuffled-label AUC:  mean=%.4f  min=%.4f  max=%.4f (over %d runs)",
        sum(result.shuffled_scores) / len(result.shuffled_scores),
        min(result.shuffled_scores), max(result.shuffled_scores), result.runs,
    )
    logger.info("p-value: %.3f (fraction of shuffled runs scoring >= the real model)", result.p_value)
    logger.info(
        "=== %s (p < %.2f required) ===",
        "PASSES -- distinguishable from noise" if result.passes else "DOES NOT PASS -- consistent with noise",
        PERMUTATION_TEST_MAX_P_VALUE,
    )


if __name__ == "__main__":
    main()

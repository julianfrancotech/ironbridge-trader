"""One-off experiment, not part of the production pipeline: does
dropping near-flat "dead zone" bars from training -- where whether the
next bar closed up or down is closest to an arbitrary coin flip, and
provides the least genuine signal either way -- produce a model that's
actually better at the calls that have a real answer? See
features/engineering.py::build_training_frame's min_move_fraction
param.

Same pitfall as scripts/compare_symbol_pooling.py, avoided the same
way: naively comparing each model's own eval accuracy would compare
two different test sets (the dead-zone model's eval set excludes the
near-flat days; the baseline model's doesn't), which could look like
an improvement even if predicting the genuinely hard calls didn't get
any better. Both models here are evaluated on the exact same held-out,
dead-zone-filtered rows -- the only thing that differs is what data
each model was trained on.

Neither model is saved or versioned. Single comparison, not yet a
statistically tested result -- see docs/validation-plan.md.

    python scripts/compare_label_dead_zone.py
"""

from __future__ import annotations

import logging

import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score

from ironbridge_trader.config import SETTINGS
from ironbridge_trader.features.engineering import FEATURE_NAMES
from ironbridge_trader.ml.training import build_dataset, new_model
from ironbridge_trader.storage.db import Database

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# A next-bar move smaller than this is treated as "too flat to call."
# ~0.5% is roughly a third of a typical liquid-equity daily move (std
# around 1-1.5%), so it filters out a meaningful chunk of the noisiest
# days without gutting the dataset -- a starting point, not a tuned value.
MIN_MOVE_FRACTION = 0.005


def _fit_and_score(train: pd.DataFrame, eval_: pd.DataFrame) -> tuple[float, float]:
    model = new_model()
    model.fit(train[FEATURE_NAMES], train["label"])
    pred = model.predict(eval_[FEATURE_NAMES])
    proba = model.predict_proba(eval_[FEATURE_NAMES])[:, 1]
    accuracy = float(accuracy_score(eval_["label"], pred))
    auc = float(roc_auc_score(eval_["label"], proba)) if eval_["label"].nunique() > 1 else float("nan")
    return accuracy, auc


def main() -> None:
    db = Database(SETTINGS.db_path)
    symbols = db.symbols_with_bars()

    baseline_dataset = build_dataset(db, symbols)
    split_idx = int(len(baseline_dataset) * SETTINGS.train_test_split_ratio)
    cutoff = baseline_dataset.iloc[split_idx]["timestamp"]
    baseline_train = baseline_dataset[baseline_dataset["timestamp"] < cutoff]
    baseline_eval_candidates = baseline_dataset[baseline_dataset["timestamp"] >= cutoff]

    dead_zone_dataset = build_dataset(db, symbols, min_move_fraction=MIN_MOVE_FRACTION)
    dead_zone_train = dead_zone_dataset[dead_zone_dataset["timestamp"] < cutoff]
    dead_zone_eval = dead_zone_dataset[dead_zone_dataset["timestamp"] >= cutoff]

    if baseline_train.empty or dead_zone_train.empty or dead_zone_eval.empty:
        raise SystemExit("not enough rows for this comparison -- fetch more history")

    baseline_accuracy, baseline_auc = _fit_and_score(baseline_train, dead_zone_eval)
    dead_zone_accuracy, dead_zone_auc = _fit_and_score(dead_zone_train, dead_zone_eval)

    dropped = len(baseline_eval_candidates) - len(dead_zone_eval)
    logger.info("=== Label dead-zone comparison (min_move_fraction=%.3f) ===", MIN_MOVE_FRACTION)
    logger.info(
        "dead zone dropped %d of %d eval-window bars as too flat to call (%.1f%%)",
        dropped, len(baseline_eval_candidates), 100 * dropped / len(baseline_eval_candidates),
    )
    logger.info(
        "both evaluated on the same %d held-out non-flat rows, cutoff %s", len(dead_zone_eval), cutoff,
    )
    logger.info(
        "baseline (all bars) training:  %5d rows -- accuracy=%.4f, auc=%.4f",
        len(baseline_train), baseline_accuracy, baseline_auc,
    )
    logger.info(
        "dead-zone training:            %5d rows -- accuracy=%.4f, auc=%.4f",
        len(dead_zone_train), dead_zone_accuracy, dead_zone_auc,
    )
    if dead_zone_accuracy == baseline_accuracy:
        verdict = "tie on accuracy"
    elif dead_zone_accuracy > baseline_accuracy:
        verdict = "dead-zone wins on accuracy"
    else:
        verdict = "baseline wins on accuracy"
    logger.info("=== %s ===", verdict)


if __name__ == "__main__":
    main()

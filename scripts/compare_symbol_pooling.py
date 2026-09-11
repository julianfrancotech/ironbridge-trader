"""One-off experiment, not part of the production pipeline: does
pooling BTC-USD together with equities into one training set hurt the
model's ability to predict the STOCKS specifically, versus training on
stocks alone? See ml/training.py::build_dataset's docstring for why
this is a real, previously-untested assumption, not settled fact.

Naive comparison pitfall this avoids: simply comparing each model's
own eval_accuracy would conflate two different questions, since a
pooled model's eval set includes BTC-USD rows (which may just be
inherently harder to predict) while a stocks-only model's doesn't --
a lower pooled number could mean "crypto rows drag the average down,"
not "training on crypto hurt the stock predictions." This script
controls for that: both models are evaluated on the exact same
held-out stock rows, over the exact same date range (the eval window
is defined once, from the stocks-only dataset, and reused for both --
not just "the same row-count fraction," which would drift to a
different date given BTC-USD's denser, 7-day-a-week bar history). The
only thing that differs between the two runs is what data the model
was trained on.

Neither model is saved or versioned -- this never touches the
production model_versions history or model_dir. This is also a single
comparison, not yet the kind of repeated/statistically-tested result
docs/validation-plan.md's permutation test would require -- treat the
verdict as directional evidence for what to try next, not proof.

    python scripts/compare_symbol_pooling.py
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
    all_symbols = db.symbols_with_bars()
    stock_symbols = [s for s in all_symbols if "-" not in s]  # excludes BTC-USD-style crypto pairs
    logger.info("all symbols: %s -- stocks-only: %s", all_symbols, stock_symbols)

    stocks_dataset = build_dataset(db, stock_symbols)
    split_idx = int(len(stocks_dataset) * SETTINGS.train_test_split_ratio)
    cutoff = stocks_dataset.iloc[split_idx]["timestamp"]
    stock_train = stocks_dataset[stocks_dataset["timestamp"] < cutoff]
    stock_eval = stocks_dataset[stocks_dataset["timestamp"] >= cutoff]

    pooled_dataset = build_dataset(db, all_symbols)
    pooled_train = pooled_dataset[pooled_dataset["timestamp"] < cutoff]

    if stock_train.empty or stock_eval.empty or pooled_train.empty:
        raise SystemExit("not enough rows for this comparison -- fetch more history")

    stocks_only_accuracy, stocks_only_auc = _fit_and_score(stock_train, stock_eval)
    pooled_accuracy, pooled_auc = _fit_and_score(pooled_train, stock_eval)  # same eval set both times

    logger.info(
        "=== Symbol-pooling comparison (both evaluated on the same %d held-out stock rows, cutoff %s) ===",
        len(stock_eval), cutoff,
    )
    logger.info(
        "stocks-only training: %5d rows -- accuracy=%.4f, auc=%.4f",
        len(stock_train), stocks_only_accuracy, stocks_only_auc,
    )
    logger.info(
        "pooled training:      %5d rows -- accuracy=%.4f, auc=%.4f",
        len(pooled_train), pooled_accuracy, pooled_auc,
    )
    if stocks_only_accuracy == pooled_accuracy:
        verdict = "tie on accuracy"
    else:
        verdict = "stocks-only wins on accuracy" if stocks_only_accuracy > pooled_accuracy else "pooled wins on accuracy"
    logger.info("=== %s ===", verdict)


if __name__ == "__main__":
    main()

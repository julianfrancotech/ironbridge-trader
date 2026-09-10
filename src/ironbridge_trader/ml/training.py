"""Train, walk-forward evaluate, and version a new model.

Called from orchestration/retrain_model.py, on demand or on a cron.
Every call produces a NEW timestamped version; old versions stay on
disk and in the DB's model_versions table, so the dashboard can chart
accuracy across retrains. That is the "learns and improves over time"
loop this app implements: not online learning inside the live trading
loop (which would let one noisy day silently warp decisions), but an
explicit, auditable, re-runnable retrain step with visible before/after
metrics -- you decide when a new version is trusted, not the model.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from ironbridge_trader.config import Settings
from ironbridge_trader.features.engineering import FEATURE_NAMES, build_training_frame
from ironbridge_trader.ml.model import PricePredictor
from ironbridge_trader.storage.db import Database


def build_dataset(db: Database) -> pd.DataFrame:
    """Pool training rows across every symbol with stored bars.

    One pooled model rather than one per symbol: every feature is a
    normalized ratio or z-score, not a raw price (see
    features/engineering.py), so patterns learned on a heavily-traded
    symbol transfer to a thinner one, and the model has years x N-symbols
    of rows to learn from instead of years x 1.
    """
    frames = []
    for symbol in db.symbols_with_bars():
        bars = db.get_bars(symbol)
        frame = build_training_frame(bars)
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        raise ValueError("no bars in the database -- run scripts/fetch_data.py first")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def train_and_version(db: Database, settings: Settings) -> PricePredictor:
    dataset = build_dataset(db)
    split_idx = int(len(dataset) * settings.train_test_split_ratio)
    train, eval_ = dataset.iloc[:split_idx], dataset.iloc[split_idx:]
    if train.empty or eval_.empty:
        raise ValueError("not enough rows for a train/eval split -- fetch more history")

    model = HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200)
    model.fit(train[FEATURE_NAMES], train["label"])

    eval_pred = model.predict(eval_[FEATURE_NAMES])
    eval_proba = model.predict_proba(eval_[FEATURE_NAMES])[:, 1]
    accuracy = float(accuracy_score(eval_["label"], eval_pred))
    auc = (
        float(roc_auc_score(eval_["label"], eval_proba))
        if eval_["label"].nunique() > 1
        else float("nan")
    )

    version = datetime.now(UTC).strftime("v%Y%m%d-%H%M%S")
    predictor = PricePredictor(model=model, version=version)
    predictor.save(settings.model_dir)

    db.insert_model_version(
        version=version,
        trained_at=datetime.now(UTC),
        train_rows=len(train),
        eval_rows=len(eval_),
        eval_accuracy=accuracy,
        eval_auc=auc,
        feature_names=FEATURE_NAMES,
    )
    return predictor

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


def new_model() -> HistGradientBoostingClassifier:
    """The one place these hyperparameters are set -- shared by the
    production training path and any experiment/diagnostic script
    (e.g. scripts/compare_symbol_pooling.py) that needs to fit a
    directly comparable model, so the two can never silently drift
    apart.
    """
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.06, max_iter=200)


def build_dataset(
    db: Database, symbols: list[str] | None = None, min_move_fraction: float = 0.0
) -> pd.DataFrame:
    """Pool training rows across `symbols` (default: every symbol with
    stored bars).

    Pooling across symbols, rather than training one model per symbol,
    is the working assumption, not settled fact: every feature is a
    normalized ratio or z-score, not a raw price (see
    features/engineering.py), so patterns learned on a heavily-traded
    symbol transfer to a thinner one in principle, and the model has
    years x N-symbols of rows to learn from instead of years x 1. But
    pooling instruments as different as a large-cap stock and BTC-USD
    also assumes the learned relationship is the same for both, which
    is a real assumption worth checking, not free -- see
    scripts/compare_symbol_pooling.py, which uses this `symbols` param
    to test it directly.

    min_move_fraction is passed straight through to
    build_training_frame (default 0, no dead zone) -- see that
    docstring and scripts/compare_label_dead_zone.py.
    """
    frames = []
    for symbol in symbols if symbols is not None else db.symbols_with_bars():
        bars = db.get_bars(symbol)
        frame = build_training_frame(bars, min_move_fraction=min_move_fraction)
        frame["symbol"] = symbol
        frames.append(frame)
    if not frames:
        raise ValueError("no bars in the database -- run scripts/fetch_data.py first")
    return pd.concat(frames, ignore_index=True).sort_values("timestamp").reset_index(drop=True)


def train_eval_split(
    db: Database, settings: Settings, symbols: list[str] | None = None, min_move_fraction: float = 0.0
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """build_dataset, then a walk-forward split by
    settings.train_test_split_ratio (train = earlier rows, eval = later
    rows, no shuffling -- shuffling here would leak future information
    into training). Shared by the production path and
    experiment/diagnostic scripts (scripts/permutation_test_dead_zone.py)
    so the split logic never drifts between them.
    """
    dataset = build_dataset(db, symbols, min_move_fraction=min_move_fraction)
    split_idx = int(len(dataset) * settings.train_test_split_ratio)
    train, eval_ = dataset.iloc[:split_idx], dataset.iloc[split_idx:]
    if train.empty or eval_.empty:
        raise ValueError("not enough rows for a train/eval split -- fetch more history")
    return train, eval_


def train_and_version(db: Database, settings: Settings) -> PricePredictor:
    train, eval_ = train_eval_split(db, settings, min_move_fraction=settings.min_move_fraction)

    model = new_model()
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

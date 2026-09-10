"""Workflow: train a new model version from everything currently stored
and report its walk-forward evaluation metrics. Thin wrapper around
ml/training.py -- the orchestration layer's job is picking *which*
Database and Settings to run it against and shaping the result for
whoever called it (a script today; a scheduler tomorrow).
"""

from __future__ import annotations

from dataclasses import dataclass

from ironbridge_trader.config import Settings
from ironbridge_trader.ml.model import PricePredictor
from ironbridge_trader.ml.training import train_and_version
from ironbridge_trader.storage.db import Database


@dataclass(frozen=True, slots=True)
class RetrainResult:
    predictor: PricePredictor
    train_rows: int
    eval_rows: int
    eval_accuracy: float
    eval_auc: float


def run(db: Database, settings: Settings) -> RetrainResult:
    predictor = train_and_version(db, settings)
    latest = next(v for v in db.model_versions() if v["version"] == predictor.version)
    return RetrainResult(
        predictor=predictor,
        train_rows=latest["train_rows"],
        eval_rows=latest["eval_rows"],
        eval_accuracy=latest["eval_accuracy"],
        eval_auc=latest["eval_auc"],
    )

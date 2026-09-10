"""The trained predictive model, wrapped to satisfy the Predictor protocol.

Gradient-boosted trees, not a neural net: this is tabular data (a few
thousand rows, nine engineered features), exactly the regime where
boosted trees reliably match or beat deep learning, need no GPU or
feature scaling, train in seconds, and stay inspectable via
feature_importances_. Reaching for a neural network here would be the
overengineering the brief explicitly warns against.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from ironbridge_trader.domain.models import Bar, QuantSignal
from ironbridge_trader.features.engineering import FEATURE_NAMES, compute_features


class PricePredictor:
    """predict(symbol, bars) -> QuantSignal. Satisfies protocols.Predictor."""

    def __init__(self, model: HistGradientBoostingClassifier, version: str) -> None:
        self._model = model
        self.version = version

    def predict(self, symbol: str, bars: list[Bar]) -> QuantSignal:
        features = compute_features(bars)
        if features is None:
            raise ValueError(f"not enough bar history for {symbol} to compute features")
        x = pd.DataFrame([[features[name] for name in FEATURE_NAMES]], columns=FEATURE_NAMES)
        prob_up = float(self._model.predict_proba(x)[0][1])
        return QuantSignal(
            symbol=symbol,
            as_of=bars[-1].timestamp,
            prob_up=prob_up,
            model_version=self.version,
            features=features,
        )

    def save(self, model_dir: Path) -> Path:
        path = model_dir / f"{self.version}.pkl"
        with open(path, "wb") as f:
            pickle.dump(self._model, f)
        return path

    @classmethod
    def load(cls, model_dir: Path, version: str) -> PricePredictor:
        with open(model_dir / f"{version}.pkl", "rb") as f:
            model = pickle.load(f)
        return cls(model=model, version=version)

"""Permutation test: is a result distinguishable from what a model
with no real relationship between features and labels would produce
by chance? Shuffles the TRAINING labels only (breaking any real
feature-label relationship while keeping the same class balance) and
retrains PERMUTATION_TEST_RUNS times, building a null distribution for
whatever metric the caller cares about; the real (unshuffled) result's
p-value is the fraction of shuffled runs that scored at least as well.
The held-out eval set's labels are never shuffled -- only training
data changes between runs, so every run is graded against the same
true outcomes.

This is the model-level version of docs/validation-plan.md's
criterion D. That document's literal wording is about backtested
excess RETURN over the benchmark, which would need a full bar-by-bar
backtest per permutation run -- much more expensive than retraining
alone (100 full backtests vs. 100 fast model fits). This tests the
same underlying question (is this better than noise) at the ML-metric
level (AUC/accuracy on a held-out set), which is what's actually
needed to validate a training/feature-engineering change (e.g.
scripts/compare_label_dead_zone.py's result) before it's even worth
running the heavier backtest-level version on it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from ironbridge_trader.analytics.validation_bar import (
    PERMUTATION_TEST_MAX_P_VALUE,
    PERMUTATION_TEST_RUNS,
)
from ironbridge_trader.features.engineering import FEATURE_NAMES
from ironbridge_trader.ml.training import new_model


class ScoreFn(Protocol):
    def __call__(self, model: object, eval_: pd.DataFrame) -> float: ...


@dataclass(frozen=True, slots=True)
class PermutationTestResult:
    real_score: float
    shuffled_scores: list[float]
    runs: int

    @property
    def p_value(self) -> float:
        """Fraction of label-shuffled (no real signal, by construction)
        runs that scored at least as well as the real model. Low means
        a model with no real relationship rarely does this well --
        i.e. the real result is unlikely to be noise.
        """
        at_least_as_good = sum(1 for s in self.shuffled_scores if s >= self.real_score)
        return at_least_as_good / self.runs

    @property
    def passes(self) -> bool:
        return self.p_value < PERMUTATION_TEST_MAX_P_VALUE


def run_permutation_test(
    train: pd.DataFrame,
    eval_: pd.DataFrame,
    score_fn: ScoreFn,
    *,
    runs: int = PERMUTATION_TEST_RUNS,
    seed: int = 0,
) -> PermutationTestResult:
    """train/eval: frames shaped like ml.training.build_dataset's output
    (FEATURE_NAMES columns + "label"). score_fn(fitted_model, eval_) ->
    float, higher-is-better (e.g. AUC), evaluated against the same
    held-out `eval_` for both the real model and every shuffled run.
    """
    real_model = new_model()
    real_model.fit(train[FEATURE_NAMES], train["label"])
    real_score = score_fn(real_model, eval_)

    labels = train["label"].to_numpy()
    features = train[FEATURE_NAMES]
    rng = np.random.default_rng(seed)

    shuffled_scores = []
    for _ in range(runs):
        shuffled_labels = rng.permutation(labels)
        model = new_model()
        model.fit(features, shuffled_labels)
        shuffled_scores.append(score_fn(model, eval_))

    return PermutationTestResult(real_score=real_score, shuffled_scores=shuffled_scores, runs=runs)

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from ironbridge_trader.analytics.permutation_test import run_permutation_test
from ironbridge_trader.features.engineering import FEATURE_NAMES


def _auc_score(model, eval_: pd.DataFrame) -> float:
    proba = model.predict_proba(eval_[FEATURE_NAMES])[:, 1]
    return float(roc_auc_score(eval_["label"], proba))


def _frame(rng: np.random.Generator, n: int, signal_feature: np.ndarray | None) -> pd.DataFrame:
    frame = pd.DataFrame({name: rng.normal(size=n) for name in FEATURE_NAMES})
    if signal_feature is not None:
        frame[FEATURE_NAMES[0]] = signal_feature
        frame["label"] = (signal_feature > 0).astype(float)
    else:
        frame["label"] = rng.integers(0, 2, size=n).astype(float)
    return frame


def test_run_permutation_test_detects_a_strong_real_signal():
    # An easy, cleanly separable problem -- FEATURE_NAMES[0] alone
    # perfectly determines the label. Any reasonable classifier should
    # learn this almost perfectly, so the real model's AUC should be
    # far above what any label-shuffled (no real relationship) run
    # could match.
    rng = np.random.default_rng(1)
    train = _frame(rng, 300, signal_feature=rng.normal(size=300))
    eval_ = _frame(rng, 150, signal_feature=rng.normal(size=150))

    result = run_permutation_test(train, eval_, _auc_score, runs=25)

    assert result.real_score > 0.9
    assert result.passes is True
    assert result.p_value < 0.05


def test_run_permutation_test_does_not_flag_pure_noise_as_significant():
    # Labels have no relationship to any feature at all -- the real
    # model here is, structurally, just another draw from the same
    # "no signal" process the shuffled runs are, so it should land
    # comfortably inside that null distribution, not stand out from it.
    rng = np.random.default_rng(7)
    train = _frame(rng, 300, signal_feature=None)
    eval_ = _frame(rng, 150, signal_feature=None)

    result = run_permutation_test(train, eval_, _auc_score, runs=40)

    assert result.real_score < 0.65  # nowhere near the strong-signal case's >0.9
    assert result.passes is False


def test_permutation_test_result_p_value_uses_the_actual_run_count():
    result = run_permutation_test(
        pd.DataFrame({**{name: [0.0, 1.0] for name in FEATURE_NAMES}, "label": [0.0, 1.0]}),
        pd.DataFrame({**{name: [0.0, 1.0] for name in FEATURE_NAMES}, "label": [0.0, 1.0]}),
        _auc_score,
        runs=5,
    )

    assert result.runs == 5
    assert len(result.shuffled_scores) == 5

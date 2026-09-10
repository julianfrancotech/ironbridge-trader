from ironbridge_trader.features.engineering import (
    FEATURE_NAMES,
    build_training_frame,
    compute_features,
)


def test_compute_features_none_when_not_enough_history(bars):
    assert compute_features(bars[:5]) is None


def test_compute_features_none_on_empty_history():
    assert compute_features([]) is None


def test_compute_features_returns_all_feature_names(bars):
    features = compute_features(bars)
    assert features is not None
    assert set(features.keys()) == set(FEATURE_NAMES)
    assert all(isinstance(v, float) for v in features.values())


def test_build_training_frame_has_label_and_no_nan(bars):
    frame = build_training_frame(bars)
    assert "label" in frame.columns
    assert frame[FEATURE_NAMES + ["label"]].isna().sum().sum() == 0
    assert 0 < len(frame) < len(bars)  # some leading bars dropped for lookback, last bar has no label
    assert set(frame["label"].unique()) <= {0.0, 1.0}

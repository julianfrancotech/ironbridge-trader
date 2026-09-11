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


def test_build_training_frame_dead_zone_drops_small_moves(bars):
    baseline = build_training_frame(bars)
    filtered = build_training_frame(bars, min_move_fraction=0.01)

    assert 0 < len(filtered) < len(baseline)
    assert (filtered["next_return"].abs() > 0.01).all()


def test_build_training_frame_label_still_matches_next_return_sign_with_dead_zone(bars):
    filtered = build_training_frame(bars, min_move_fraction=0.005)

    assert (filtered.loc[filtered["label"] == 1.0, "next_return"] > 0).all()
    assert (filtered.loc[filtered["label"] == 0.0, "next_return"] < 0).all()


def test_build_training_frame_zero_min_move_fraction_matches_default(bars):
    baseline = build_training_frame(bars)
    explicit_zero = build_training_frame(bars, min_move_fraction=0.0)

    assert len(baseline) == len(explicit_zero)

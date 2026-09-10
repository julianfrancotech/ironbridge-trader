from sklearn.ensemble import HistGradientBoostingClassifier

from ironbridge_trader.features.engineering import FEATURE_NAMES, build_training_frame
from ironbridge_trader.ml.model import PricePredictor


def test_predictor_predict_returns_valid_signal(bars, tmp_path):
    frame = build_training_frame(bars)
    model = HistGradientBoostingClassifier(max_iter=20).fit(frame[FEATURE_NAMES], frame["label"])
    predictor = PricePredictor(model=model, version="vtest")

    signal = predictor.predict("TEST", bars)

    assert signal.symbol == "TEST"
    assert 0.0 <= signal.prob_up <= 1.0
    assert signal.model_version == "vtest"
    assert set(signal.features.keys()) == set(FEATURE_NAMES)


def test_predictor_save_and_load_roundtrip(bars, tmp_path):
    frame = build_training_frame(bars)
    model = HistGradientBoostingClassifier(max_iter=20).fit(frame[FEATURE_NAMES], frame["label"])
    predictor = PricePredictor(model=model, version="vtest")
    predictor.save(tmp_path)

    loaded = PricePredictor.load(tmp_path, "vtest")
    original = predictor.predict("TEST", bars)
    restored = loaded.predict("TEST", bars)

    assert original.prob_up == restored.prob_up

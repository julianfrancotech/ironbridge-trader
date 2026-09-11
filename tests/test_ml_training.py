import pytest

from ironbridge_trader.config import Settings
from ironbridge_trader.features.engineering import FEATURE_NAMES
from ironbridge_trader.ml.training import build_dataset, new_model, train_eval_split
from ironbridge_trader.storage.db import Database


def test_build_dataset_pools_every_symbol_by_default(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)  # conftest's default bars fixture uses symbol "TEST"

    dataset = build_dataset(db)

    assert set(dataset["symbol"]) == {"TEST"}


def test_build_dataset_filters_to_the_given_symbols(tmp_path):
    from conftest import make_bars

    db = Database(tmp_path / "t.db")
    db.upsert_bars(make_bars(symbol="A"))
    db.upsert_bars(make_bars(symbol="B"))

    dataset = build_dataset(db, symbols=["A"])

    assert set(dataset["symbol"]) == {"A"}


def test_build_dataset_passes_min_move_fraction_through(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)

    baseline = build_dataset(db)
    dead_zone = build_dataset(db, min_move_fraction=0.01)

    assert len(dead_zone) < len(baseline)


def test_build_dataset_raises_when_no_bars_exist(tmp_path):
    db = Database(tmp_path / "t.db")
    with pytest.raises(ValueError):
        build_dataset(db)


def test_new_model_returns_a_fresh_unfitted_instance_each_call():
    a = new_model()
    b = new_model()

    assert a is not b  # independent instances, no shared/mutated state across calls


def test_train_eval_split_is_walk_forward_not_shuffled(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)
    settings = Settings(train_test_split_ratio=0.8)

    train, eval_ = train_eval_split(db, settings)

    assert not train.empty
    assert not eval_.empty
    assert set(train.columns) >= {*FEATURE_NAMES, "label", "timestamp"}
    assert train["timestamp"].max() <= eval_["timestamp"].min()  # train is strictly earlier


def test_train_eval_split_raises_when_not_enough_rows_for_either_half(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)
    settings = Settings(train_test_split_ratio=1.0)  # everything goes to train, eval is empty

    with pytest.raises(ValueError):
        train_eval_split(db, settings)

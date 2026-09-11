import json
from decimal import Decimal

import pytest

from ironbridge_trader.config import Settings, _read_overrides, save_overrides


def test_round_trips_decimal_and_tuple_fields(tmp_path):
    path = tmp_path / "overrides.json"
    save_overrides(
        {
            "risk_fraction": Decimal("0.02"),
            "symbols": ("AAPL", "NVDA"),
            "agent_max_tool_iterations": 4,
        },
        path=path,
    )

    overrides = _read_overrides(path)

    assert overrides["risk_fraction"] == Decimal("0.02")
    assert isinstance(overrides["risk_fraction"], Decimal)
    assert overrides["symbols"] == ("AAPL", "NVDA")
    assert overrides["agent_max_tool_iterations"] == 4


def test_settings_built_from_overrides_uses_saved_values(tmp_path):
    path = tmp_path / "overrides.json"
    save_overrides({"max_position_fraction": Decimal("0.5")}, path=path)

    settings = Settings(**_read_overrides(path))

    assert settings.max_position_fraction == Decimal("0.5")
    assert settings.margin_rate == Decimal("0.25")  # untouched fields keep their default


def test_missing_file_yields_no_overrides(tmp_path):
    assert _read_overrides(tmp_path / "does_not_exist.json") == {}


def test_corrupt_file_yields_no_overrides_instead_of_raising(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text("{not valid json")
    assert _read_overrides(path) == {}


def test_unknown_key_is_ignored_on_read(tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"anthropic_api_key": "sk-should-not-load", "max_position_fraction": 0.5}))

    overrides = _read_overrides(path)

    assert "anthropic_api_key" not in overrides
    assert overrides["max_position_fraction"] == Decimal("0.5")


def test_save_rejects_non_tunable_key(tmp_path):
    with pytest.raises(ValueError):
        save_overrides({"anthropic_api_key": "sk-nope"}, path=tmp_path / "overrides.json")


def test_save_rejects_alpaca_secrets(tmp_path):
    # Same reasoning as anthropic_api_key: a secret, edited via .env only,
    # never through a UI meant to be safe to screen-share.
    with pytest.raises(ValueError):
        save_overrides({"alpaca_api_key": "nope"}, path=tmp_path / "overrides.json")
    with pytest.raises(ValueError):
        save_overrides({"alpaca_secret_key": "nope"}, path=tmp_path / "overrides.json")


def test_data_and_execution_provider_round_trip(tmp_path):
    path = tmp_path / "overrides.json"
    save_overrides({"data_provider": "alpaca", "execution_provider": "alpaca_paper"}, path=path)

    settings = Settings(**_read_overrides(path))

    assert settings.data_provider == "alpaca"
    assert settings.execution_provider == "alpaca_paper"


def test_trading_enabled_round_trips_as_a_bool(tmp_path):
    path = tmp_path / "overrides.json"
    save_overrides({"trading_enabled": False}, path=path)

    overrides = _read_overrides(path)

    assert overrides["trading_enabled"] is False
    assert Settings(**overrides).trading_enabled is False


def test_transaction_cost_fields_round_trip_as_decimal(tmp_path):
    path = tmp_path / "overrides.json"
    save_overrides(
        {"transaction_cost_bps": Decimal(25), "commission_per_trade": Decimal("1.50")}, path=path
    )

    overrides = _read_overrides(path)

    assert overrides["transaction_cost_bps"] == Decimal(25)
    assert isinstance(overrides["commission_per_trade"], Decimal)
    settings = Settings(**overrides)
    assert settings.transaction_cost_bps == Decimal(25)
    assert settings.commission_per_trade == Decimal("1.50")


def test_min_move_fraction_round_trips_as_a_plain_float(tmp_path):
    path = tmp_path / "overrides.json"
    save_overrides({"min_move_fraction": 0.01}, path=path)

    overrides = _read_overrides(path)

    assert overrides["min_move_fraction"] == 0.01
    assert Settings(**overrides).min_move_fraction == 0.01

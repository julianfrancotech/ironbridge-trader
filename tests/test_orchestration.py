from ironbridge_trader.config import Settings
from ironbridge_trader.orchestration.run_decision_cycle import build_decision_maker
from ironbridge_trader.services.threshold_agent import ThresholdDecisionService
from ironbridge_trader.storage.db import Database


def test_build_decision_maker_falls_back_without_api_key(tmp_path):
    settings = Settings(anthropic_api_key=None)
    db = Database(tmp_path / "t.db")

    decision_maker = build_decision_maker(settings, db)

    assert isinstance(decision_maker, ThresholdDecisionService)


def test_build_decision_maker_uses_claude_when_api_key_present(tmp_path):
    settings = Settings(anthropic_api_key="sk-fake-key-for-wiring-test-only")
    db = Database(tmp_path / "t.db")

    decision_maker = build_decision_maker(settings, db)

    assert type(decision_maker).__name__ == "ClaudeTradingAgent"

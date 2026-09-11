from ironbridge_trader.adapters.alpaca_broker import AlpacaBroker
from ironbridge_trader.adapters.paper_broker import PaperBroker
from ironbridge_trader.config import Settings
from ironbridge_trader.orchestration import run_decision_cycle
from ironbridge_trader.orchestration.run_decision_cycle import (
    build_decision_maker,
    build_execution_client,
)
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


def test_build_execution_client_defaults_to_paper_broker():
    settings = Settings()

    assert isinstance(build_execution_client(settings), PaperBroker)


def test_build_execution_client_ignores_alpaca_credentials_unless_explicitly_selected():
    # execution_provider still defaults to paper_broker even with Alpaca
    # keys present -- unlike the decision-maker, this isn't auto-switched
    # (see build_execution_client's docstring: real orders deserve an
    # explicit opt-in, not an implicit one).
    settings = Settings(alpaca_api_key="key", alpaca_secret_key="secret")

    assert isinstance(build_execution_client(settings), PaperBroker)


def test_build_execution_client_uses_alpaca_when_selected_and_credentials_present():
    settings = Settings(
        execution_provider="alpaca_paper", alpaca_api_key="key", alpaca_secret_key="secret",
    )

    assert isinstance(build_execution_client(settings), AlpacaBroker)


def test_build_execution_client_falls_back_to_paper_broker_without_alpaca_credentials():
    settings = Settings(execution_provider="alpaca_paper", alpaca_api_key=None, alpaca_secret_key=None)

    assert isinstance(build_execution_client(settings), PaperBroker)


def test_run_skips_the_whole_cycle_when_trading_is_disabled(tmp_path):
    # No trained model exists in this fresh db -- if the kill switch
    # check didn't run first, this would raise SystemExit instead of
    # returning cleanly, proving the check happens before anything else.
    settings = Settings(trading_enabled=False)
    db = Database(tmp_path / "t.db")

    assert run_decision_cycle.run(db, settings) == []

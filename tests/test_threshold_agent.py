from datetime import UTC, datetime

import pytest

from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Action, Position, QuantSignal
from ironbridge_trader.services.threshold_agent import ThresholdDecisionService


def make_signal(prob_up: float) -> QuantSignal:
    return QuantSignal(
        symbol="TEST", as_of=datetime.now(UTC), prob_up=prob_up,
        model_version="vtest", features={},
    )


def test_holds_when_signal_is_weak():
    service = ThresholdDecisionService(Settings(decision_confidence_floor=0.55))
    decision = service.decide("TEST", bars=[], signal=make_signal(0.52), position=Position(symbol="TEST"))
    assert decision.action is Action.HOLD


def test_buys_on_strong_upward_signal():
    service = ThresholdDecisionService(Settings(decision_confidence_floor=0.55))
    decision = service.decide("TEST", bars=[], signal=make_signal(0.9), position=Position(symbol="TEST"))
    assert decision.action is Action.BUY
    assert decision.confidence == pytest.approx(0.8)


def test_sells_on_strong_downward_signal():
    service = ThresholdDecisionService(Settings(decision_confidence_floor=0.55))
    decision = service.decide("TEST", bars=[], signal=make_signal(0.05), position=Position(symbol="TEST"))
    assert decision.action is Action.SELL

"""Deterministic fallback decision-maker: no LLM call, just a threshold on
the quant model's probability. Two jobs:

  1. Lets the whole app run end-to-end with zero API key (dashboard,
     paper trading, backtest all work) -- the LLM agent is additive, not
     load-bearing.
  2. Gives the ClaudeTradingAgent something to be *compared against* in
     the dashboard: is the tool-using agent's judgment actually adding
     anything over "just trust the model above a confidence floor"? If
     it isn't, that's worth knowing, not hand-waved away.

Satisfies the same protocols.DecisionMaker as ClaudeTradingAgent -- the
engine, risk manager, and broker never know which one produced a Decision.
"""

from __future__ import annotations

from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Action, Bar, Decision, Position, QuantSignal


class ThresholdDecisionService:
    """agent_kind == "quant_threshold"."""

    def __init__(self, settings: Settings) -> None:
        self._confidence_floor = settings.decision_confidence_floor

    def decide(
        self, symbol: str, bars: list[Bar], signal: QuantSignal, position: Position
    ) -> Decision:
        # prob_up in [0,1]; distance from 0.5 maps linearly to confidence in [0,1].
        confidence = abs(signal.prob_up - 0.5) * 2

        if confidence < self._confidence_floor:
            action = Action.HOLD
            reason = (
                f"prob_up={signal.prob_up:.2f} is too close to 0.5 "
                f"(confidence {confidence:.2f} < floor {self._confidence_floor:.2f})."
            )
        elif signal.prob_up > 0.5:
            action = Action.BUY
            reason = f"prob_up={signal.prob_up:.2f} clears the confidence floor on the upside."
        else:
            action = Action.SELL
            reason = f"prob_up={signal.prob_up:.2f} clears the confidence floor on the downside."

        return Decision(
            symbol=symbol, as_of=signal.as_of, action=action, confidence=confidence,
            rationale=reason, quant_signal=signal, agent_kind="quant_threshold",
        )

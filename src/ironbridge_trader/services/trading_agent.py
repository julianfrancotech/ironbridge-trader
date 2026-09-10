"""ReAct-style trading agent: Claude drives a bounded tool-calling loop to
decide BUY/SELL/HOLD for one symbol, grounded in tool results
(services/tools.py) instead of free-form reasoning over numbers typed
into the prompt.

This is an "external services" module in the same sense a production
system's identity-service or accounts-service client would be -- a
client wrapping calls to a system this app doesn't own (here, the
Claude API instead of another internal microservice).

Why an agent loop instead of one classification call: a single call
forces every possible piece of context into one prompt whether it's
needed or not. Here the model decides what it needs -- for a strongly
directional quant signal it might submit a decision after one tool
call; for a borderline one it might also pull price history and its
own past decisions before deciding. That's the actual definition of
"agentic" being used here: bounded autonomy over *which actions to take
to gather information*, not autonomy over the consequential action
itself (placing a trade), which always still passes through the
deterministic risk manager afterward.

Two guardrails against the standard failure modes of tool-calling agents:
  - agent_max_tool_iterations bounds the loop, so a confused agent can't
    call tools forever instead of deciding.
  - if the budget runs out, decide() returns a HOLD, not an exception --
    "I don't know" fails safe to "do nothing" in a trading context.
"""

from __future__ import annotations

import json
import logging

import anthropic

from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Action, Bar, Decision, Position, QuantSignal
from ironbridge_trader.resilience import retry_with_backoff
from ironbridge_trader.services.tools import TOOL_SCHEMAS, AgentContext, run_tool
from ironbridge_trader.storage.db import Database

logger = logging.getLogger(__name__)

# Only retry failures the API itself flags as transient (dropped
# connection, timeout, rate limit, its own 5xx). A 4xx like a bad API
# key or a malformed request will fail identically on every attempt --
# retrying it just burns the iteration budget and delays the real error.
_TRANSIENT_ANTHROPIC_ERRORS = (
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
)

SYSTEM_PROMPT = """You are a cautious trading analyst assistant for a PAPER-TRADING \
simulation -- no real money moves as a result of your decisions. For the symbol you are \
given, use the tools available to gather context, then call submit_decision exactly once.

Ground every claim in a tool result you actually retrieved. Do not invent prices, \
indicator values, or past decisions. The quant model's prob_up is your single most \
reliable signal; treat price history and your own recent decisions as context that can \
raise or lower your confidence in it, not replace it. When the model is only weakly \
directional (prob_up close to 0.5) or your recent decisions have been flip-flopping \
without a persistent trend, prefer HOLD over forcing a call.

You have a limited number of tool calls. Gather only what would change your decision, \
then submit."""


class ClaudeTradingAgent:
    """Satisfies protocols.DecisionMaker. Produces Decisions with agent_kind="claude"."""

    def __init__(self, settings: Settings, db: Database) -> None:
        self._settings = settings
        self._db = db
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def decide(
        self, symbol: str, bars: list[Bar], signal: QuantSignal, position: Position
    ) -> Decision:
        ctx = AgentContext(
            symbol=symbol, bars=bars, signal=signal, position=position,
            db=self._db, settings=self._settings,
        )
        messages: list[dict] = [
            {
                "role": "user",
                "content": (
                    f"Decide an action for {symbol} as of {bars[-1].timestamp.date()}. "
                    "Use tools as needed, then call submit_decision."
                ),
            }
        ]

        for _ in range(self._settings.agent_max_tool_iterations):
            response = self._call_claude(messages)
            messages.append({"role": "assistant", "content": response.content})

            tool_uses = [block for block in response.content if block.type == "tool_use"]
            if not tool_uses:
                break  # model stopped talking without deciding -- fall through to the safe default

            tool_results = []
            for block in tool_uses:
                if block.name == "submit_decision":
                    return self._build_decision(symbol, signal, block.input)
                result = run_tool(block.name, block.input, ctx)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)}
                )
            messages.append({"role": "user", "content": tool_results})

        logger.warning(
            "agent exhausted its tool-call budget for %s without submit_decision; defaulting to HOLD",
            symbol,
        )
        return Decision(
            symbol=symbol, as_of=signal.as_of, action=Action.HOLD, confidence=0.0,
            rationale=(
                "Agent did not reach a decision within its tool-call budget; "
                "defaulting to HOLD as a safe fallback."
            ),
            quant_signal=signal, agent_kind="claude",
        )

    @retry_with_backoff(max_attempts=3, backoff_seconds=2.0, retry_on=_TRANSIENT_ANTHROPIC_ERRORS)
    def _call_claude(self, messages: list[dict]) -> anthropic.types.Message:
        return self._client.messages.create(
            model=self._settings.anthropic_model,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            messages=messages,
        )

    @staticmethod
    def _build_decision(symbol: str, signal: QuantSignal, tool_input: dict) -> Decision:
        action = Action[tool_input["action"]]
        confidence = max(0.0, min(1.0, float(tool_input["confidence"])))
        return Decision(
            symbol=symbol, as_of=signal.as_of, action=action, confidence=confidence,
            rationale=str(tool_input["rationale"]), quant_signal=signal, agent_kind="claude",
        )

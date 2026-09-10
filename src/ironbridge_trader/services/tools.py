"""Tool definitions for the ReAct-style trading agent (services/trading_agent.py).

Each tool is a read-only window onto state the app already computed --
the quant model's opinion, the current position, the agent's own past
decisions for this symbol. The agent chooses which of these to call and
in what order (that autonomy is what makes this "agentic" rather than a
fixed pipeline), but every tool here is side-effect-free. The one tool
with teeth, submit_decision, doesn't touch the world either -- it just
ends the agent's turn with a structured verdict. The deterministic risk
layer and paper broker act on that verdict afterward, never the agent
directly. That separation -- the agent proposes, code disposes -- is
the safety pattern this module exists to enforce.
"""

from __future__ import annotations

from dataclasses import dataclass

from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Bar, Position, QuantSignal
from ironbridge_trader.storage.db import Database

TOOL_SCHEMAS = [
    {
        "name": "get_price_history",
        "description": (
            "Recent daily closes for the symbol plus simple return stats "
            "over a few horizons. Use this to see the actual price action, "
            "not just the model's summary of it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lookback_bars": {
                    "type": "integer",
                    "description": "How many recent daily bars to return (default 15, max 60).",
                }
            },
        },
    },
    {
        "name": "get_quant_signal",
        "description": (
            "The trained model's current opinion for this symbol: prob_up is its "
            "estimated probability that tomorrow's close is higher than today's, "
            "plus the technical-indicator feature values it was computed from."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_position",
        "description": "Current paper position (quantity and average price) held in this symbol.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_recent_decisions",
        "description": (
            "Your own last few decisions for this symbol, with the action, confidence, "
            "and rationale you gave each time. Use this to avoid flip-flopping on noise "
            "and to check whether your recent reasoning has been panning out."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many past decisions to return (default 5)."}
            },
        },
    },
    {
        "name": "get_risk_limits",
        "description": (
            "The account's hard risk limits. A BUY/SELL you submit may still be "
            "blocked afterward by the risk manager if it would breach these -- "
            "factor that in rather than being surprised by it."
        ),
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "submit_decision",
        "description": (
            "Submit your final decision for this symbol. Call this exactly once, "
            "after you've gathered whatever context you need -- it ends your turn."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["BUY", "SELL", "HOLD"]},
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in this action, from 0.0 to 1.0.",
                },
                "rationale": {
                    "type": "string",
                    "description": "2-4 sentences explaining the decision in plain language, "
                    "grounded in the tool results you looked at.",
                },
            },
            "required": ["action", "confidence", "rationale"],
        },
    },
]


@dataclass(frozen=True, slots=True)
class AgentContext:
    symbol: str
    bars: list[Bar]
    signal: QuantSignal
    position: Position
    db: Database
    settings: Settings


def run_tool(name: str, tool_input: dict, ctx: AgentContext) -> dict:
    """Execute one non-terminal tool call and return a JSON-serializable result.
    submit_decision is handled separately by the caller, not here.
    """
    if name == "get_price_history":
        lookback = min(int(tool_input.get("lookback_bars", 15)), 60)
        window = ctx.bars[-lookback:]
        closes = [float(b.close) for b in window]

        def pct_return(n: int) -> float | None:
            if len(ctx.bars) <= n:
                return None
            return round(float(ctx.bars[-1].close) / float(ctx.bars[-1 - n].close) - 1, 4)

        return {
            "symbol": ctx.symbol,
            "closes_oldest_to_newest": closes,
            "return_5_bars": pct_return(5),
            "return_20_bars": pct_return(20),
            "return_60_bars": pct_return(60),
        }

    if name == "get_quant_signal":
        return {
            "symbol": ctx.signal.symbol,
            "prob_up": round(ctx.signal.prob_up, 4),
            "model_version": ctx.signal.model_version,
            "features": {k: round(v, 4) for k, v in ctx.signal.features.items()},
        }

    if name == "get_position":
        return {
            "symbol": ctx.position.symbol,
            "quantity": ctx.position.quantity,
            "average_price": float(ctx.position.average_price),
        }

    if name == "get_recent_decisions":
        limit = int(tool_input.get("limit", 5))
        return {"decisions": ctx.db.recent_decisions(ctx.symbol, limit=limit)}

    if name == "get_risk_limits":
        return {
            "max_position_size": ctx.settings.max_position_size,
            "margin_rate": float(ctx.settings.margin_rate),
            "account_equity": float(ctx.settings.account_equity),
            "risk_fraction": float(ctx.settings.risk_fraction),
            "stop_loss_fraction": float(ctx.settings.stop_loss_fraction),
        }

    raise ValueError(f"unknown tool: {name}")

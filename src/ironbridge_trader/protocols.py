"""Structural interfaces (typing.Protocol, not ABCs) the engine and
orchestration layer depend on. This is the same role a production
trading service gets from its adapter/service boundaries: a stable
interface concrete adapters and services satisfy structurally, so the
business logic never imports a concrete implementation directly. Swap
yfinance for a real broker feed, or the Claude agent for a different
one, and nothing that depends on these protocols changes.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ironbridge_trader.domain.models import (
    Bar,
    Decision,
    Fill,
    Order,
    Position,
    QuantSignal,
)


@runtime_checkable
class MarketDataSource(Protocol):
    """Anything that can hand back recent bars for a symbol."""

    def get_bars(self, symbol: str, lookback: int) -> list[Bar]:
        """Return up to `lookback` most recent bars, oldest first."""
        ...


@runtime_checkable
class Predictor(Protocol):
    """Anything that can turn bar history into a QuantSignal."""

    def predict(self, symbol: str, bars: list[Bar]) -> QuantSignal: ...


@runtime_checkable
class DecisionMaker(Protocol):
    """Anything that can turn (bars, quant signal, position) into a final Decision.
    ThresholdDecisionService and ClaudeTradingAgent both satisfy this with
    zero shared inheritance.
    """

    def decide(
        self,
        symbol: str,
        bars: list[Bar],
        signal: QuantSignal,
        position: Position,
    ) -> Decision: ...


@runtime_checkable
class ExecutionClient(Protocol):
    """Anything that can turn an Order into a Fill."""

    def place_order(self, order: Order) -> Fill: ...

"""Counts how often risk/manager.py vetoes a trade the decision-maker
wanted, and why -- see scripts/risk_veto_report.py. Built to answer a
question raised by the multi-regime backtest results: whether
max_position_size, then a fixed unit cap, was binding often enough
against a dollar-risk-based position sizer to be contaminating those
results independent of whether the underlying signal has real edge.
It was (71% of AAPL's directional decisions, 0% of MSFT/SPY's) -- see
risk/manager.py, now max_position_fraction. Kept as a general-purpose
diagnostic: worth re-running after any change to risk/position-sizing
settings, not just that one past incident.

Captures engine/trading_engine.py's "order blocked" warning via its
structured log args (symbol, exception type name) rather than parsing
the formatted message text, so counting survives a wording change.
Filters on the exact message *template* (`record.msg`, before %
substitution), not just the logger name -- that logger also emits a
differently-shaped warning ("skipping %s this cycle: ...") for a
different reason (insufficient history), which must not be counted
as a risk veto.
"""

from __future__ import annotations

import logging
from collections import Counter
from collections.abc import Iterator
from contextlib import contextmanager

_ENGINE_LOGGER_NAME = "ironbridge_trader.engine.trading_engine"
_VETO_MESSAGE_TEMPLATE = "order for %s blocked: %s: %s"


class RiskVetoCounter(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.by_symbol_and_reason: Counter[tuple[str, str]] = Counter()

    def emit(self, record: logging.LogRecord) -> None:
        if record.name != _ENGINE_LOGGER_NAME or record.msg != _VETO_MESSAGE_TEMPLATE:
            return
        symbol, exc_type, _ = record.args
        self.by_symbol_and_reason[(str(symbol), str(exc_type))] += 1

    @property
    def total(self) -> int:
        return sum(self.by_symbol_and_reason.values())

    def by_reason(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for (_, reason), n in self.by_symbol_and_reason.items():
            counts[reason] += n
        return counts

    def by_symbol(self) -> Counter[str]:
        counts: Counter[str] = Counter()
        for (symbol, _), n in self.by_symbol_and_reason.items():
            counts[symbol] += n
        return counts


@contextmanager
def capture_risk_vetoes() -> Iterator[RiskVetoCounter]:
    """Attach a RiskVetoCounter to the engine's logger for the duration
    of the `with` block, detaching it again afterward -- so a caller
    can wrap exactly one backtest/paper-trading run's worth of engine
    activity without leaking a handler onto the logger permanently.
    """
    counter = RiskVetoCounter()
    engine_logger = logging.getLogger(_ENGINE_LOGGER_NAME)
    engine_logger.addHandler(counter)
    try:
        yield counter
    finally:
        engine_logger.removeHandler(counter)

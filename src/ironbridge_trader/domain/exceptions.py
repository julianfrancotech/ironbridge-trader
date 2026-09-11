"""Exception hierarchy. A common root lets orchestration code do one broad
`except TraderError` (log + move to the next symbol) while callers closer
to the source can still catch a specific subclass -- same shape as
ironbridge-capital's CQGError hierarchy before it.
"""

from decimal import Decimal


class TraderError(Exception):
    """Base class for every error raised by this app. Never raised directly."""


class MarketDataError(TraderError):
    """Raised when bar data can't be fetched for a symbol."""


class AgentError(TraderError):
    """Raised when the decision-making service fails to produce a usable Decision
    (e.g. the LLM never calls submit_decision within the iteration budget).
    """


class BrokerError(TraderError):
    """Raised when the paper broker can't fill an order."""


class InsufficientMarginError(TraderError):
    def __init__(self, order_id: str, required: Decimal, available: Decimal) -> None:
        self.order_id = order_id
        self.required = required
        self.available = available
        super().__init__(f"Order {order_id} needs {required} margin, only {available} available")


class PositionLimitExceededError(TraderError):
    """Notional-value based, not a unit count -- see risk/manager.py's
    docstring for why a fixed unit cap doesn't make sense next to a
    dollar-risk-based position sizer.
    """

    def __init__(self, symbol: str, projected_notional: Decimal, max_notional: Decimal) -> None:
        self.symbol = symbol
        self.projected_notional = projected_notional
        self.max_notional = max_notional
        super().__init__(
            f"Order on {symbol} would move position notional to {projected_notional:.2f}, "
            f"limit is {max_notional:.2f}"
        )

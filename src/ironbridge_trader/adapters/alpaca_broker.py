"""Execution adapter: Alpaca's paper-trading endpoint as an alternative
to adapters/paper_broker.py. Satisfies protocols.ExecutionClient --
nothing upstream (engine, risk, services) needs to know which broker
it got. Paper endpoint only (TradingClient(paper=True)): still no real
money, matching every other design decision in this app.

Idempotency: `order.order_id` (a UUID -- see engine/trading_engine.py)
is sent as Alpaca's client_order_id. Alpaca deduplicates submissions on
that field server-side, so a retried submission -- the exact failure
mode retry_with_backoff would otherwise turn dangerous -- can't
double-place the same order.

No persistent connection / no state machine. Alpaca's trade-updates
WebSocket stream is the closer analog to a CQG-style persistent
session, and was deliberately not used: this app places at most a
handful of real orders per daily cycle regardless of watchlist size
(order count tracks how often the model clears the confidence floor,
not symbols watched), so a connection that mostly sits idle doesn't
earn a lifecycle to manage. See docs/platform-boundaries.md.

Per-order latency, not a cross-order "bulk reconcile": place_order
still has to return one Fill per call, synchronously, because that's
protocols.ExecutionClient's contract -- shared with PaperBroker, and
called independently from inside each symbol's own concurrent task
(engine/trading_engine.py's run_cycle, Phase B). Batching resolution
*across* orders would mean splitting that contract into a submit phase
and a separate reconcile phase called once after every symbol's task
finishes -- a bigger, protocol-wide change for a benefit Phase B's
concurrency already delivers: each order's (short, bounded) wait for a
terminal status already runs in parallel with every other symbol's,
not sequentially. In practice this rarely matters at all -- Alpaca's
paper engine fills a market order during market hours essentially
synchronously, so submit_order's own response is usually already
terminal and place_order makes exactly one API call.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC
from decimal import Decimal

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus
from alpaca.trading.requests import MarketOrderRequest

from ironbridge_trader.domain.exceptions import BrokerError
from ironbridge_trader.domain.models import Fill, Order, Side
from ironbridge_trader.resilience import retry_with_backoff

logger = logging.getLogger(__name__)

_TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

# Bounded fallback wait for the rare case submit_order doesn't already
# return a terminal status (e.g. after-hours, or a brief processing
# delay) -- short and fixed, not open-ended polling.
_POLL_ATTEMPTS = 5
_POLL_INTERVAL_SECONDS = 1.0

_TERMINAL_BAD_STATUSES = {
    AlpacaOrderStatus.REJECTED, AlpacaOrderStatus.CANCELED,
    AlpacaOrderStatus.EXPIRED, AlpacaOrderStatus.SUSPENDED,
}


class _TransientAlpacaError(Exception):
    """Marker used only to route an Alpaca APIError through retry_with_backoff --
    never raised or caught outside this module.
    """


def _reraise_for_retry(exc: APIError) -> None:
    if exc.status_code in _TRANSIENT_STATUS_CODES:
        raise _TransientAlpacaError(str(exc)) from exc
    raise exc


def _is_crypto(symbol: str) -> bool:
    return "-" in symbol  # this app's watchlist convention, e.g. "BTC-USD"


def _alpaca_symbol(symbol: str) -> str:
    return symbol.replace("-", "/") if _is_crypto(symbol) else symbol


def _fill_from_alpaca_order(order_id: str, symbol: str, side: Side, alpaca_order) -> Fill:
    # filled_avg_price/filled_qty come back as str | float | None -- routed
    # through str() first so a float never gets fed straight to Decimal()
    # and picks up binary-float noise, same convention as the rest of the app.
    # commission=0 explicitly, not just the Fill default: this is a real
    # fill, already reflecting whatever the market and Alpaca actually
    # charged (nothing, for equities/crypto) -- unlike PaperBroker, there
    # is no separate cost model here to apply.
    return Fill(
        order_id=order_id, symbol=symbol, side=side,
        price=Decimal(str(alpaca_order.filled_avg_price)),
        quantity=int(float(alpaca_order.filled_qty)),
        timestamp=alpaca_order.filled_at.astimezone(UTC),
        commission=Decimal(0),
    )


class AlpacaBroker:
    def __init__(self, api_key: str, secret_key: str) -> None:
        self._client = TradingClient(api_key, secret_key, paper=True)

    def place_order(self, order: Order) -> Fill:
        # protocols.ExecutionClient's implicit contract (PaperBroker
        # satisfies it trivially by never raising) is that failures come
        # out as TraderError -- that's what lets TradingEngine._maybe_execute
        # catch-and-skip a blocked order without aborting the whole cycle,
        # even when this call is running inside one of Phase B's concurrent
        # per-symbol tasks. So every non-TraderError this method's Alpaca
        # calls can raise gets translated to BrokerError here, not left to
        # escape as APIError/ConnectionError/TimeoutError.
        try:
            alpaca_order = self._submit(order)
        except (APIError, _TransientAlpacaError, ConnectionError, TimeoutError) as exc:
            raise BrokerError(f"Alpaca rejected order {order.order_id}: {exc}") from exc

        if alpaca_order.status == AlpacaOrderStatus.FILLED:
            return _fill_from_alpaca_order(order.order_id, order.symbol, order.side, alpaca_order)
        if alpaca_order.status in _TERMINAL_BAD_STATUSES:
            raise BrokerError(f"Alpaca order {order.order_id} ended in status {alpaca_order.status}")

        try:
            alpaca_order = self._wait_for_terminal_status(order)
        except (APIError, ConnectionError, TimeoutError) as exc:
            raise BrokerError(f"Alpaca order {order.order_id}: error checking status: {exc}") from exc

        if alpaca_order.status == AlpacaOrderStatus.FILLED:
            return _fill_from_alpaca_order(order.order_id, order.symbol, order.side, alpaca_order)
        raise BrokerError(
            f"Alpaca order {order.order_id} still {alpaca_order.status} after "
            f"{_POLL_ATTEMPTS * _POLL_INTERVAL_SECONDS:.0f}s -- check Alpaca's own dashboard"
        )

    @retry_with_backoff(
        max_attempts=3, backoff_seconds=1.0,
        retry_on=(_TransientAlpacaError, ConnectionError, TimeoutError),
    )
    def _submit(self, order: Order):
        request = MarketOrderRequest(
            symbol=_alpaca_symbol(order.symbol),
            qty=order.quantity,
            side=OrderSide.BUY if order.side is Side.BUY else OrderSide.SELL,
            # Alpaca crypto orders don't support DAY -- GTC is the closest
            # equivalent to "let it work until filled or explicitly canceled".
            time_in_force=TimeInForce.GTC if _is_crypto(order.symbol) else TimeInForce.DAY,
            client_order_id=order.order_id,
        )
        try:
            return self._client.submit_order(request)
        except APIError as exc:
            _reraise_for_retry(exc)
            raise  # unreachable -- _reraise_for_retry always raises

    def _wait_for_terminal_status(self, order: Order):
        alpaca_order = None
        for _ in range(_POLL_ATTEMPTS):
            time.sleep(_POLL_INTERVAL_SECONDS)
            alpaca_order = self._client.get_order_by_client_id(order.order_id)
            if alpaca_order.status == AlpacaOrderStatus.FILLED or alpaca_order.status in _TERMINAL_BAD_STATUSES:
                return alpaca_order
        logger.warning(
            "%s: still %s after %d checks", order.order_id, alpaca_order.status, _POLL_ATTEMPTS,
        )
        return alpaca_order

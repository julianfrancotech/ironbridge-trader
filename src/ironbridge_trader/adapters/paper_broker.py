"""Execution adapter: simulated broker, fills every order immediately at
its reference price (the bar close it was decided on) adjusted for
transaction costs -- no partial fills, no rejections; this app's realism
budget goes into the prediction and decision layers, and now the cost
model, not a full fill simulator. Satisfies protocols.ExecutionClient.

Cost model: `transaction_cost_bps` blends spread and slippage into one
number, applied against the trader (a BUY fills higher, a SELL fills
lower, than the naive reference price) -- bar data has no bid/ask to
model them as separate line items, so pretending to would be fake
precision. `commission_per_trade` is a separate flat fee, matching how
a real brokerage statement shows execution price and commission as two
different numbers. Without this, every backtest/paper number implicitly
assumed zero-friction trading, which real trading never gets -- see
config.py's Settings.transaction_cost_bps docstring for the reasoning
behind the defaults.

This is the adapter that would be swapped for a real broker (or an
exchange-connectivity API like CQG) later: write one class with a
place_order(order) -> Fill method that talks to a real API, and nothing
upstream (engine, risk, services) needs to change or even know about it
-- see adapters/alpaca_broker.py, which needs none of this cost model
since its fills are real.
"""

from __future__ import annotations

from decimal import Decimal

from ironbridge_trader.domain.models import Fill, Order, Side


class PaperBroker:
    def __init__(self, transaction_cost_bps: Decimal, commission_per_trade: Decimal) -> None:
        self._cost_fraction = transaction_cost_bps / Decimal(10_000)
        self._commission_per_trade = commission_per_trade

    def place_order(self, order: Order) -> Fill:
        haircut = 1 + self._cost_fraction if order.side is Side.BUY else 1 - self._cost_fraction
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=order.reference_price * haircut,
            quantity=order.quantity,
            timestamp=order.as_of,
            commission=self._commission_per_trade,
        )

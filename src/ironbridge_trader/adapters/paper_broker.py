"""Execution adapter: simulated broker, fills every order immediately at
its reference price (the bar close it was decided on). No slippage
model, no partial fills, no rejections -- this app's realism budget goes
into the prediction and decision layers, not the fill simulator.
Satisfies protocols.ExecutionClient.

This is the adapter that would be swapped for a real broker (or an
exchange-connectivity API like CQG) later: write one class with a
place_order(order) -> Fill method that talks to a real API, and nothing
upstream (engine, risk, services) needs to change or even know about it.
"""

from __future__ import annotations

from ironbridge_trader.domain.models import Fill, Order


class PaperBroker:
    def place_order(self, order: Order) -> Fill:
        return Fill(
            order_id=order.order_id,
            symbol=order.symbol,
            side=order.side,
            price=order.reference_price,
            quantity=order.quantity,
            timestamp=order.as_of,
        )

"""Fixed-fractional position sizing: how many whole units to buy or sell,
derived from account risk tolerance -- not a fixed quantity that's the
same regardless of price or account size. A flat `order_quantity = 10`
means the same 10 units whether the reference price is a $10 stock or
$110,000 of Bitcoin, i.e. wildly different dollar risk per trade. This
computes size from the actual risk budget instead:

    size = (equity * risk_fraction) / (reference_price * stop_loss_fraction)

risk_fraction is the fraction of account equity you're willing to lose
on this one trade if it goes wrong. stop_loss_fraction is the assumed
adverse-move distance used ONLY for this sizing calculation -- this app
does not place a real stop-loss order (PaperBroker only fills market
orders at the reference price), so stop_loss_fraction is a risk-budget
assumption, not a live protective order. Treat it as "if this moved
against me by X% before I noticed, how much am I willing to have lost."

Deliberately does NOT take the decision's own confidence score as an
input. That confidence comes from the same model whose overall
walk-forward accuracy is close to a coin flip (see README's Trading
theory section) -- scaling size directly off an unvalidated confidence
number is exactly the full-Kelly mistake: it treats a possibly
miscalibrated probability as trustworthy enough to bet big on.
risk_fraction and stop_loss_fraction are hand-set account-risk policy,
not model output, and stay constant regardless of how confident any
one decision claims to be.

A note on what this deliberately does NOT solve: for a high-priced
asset relative to the risk budget (e.g. one whole BTC-USD unit against
a modest account and a conservative risk_fraction), the formula can
legitimately compute zero -- you cannot afford even one whole unit
without exceeding your risk budget. That is not a bug to round away;
it is the correct, honest answer for an account this size, and the
caller should treat size() == 0 as "no trade," not as "buy 1 anyway."
Supporting fractional units (real crypto brokers do) would fix this,
but Order/Position model whole-unit quantities throughout this app --
out of scope for this change.
"""

from __future__ import annotations

from decimal import Decimal


class PositionSizer:
    def __init__(self, risk_fraction: Decimal, stop_loss_fraction: Decimal) -> None:
        self._risk_fraction = risk_fraction
        self._stop_loss_fraction = stop_loss_fraction

    def size(self, reference_price: Decimal, available_equity: Decimal) -> int:
        if reference_price <= 0 or self._stop_loss_fraction <= 0:
            return 0
        risk_amount = available_equity * self._risk_fraction
        risk_per_unit = reference_price * self._stop_loss_fraction
        return int(risk_amount / risk_per_unit)  # truncates toward zero -- whole units only

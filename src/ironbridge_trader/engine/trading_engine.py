"""The per-symbol decision loop: for one symbol, predict -> decide ->
risk-check -> execute -> persist. Built from collaborators passed into
__init__ -- pure dependency injection. This is the reusable, testable
core; it does not construct its own collaborators or know where they
came from -- that wiring lives one layer up, in orchestration/, which
plays the role a production service's orchestration workflow files
play: deciding *which* concrete adapters/services to use for a given
run and invoking the business logic with them.

Four of the collaborators are protocols (market_data, predictor,
decision_maker, execution) -- genuine swap points: a live feed instead
of stored bars, a different model, the threshold service instead of
Claude, a real broker instead of the paper one. `risk`, `position_sizer`,
and `db` are concrete, not protocols; each has exactly one
implementation and there is deliberately no second one to swap in.

Position is deliberately NOT engine state. This loop is meant to be
invoked once per bar (e.g. once a day, via cron) rather than kept
running in a long-lived process, so each call rebuilds the current
position by replaying that symbol's stored fills -- the fills table is
the single source of truth, so there's nothing to go stale between runs.

Idempotency: before doing any work for a symbol, _run_symbol checks
whether a decision already exists for that symbol's latest bar
(db.has_decision_for). A retried or duplicated cron trigger -- the
realistic failure mode for a scheduled job -- is a no-op, not a second
decision, a second (possibly LLM-backed, possibly costly) API call, or
a duplicate row the dashboard would show as two decisions on one day.
storage/db.py's UNIQUE(symbol, ts) constraint on `decisions` is the
backstop underneath this check, not a replacement for it.

Concurrency: run_cycle processes symbols with a bounded thread pool
(concurrency.run_bounded), one task per symbol, never two tasks for
the same symbol. That constraint is what keeps this safe without any
new locking: each symbol's has_decision_for -> decide -> insert_decision
sequence stays fully contained inside its own task, so no cross-symbol
state is ever touched from two threads at once -- the only thing that
changes versus a plain for-loop is that *different* symbols' work now
interleaves instead of running strictly back-to-back. `Order.order_id`
is a UUID rather than a shared counter for the same reason: nothing to
coordinate across threads, and no cross-run collision risk once this
ID is handed to a real broker as an idempotency key (see Phase C).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from decimal import Decimal

from ironbridge_trader.concurrency import run_bounded
from ironbridge_trader.config import Settings
from ironbridge_trader.domain.exceptions import TraderError
from ironbridge_trader.domain.models import (
    Action,
    Decision,
    Fill,
    Order,
    Position,
    Side,
)
from ironbridge_trader.protocols import (
    DecisionMaker,
    ExecutionClient,
    MarketDataSource,
    Predictor,
)
from ironbridge_trader.risk.manager import RiskManager
from ironbridge_trader.risk.position_sizer import PositionSizer
from ironbridge_trader.storage.db import Database

logger = logging.getLogger(__name__)


class TradingEngine:
    def __init__(
        self,
        market_data: MarketDataSource,
        predictor: Predictor,
        decision_maker: DecisionMaker,
        execution: ExecutionClient,
        risk: RiskManager,
        position_sizer: PositionSizer,
        db: Database,
        settings: Settings,
    ) -> None:
        self._market_data = market_data
        self._predictor = predictor
        self._decision_maker = decision_maker
        self._execution = execution
        self._risk = risk
        self._position_sizer = position_sizer
        self._db = db
        self._settings = settings

    def run_cycle(self, symbols: list[str]) -> list[Decision]:
        """Evaluate every symbol, at most settings.max_concurrent_symbols
        at a time. Returns every NEW Decision made (including HOLDs). A
        symbol already decided for its latest bar is silently idempotent
        (INFO log, nothing appended) -- that's expected behavior for a
        retried run, not an error. An actual failure (e.g. not enough
        history yet) is logged as a WARNING and skipped rather than
        aborting the whole cycle.
        """
        futures = run_bounded(symbols, self._run_symbol, self._settings.max_concurrent_symbols)
        decisions: list[Decision] = []
        for symbol, future in zip(symbols, futures, strict=True):
            try:
                decision = future.result()
            except TraderError as exc:
                logger.warning("skipping %s this cycle: %s: %s", symbol, type(exc).__name__, exc)
                continue
            if decision is not None:
                decisions.append(decision)
        return decisions

    def _run_symbol(self, symbol: str) -> Decision | None:
        bars = self._market_data.get_bars(symbol, self._settings.feature_lookback + 40)
        if len(bars) < 30:
            raise TraderError(f"not enough stored bars for {symbol}; run scripts/fetch_data.py")

        latest_bar_time = bars[-1].timestamp
        if self._db.has_decision_for(symbol, latest_bar_time):
            logger.info(
                "%s already has a decision for %s -- cycle is idempotent, skipping",
                symbol, latest_bar_time.date(),
            )
            return None

        position = self._current_position(symbol)
        signal = self._predictor.predict(symbol, bars)
        decision = self._decision_maker.decide(symbol, bars, signal, position)
        self._db.insert_decision(decision)

        self._maybe_execute(decision, position, bars[-1].close)
        return decision

    def _current_position(self, symbol: str) -> Position:
        """Rebuild the running position by replaying every stored fill in order."""
        position = Position(symbol=symbol)
        for row in self._db.all_fills(symbol):
            fill = Fill(
                order_id=row["order_id"], symbol=row["symbol"], side=Side[row["side"]],
                price=Decimal(str(row["price"])), quantity=row["quantity"],
                timestamp=datetime.fromisoformat(row["ts"]),
            )
            position.apply_fill(fill)
        return position

    def _maybe_execute(self, decision: Decision, position: Position, reference_price: Decimal) -> None:
        desired_side = {Action.BUY: Side.BUY, Action.SELL: Side.SELL}.get(decision.action)
        current_side = (
            Side.BUY if position.quantity > 0 else Side.SELL if position.quantity < 0 else None
        )
        if desired_side is None or desired_side == current_side:
            return  # HOLD, or a signal that just repeats our current position -- no new order

        quantity = self._position_sizer.size(reference_price, self._settings.account_equity)
        if quantity <= 0:
            # Honest outcome, not a bug: the risk budget can't afford even
            # one whole unit at this price (see PositionSizer's docstring).
            # Rounding up to 1 anyway would silently take on more risk
            # than risk_fraction says you're willing to.
            logger.info(
                "%s: risk-sized quantity is 0 at reference price %s -- no trade",
                decision.symbol, reference_price,
            )
            return

        order = Order.market_order(
            order_id=f"{decision.symbol}-{uuid.uuid4().hex[:10]}",
            symbol=decision.symbol, side=desired_side,
            quantity=quantity, reference_price=reference_price,
            as_of=decision.as_of,
        )
        try:
            self._risk.check_order(order, position, self._settings.account_equity)
            fill = self._execution.place_order(order)
        except TraderError as exc:
            logger.warning("order for %s blocked: %s: %s", decision.symbol, type(exc).__name__, exc)
            return

        self._db.insert_fill(fill)
        position.apply_fill(fill)
        self._db.record_equity_point(
            fill.timestamp, decision.symbol, position.quantity,
            position.average_price * abs(position.quantity),
        )
        logger.info(
            "%s: %s %d filled at %s, position now %s @ avg %s",
            decision.symbol, desired_side.name, fill.quantity, fill.price,
            position.quantity, position.average_price,
        )

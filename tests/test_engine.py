import threading
from decimal import Decimal

from conftest import make_bars

from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Action, Decision, Fill, Order, QuantSignal
from ironbridge_trader.engine.trading_engine import TradingEngine
from ironbridge_trader.risk.manager import RiskManager
from ironbridge_trader.risk.position_sizer import PositionSizer
from ironbridge_trader.storage.db import Database


class StubPredictor:
    def predict(self, symbol, bars):
        return QuantSignal(
            symbol=symbol, as_of=bars[-1].timestamp, prob_up=0.9,
            model_version="stub", features={},
        )


class AlwaysBuyDecisionMaker:
    def decide(self, symbol, bars, signal, position):
        return Decision(
            symbol=symbol, as_of=signal.as_of, action=Action.BUY, confidence=0.9,
            rationale="stub always buys", quant_signal=signal, agent_kind="stub",
        )


class RecordingBroker:
    def __init__(self):
        self.orders: list[Order] = []
        self._lock = threading.Lock()

    def place_order(self, order: Order) -> Fill:
        with self._lock:
            self.orders.append(order)
        return Fill(
            order_id=order.order_id, symbol=order.symbol, side=order.side,
            price=order.reference_price, quantity=order.quantity, timestamp=order.as_of,
        )


def make_settings(**overrides) -> Settings:
    # risk_fraction/stop_loss_fraction chosen so a ~$100 reference price
    # (conftest's default bar fixture) sizes to a few dozen units --
    # comfortably under max_position_size=100 for tests that expect a
    # BUY to succeed, without hardcoding an exact quantity anywhere here.
    return Settings(**{
        "risk_fraction": Decimal("0.002"), "stop_loss_fraction": Decimal("0.05"),
        "max_position_size": 100, **overrides,
    })


def make_engine(db: Database, settings: Settings, *, decision_maker=None, execution=None) -> TradingEngine:
    return TradingEngine(
        market_data=db, predictor=StubPredictor(),
        decision_maker=decision_maker or AlwaysBuyDecisionMaker(),
        execution=execution or RecordingBroker(),
        risk=RiskManager(settings.max_position_size, settings.margin_rate),
        position_sizer=PositionSizer(settings.risk_fraction, settings.stop_loss_fraction),
        db=db, settings=settings,
    )


def test_run_cycle_buys_and_persists_fill(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)
    broker = RecordingBroker()
    settings = make_settings()
    expected_qty = PositionSizer(settings.risk_fraction, settings.stop_loss_fraction).size(
        bars[-1].close, settings.account_equity
    )

    engine = make_engine(db, settings, execution=broker)
    decisions = engine.run_cycle(["TEST"])

    assert len(decisions) == 1
    assert decisions[0].action is Action.BUY
    assert len(broker.orders) == 1
    fills = db.all_fills("TEST")
    assert len(fills) == 1
    assert fills[0]["quantity"] == expected_qty > 0
    stored_decisions = db.all_decisions("TEST")
    assert len(stored_decisions) == 1


def test_run_cycle_sizes_zero_quantity_as_no_trade(tmp_path, bars):
    # An extremely tight risk budget means even one unit can't be
    # afforded -- this must be a skipped trade, not a rounded-up-to-1 one.
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)
    broker = RecordingBroker()
    settings = make_settings(risk_fraction=Decimal("0.0000001"))

    engine = make_engine(db, settings, execution=broker)
    decisions = engine.run_cycle(["TEST"])

    assert len(decisions) == 1
    assert decisions[0].action is Action.BUY  # the decision itself still happened
    assert len(broker.orders) == 0  # but sizing to 0 means no order was placed
    assert db.all_fills("TEST") == []


def test_run_cycle_does_not_reorder_when_already_in_position(tmp_path, bars):
    # Two DIFFERENT bars (so the idempotency check doesn't short-circuit
    # the second cycle) both produce a BUY -- this exercises the engine's
    # own "repeat signal while already in that position" no-op, not idempotency.
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars[:-1])
    broker = RecordingBroker()
    settings = make_settings()
    engine = make_engine(db, settings, execution=broker)

    engine.run_cycle(["TEST"])
    db.upsert_bars(bars[-1:])  # advance one new bar -- still BUY, but we're already long
    engine.run_cycle(["TEST"])

    assert len(broker.orders) == 1


def test_run_cycle_is_idempotent_for_an_already_decided_bar(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)
    broker = RecordingBroker()
    settings = make_settings()
    engine = make_engine(db, settings, execution=broker)

    first = engine.run_cycle(["TEST"])
    second = engine.run_cycle(["TEST"])  # same bars -- a retried/duplicated trigger

    assert len(first) == 1
    assert second == []  # no new Decision, not even a repeated one
    assert len(broker.orders) == 1
    assert len(db.all_decisions("TEST")) == 1  # no duplicate row either


def test_run_cycle_skips_symbol_with_insufficient_history(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars[:10])  # well under the minimum feature lookback
    settings = make_settings()
    engine = make_engine(db, settings)

    decisions = engine.run_cycle(["TEST"])

    assert decisions == []


def test_risk_manager_blocks_order_that_exceeds_position_limit(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)
    broker = RecordingBroker()
    settings = make_settings(max_position_size=1)  # sized quantity will always breach this
    engine = make_engine(db, settings, execution=broker)

    engine.run_cycle(["TEST"])

    assert len(broker.orders) == 0
    assert db.all_fills("TEST") == []


def test_run_cycle_processes_every_symbol_exactly_once_when_run_concurrently(tmp_path):
    db = Database(tmp_path / "t.db")
    symbols = ["A", "B", "C", "D"]
    for symbol in symbols:
        db.upsert_bars(make_bars(symbol=symbol))
    broker = RecordingBroker()
    settings = make_settings(max_concurrent_symbols=2)  # pool smaller than symbol count
    engine = make_engine(db, settings, execution=broker)

    decisions = engine.run_cycle(symbols)

    assert {d.symbol for d in decisions} == set(symbols)
    assert len(decisions) == len(symbols)  # no duplicates, none lost
    for symbol in symbols:
        assert len(db.all_decisions(symbol)) == 1
        assert len(db.all_fills(symbol)) == 1


def test_order_ids_never_collide_across_separate_engine_instances(tmp_path, bars):
    # Regression test for the bug the UUID-based order_id fixes: the old
    # self._order_seq counter reset to 0 every process run, so two
    # separate TradingEngine instances (simulating two different days'
    # cron invocations) on the same symbol would both mint "TEST-1".
    broker_day_one = RecordingBroker()
    db_day_one = Database(tmp_path / "day_one.db")
    db_day_one.upsert_bars(bars)
    make_engine(db_day_one, make_settings(), execution=broker_day_one).run_cycle(["TEST"])

    broker_day_two = RecordingBroker()
    db_day_two = Database(tmp_path / "day_two.db")
    db_day_two.upsert_bars(bars)
    make_engine(db_day_two, make_settings(), execution=broker_day_two).run_cycle(["TEST"])

    order_id_day_one = broker_day_one.orders[0].order_id
    order_id_day_two = broker_day_two.orders[0].order_id
    assert order_id_day_one != order_id_day_two
    assert order_id_day_one != "TEST-1"  # the exact collision the old counter produced
    assert order_id_day_two != "TEST-1"

from datetime import UTC, datetime
from decimal import Decimal

from ironbridge_trader.domain.models import Fill, Side
from ironbridge_trader.storage.db import Database


def test_latest_bar_ts_is_none_for_a_symbol_never_fetched(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.latest_bar_ts("TEST") is None


def test_latest_bar_ts_returns_the_most_recent_stored_bar(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)

    assert db.latest_bar_ts("TEST") == bars[-1].timestamp


def test_fills_lookup_by_symbol_uses_an_index_not_a_full_scan(tmp_path):
    db = Database(tmp_path / "t.db")
    with db._connect() as conn:
        plan = conn.execute("EXPLAIN QUERY PLAN SELECT * FROM fills WHERE symbol = ?", ("A",)).fetchall()
    plan_text = " ".join(str(row) for row in plan)

    assert "SCAN" not in plan_text or "USING INDEX" in plan_text


def test_equity_curve_lookup_by_symbol_uses_an_index_not_a_full_scan(tmp_path):
    db = Database(tmp_path / "t.db")
    with db._connect() as conn:
        plan = conn.execute(
            "EXPLAIN QUERY PLAN SELECT * FROM equity_curve WHERE symbol = ?", ("A",)
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan)

    assert "SCAN" not in plan_text or "USING INDEX" in plan_text


def test_all_fills_by_symbol_still_returns_only_that_symbols_rows(tmp_path):
    db = Database(tmp_path / "t.db")
    ts = datetime(2024, 1, 1, tzinfo=UTC)
    db.insert_fill(Fill(order_id="o1", symbol="A", side=Side.BUY, price=Decimal(10), quantity=1, timestamp=ts))
    db.insert_fill(Fill(order_id="o2", symbol="B", side=Side.BUY, price=Decimal(20), quantity=2, timestamp=ts))

    fills = db.all_fills("A")

    assert len(fills) == 1
    assert fills[0]["symbol"] == "A"

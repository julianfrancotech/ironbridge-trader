import sqlite3
from datetime import UTC, datetime
from decimal import Decimal

from ironbridge_trader.domain.models import Fill, Side
from ironbridge_trader.storage.db import Database


def test_latest_bar_is_none_for_a_symbol_never_fetched(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.latest_bar("TEST") is None


def test_latest_bar_returns_the_most_recent_stored_bar(tmp_path, bars):
    db = Database(tmp_path / "t.db")
    db.upsert_bars(bars)

    latest = db.latest_bar("TEST")

    assert latest.timestamp == bars[-1].timestamp
    assert latest.close == bars[-1].close


def test_heartbeat_is_none_for_a_source_never_recorded(tmp_path):
    db = Database(tmp_path / "t.db")
    assert db.latest_heartbeat("fetch_data") is None


def test_heartbeat_round_trips_and_keeps_only_the_latest_per_source(tmp_path):
    db = Database(tmp_path / "t.db")
    db.record_heartbeat("fetch_data", datetime(2024, 1, 1, tzinfo=UTC), "ok", "4 symbol(s)")
    db.record_heartbeat("fetch_data", datetime(2024, 1, 2, tzinfo=UTC), "error", "boom")
    db.record_heartbeat("run_paper_trader", datetime(2024, 1, 2, tzinfo=UTC), "ok", "1 decision")

    hb = db.latest_heartbeat("fetch_data")

    assert hb["status"] == "error"
    assert hb["detail"] == "boom"
    assert db.latest_heartbeat("run_paper_trader")["status"] == "ok"


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


def test_opening_a_pre_commission_column_db_migrates_it_in_place(tmp_path):
    # Simulates a database created before the commission column existed --
    # CREATE TABLE IF NOT EXISTS alone can't add a column to a table that
    # already exists on disk, which is exactly what Database.__init__
    # must handle for anyone with an existing data/ironbridge_trader.db.
    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE fills (id INTEGER PRIMARY KEY AUTOINCREMENT, order_id TEXT NOT NULL, "
        "symbol TEXT NOT NULL, side TEXT NOT NULL, price REAL NOT NULL, "
        "quantity INTEGER NOT NULL, ts TEXT NOT NULL)"
    )
    conn.commit()
    conn.close()

    db = Database(path)  # must not raise

    with db._connect() as new_conn:
        columns = {row[1] for row in new_conn.execute("PRAGMA table_info(fills)")}
    assert "commission" in columns

    db.insert_fill(
        Fill(
            order_id="o1", symbol="A", side=Side.BUY, price=Decimal(10), quantity=1,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC), commission=Decimal("0.50"),
        )
    )
    assert db.all_fills()[0]["commission"] == 0.5

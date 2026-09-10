import threading
import time

from conftest import make_bars

from ironbridge_trader.config import Settings
from ironbridge_trader.orchestration import fetch_market_data
from ironbridge_trader.storage.db import Database


def test_run_fetches_full_window_for_a_symbol_never_seen_before(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    calls: list[dict] = []

    def fake_fetch_bars(symbol, years, interval, start=None):
        calls.append({"symbol": symbol, "start": start})
        return make_bars(symbol=symbol, n=3)

    monkeypatch.setattr(fetch_market_data, "fetch_bars", fake_fetch_bars)
    settings = Settings(symbols=("A",))

    results = fetch_market_data.run(db, settings)

    assert calls == [{"symbol": "A", "start": None}]
    assert results[0].bar_count == 3
    assert len(db.get_bars("A")) == 3


def test_run_requests_only_bars_since_the_last_stored_one(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    existing = make_bars(symbol="A", n=5)
    db.upsert_bars(existing)
    calls: list[dict] = []

    def fake_fetch_bars(symbol, years, interval, start=None):
        calls.append({"symbol": symbol, "start": start})
        return []  # nothing new since `start` -- the realistic same-day-rerun case

    monkeypatch.setattr(fetch_market_data, "fetch_bars", fake_fetch_bars)
    settings = Settings(symbols=("A",))

    results = fetch_market_data.run(db, settings)

    assert calls == [{"symbol": "A", "start": existing[-1].timestamp}]
    assert results[0].bar_count == 0
    assert results[0].last_date == existing[-1].timestamp.date()


def test_run_processes_symbols_with_bounded_concurrency(tmp_path, monkeypatch):
    db = Database(tmp_path / "t.db")
    symbols = ("A", "B", "C", "D", "E", "F")
    lock = threading.Lock()
    state = {"active": 0, "max_active": 0}

    def fake_fetch_bars(symbol, years, interval, start=None):
        with lock:
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
        time.sleep(0.05)  # hold the "slot" long enough for overlap to be observable
        with lock:
            state["active"] -= 1
        return make_bars(symbol=symbol, n=2)

    monkeypatch.setattr(fetch_market_data, "fetch_bars", fake_fetch_bars)
    settings = Settings(symbols=symbols, max_concurrent_symbols=3)

    results = fetch_market_data.run(db, settings)

    assert {r.symbol for r in results} == set(symbols)
    assert len(results) == len(symbols)  # every symbol got exactly one result
    assert state["max_active"] > 1  # actually ran concurrently, not sequentially
    assert state["max_active"] <= 3  # but never exceeded the configured bound

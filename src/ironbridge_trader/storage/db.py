"""SQLite persistence. One small file, one connection class, plain SQL --
no ORM, no separate DAO/Repository split.

A larger production trading service might split persistence into three
layers (DAO -> Repository -> ORM Model over Postgres) because it has real
business rules that differ from raw storage shape, many tables, and
multiple call sites that need that abstraction. This app has a handful
of tables, at most one writer at a time, and every method here already
reads like the business question it answers (`recent_decisions`,
`equity_curve`) -- adding two more layers on top would be indirection
with no corresponding benefit. Every table maps to exactly one thing
the dashboard needs to show: price history, what the model predicted,
what the decision-maker decided and *why*, what actually got filled,
and how the model's accuracy has moved across retrains.

Two production-correctness details worth calling out explicitly:
WAL mode (set in _connect) lets the dashboard read while a decision
cycle writes, instead of one blocking the other; and `decisions` has a
UNIQUE(symbol, ts) constraint that backstops the idempotency check in
engine/trading_engine.py -- a re-run can never silently duplicate or
overwrite a bar that's already been decided.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from ironbridge_trader.domain.models import Bar, Decision, Fill

SCHEMA = """
CREATE TABLE IF NOT EXISTS bars (
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    open REAL NOT NULL, high REAL NOT NULL, low REAL NOT NULL,
    close REAL NOT NULL, volume INTEGER NOT NULL,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    trained_at TEXT NOT NULL,
    train_rows INTEGER NOT NULL,
    eval_rows INTEGER NOT NULL,
    eval_accuracy REAL NOT NULL,
    eval_auc REAL NOT NULL,
    feature_names TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    ts TEXT NOT NULL,
    action TEXT NOT NULL,
    confidence REAL NOT NULL,
    rationale TEXT NOT NULL,
    prob_up REAL NOT NULL,
    model_version TEXT NOT NULL,
    agent_kind TEXT NOT NULL,
    features_json TEXT NOT NULL,
    -- One decision per symbol per bar. This is what makes a decision
    -- cycle idempotent: re-running the same day's cycle (a retried cron
    -- job, a manual re-trigger) can't produce a second row for a bar
    -- that's already been decided -- see engine/trading_engine.py.
    UNIQUE (symbol, ts)
);

CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    price REAL NOT NULL,
    quantity INTEGER NOT NULL,
    ts TEXT NOT NULL,
    -- Separate from `price` on purpose -- see domain/models.py::Fill.
    -- Only a fresh CREATE gets this column here; see _migrate() below
    -- for databases that already exist on disk.
    commission REAL NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS equity_curve (
    ts TEXT NOT NULL,
    symbol TEXT NOT NULL,
    position_qty INTEGER NOT NULL,
    position_value REAL NOT NULL,
    PRIMARY KEY (ts, symbol)
);

-- One row per unattended cron script invocation (fetch_data.py,
-- run_paper_trader.py), success or failure. This is the only way to
-- notice "the job silently stopped running" -- a closed laptop, a
-- crashed process -- which is a different, more basic failure mode
-- than the model producing bad decisions, and one nothing else here
-- detects.
CREATE TABLE IF NOT EXISTS run_heartbeats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    ts TEXT NOT NULL,
    status TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_run_heartbeats_source ON run_heartbeats(source, ts);

-- fills/equity_curve are looked up by symbol far more often than by
-- ts alone (trading_engine.py's _current_position rebuild does it
-- once per symbol per cycle), but neither table's primary key leads
-- with symbol, so that lookup was a full table scan. These indexes
-- fix it without a destructive PK-reorder migration on databases that
-- already exist on disk.
CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills(symbol, ts);
CREATE INDEX IF NOT EXISTS idx_equity_curve_symbol ON equity_curve(symbol, ts);
"""


class Database:
    def __init__(self, path: Path) -> None:
        self._path = path
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            self._migrate(conn)

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Schema tweaks CREATE TABLE IF NOT EXISTS can't express -- SQLite
        has no ALTER TABLE ADD COLUMN IF NOT EXISTS, so check first. Runs
        on every connect; each check is one fast PRAGMA, negligible next
        to everything else __init__ already does.
        """
        fills_columns = {row[1] for row in conn.execute("PRAGMA table_info(fills)")}
        if "commission" not in fills_columns:
            conn.execute("ALTER TABLE fills ADD COLUMN commission REAL NOT NULL DEFAULT 0")

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        # timeout=10: if another connection briefly holds the write lock
        # (a decision cycle writing while the dashboard reads), wait up
        # to 10s instead of immediately raising "database is locked".
        # WAL mode additionally lets readers proceed concurrently with a
        # writer rather than blocking on it at all -- the combination is
        # what makes "cron job writes, dashboard reads" safe on SQLite
        # without needing a real database server for a single-writer app.
        conn = sqlite3.connect(self._path, timeout=10)
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # -- bars ---------------------------------------------------------
    def upsert_bars(self, bars: list[Bar]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO bars VALUES (?,?,?,?,?,?,?)",
                [
                    (b.symbol, b.timestamp.isoformat(), float(b.open), float(b.high),
                     float(b.low), float(b.close), b.volume)
                    for b in bars
                ],
            )

    def get_bars(self, symbol: str, lookback: int | None = None) -> list[Bar]:
        query = "SELECT symbol, ts, open, high, low, close, volume FROM bars WHERE symbol = ? ORDER BY ts"
        with self._connect() as conn:
            rows = conn.execute(query, (symbol,)).fetchall()
        bars = [
            Bar(
                symbol=r[0], timestamp=datetime.fromisoformat(r[1]),
                open=Decimal(str(r[2])), high=Decimal(str(r[3])), low=Decimal(str(r[4])),
                close=Decimal(str(r[5])), volume=r[6],
            )
            for r in rows
        ]
        return bars[-lookback:] if lookback else bars

    def symbols_with_bars(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT DISTINCT symbol FROM bars ORDER BY symbol").fetchall()
        return [r[0] for r in rows]

    def latest_bar(self, symbol: str) -> Bar | None:
        """Most recent stored bar for this symbol, or None if it has
        never been fetched. Two independent callers need this: ingestion
        asks for bars since this one's timestamp instead of redownloading
        the full history window every run (adapters/market_data.py), and
        data_quality.validate_bars needs its close to sanity-check the
        first newly-fetched bar's day-over-day move.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT symbol, ts, open, high, low, close, volume FROM bars "
                "WHERE symbol = ? ORDER BY ts DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return Bar(
            symbol=row[0], timestamp=datetime.fromisoformat(row[1]),
            open=Decimal(str(row[2])), high=Decimal(str(row[3])), low=Decimal(str(row[4])),
            close=Decimal(str(row[5])), volume=row[6],
        )

    # -- model versions -------------------------------------------------
    def insert_model_version(
        self, version: str, trained_at: datetime, train_rows: int, eval_rows: int,
        eval_accuracy: float, eval_auc: float, feature_names: list[str],
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_versions VALUES (?,?,?,?,?,?,?)",
                (version, trained_at.isoformat(), train_rows, eval_rows,
                 eval_accuracy, eval_auc, json.dumps(feature_names)),
            )

    def latest_model_version(self) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT version FROM model_versions ORDER BY trained_at DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

    def model_versions(self) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT version, trained_at, train_rows, eval_rows, eval_accuracy, eval_auc "
                "FROM model_versions ORDER BY trained_at"
            ).fetchall()
        cols = ["version", "trained_at", "train_rows", "eval_rows", "eval_accuracy", "eval_auc"]
        return [dict(zip(cols, r)) for r in rows]

    # -- decisions --------------------------------------------------------
    def insert_decision(self, decision: Decision) -> None:
        # OR IGNORE, not OR REPLACE: this is the backstop half of
        # idempotency. The engine already checks has_decision_for()
        # before doing the (possibly costly, possibly LLM-backed) work
        # of producing a Decision at all -- this just guarantees that
        # even a caller who skips that check can never silently
        # overwrite an existing decision for a bar that's already final.
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO decisions "
                "(symbol, ts, action, confidence, rationale, prob_up, model_version, agent_kind, features_json) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    decision.symbol, decision.as_of.isoformat(), decision.action.name,
                    decision.confidence, decision.rationale, decision.quant_signal.prob_up,
                    decision.quant_signal.model_version, decision.agent_kind,
                    json.dumps(decision.quant_signal.features),
                ),
            )

    def has_decision_for(self, symbol: str, as_of: datetime) -> bool:
        """True if a decision has already been recorded for this exact
        symbol + bar timestamp. The engine calls this before predicting
        or deciding at all, so a re-run of a cycle that's already run
        today skips the (possibly LLM-backed) work entirely, not just
        the database write.
        """
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM decisions WHERE symbol = ? AND ts = ? LIMIT 1",
                (symbol, as_of.isoformat()),
            ).fetchone()
        return row is not None

    def recent_decisions(self, symbol: str, limit: int = 5) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ts, action, confidence, rationale, prob_up FROM decisions "
                "WHERE symbol = ? ORDER BY ts DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
        cols = ["ts", "action", "confidence", "rationale", "prob_up"]
        return [dict(zip(cols, r)) for r in rows]

    def all_decisions(self, symbol: str | None = None) -> list[dict]:
        query = (
            "SELECT symbol, ts, action, confidence, rationale, prob_up, model_version, agent_kind "
            "FROM decisions"
        )
        params: tuple = ()
        if symbol:
            query += " WHERE symbol = ?"
            params = (symbol,)
        query += " ORDER BY ts"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        cols = ["symbol", "ts", "action", "confidence", "rationale", "prob_up", "model_version", "agent_kind"]
        return [dict(zip(cols, r)) for r in rows]

    # -- fills / equity -----------------------------------------------
    def insert_fill(self, fill: Fill) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO fills (order_id, symbol, side, price, quantity, ts, commission) "
                "VALUES (?,?,?,?,?,?,?)",
                (fill.order_id, fill.symbol, fill.side.name, float(fill.price),
                 fill.quantity, fill.timestamp.isoformat(), float(fill.commission)),
            )

    def all_fills(self, symbol: str | None = None) -> list[dict]:
        query = "SELECT order_id, symbol, side, price, quantity, ts, commission FROM fills"
        params: tuple = ()
        if symbol:
            query += " WHERE symbol = ?"
            params = (symbol,)
        query += " ORDER BY ts"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        cols = ["order_id", "symbol", "side", "price", "quantity", "ts", "commission"]
        return [dict(zip(cols, r)) for r in rows]

    def record_equity_point(self, ts: datetime, symbol: str, position_qty: int, position_value: Decimal) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO equity_curve VALUES (?,?,?,?)",
                (ts.isoformat(), symbol, position_qty, float(position_value)),
            )

    def equity_curve(self, symbol: str | None = None) -> list[dict]:
        query = "SELECT ts, symbol, position_qty, position_value FROM equity_curve"
        params: tuple = ()
        if symbol:
            query += " WHERE symbol = ?"
            params = (symbol,)
        query += " ORDER BY ts"
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        cols = ["ts", "symbol", "position_qty", "position_value"]
        return [dict(zip(cols, r)) for r in rows]

    # -- run heartbeats -------------------------------------------------
    def record_heartbeat(self, source: str, ts: datetime, status: str, detail: str = "") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_heartbeats (source, ts, status, detail) VALUES (?,?,?,?)",
                (source, ts.isoformat(), status, detail),
            )

    def latest_heartbeat(self, source: str) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT source, ts, status, detail FROM run_heartbeats "
                "WHERE source = ? ORDER BY ts DESC LIMIT 1",
                (source,),
            ).fetchone()
        if row is None:
            return None
        return dict(zip(["source", "ts", "status", "detail"], row))

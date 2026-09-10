"""Workflow: replay stored history bar-by-bar and simulate what the
deterministic threshold service would have decided each day, writing
the results to a separate database (data/backtest.db) so they never mix
with real paper-trading history.

Two honesty notes, also shown in the dashboard:

1. Uses ThresholdDecisionService, not the Claude agent. Backtesting an
   LLM agent bar-by-bar over years of history means one API call per
   symbol per day -- thousands of calls, real cost and latency, for a
   simulation whose point is to sanity-check the *quant* signal's
   history. Run scripts/run_paper_trader.py to see the Claude agent's
   reasoning live, a few decisions at a time, where it's actually meant
   to be read.

2. The model used here is whatever is currently the latest trained
   version -- trained on the FULL stored history, including data from
   after many of the simulated decision points. That means this
   backtest's results are optimistic and should NOT be read as an
   unbiased performance estimate. The honest number is the held-out
   eval_accuracy / eval_auc that orchestration/retrain_model.py reports
   (a real walk-forward split with no lookahead).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

from ironbridge_trader.adapters.paper_broker import PaperBroker
from ironbridge_trader.adapters.replay_market_data import ReplayMarketData
from ironbridge_trader.config import Settings
from ironbridge_trader.engine.trading_engine import TradingEngine
from ironbridge_trader.features.engineering import FEATURE_NAMES, MIN_BARS_REQUIRED
from ironbridge_trader.ml.model import PricePredictor
from ironbridge_trader.risk.manager import RiskManager
from ironbridge_trader.risk.position_sizer import PositionSizer
from ironbridge_trader.services.threshold_agent import ThresholdDecisionService
from ironbridge_trader.storage.db import Database

logger = logging.getLogger(__name__)

BACKTEST_DB_NAME = "backtest.db"


@dataclass(frozen=True, slots=True)
class BacktestResult:
    symbols_replayed: int
    total_fills: int
    backtest_db_path: str


def run(source_db: Database, settings: Settings) -> BacktestResult:
    version = source_db.latest_model_version()
    if version is None:
        raise SystemExit("no trained model found -- run scripts/train_model.py first")
    predictor = PricePredictor.load(settings.model_dir, version)

    backtest_db_path = settings.data_dir / BACKTEST_DB_NAME
    backtest_db_path.unlink(missing_ok=True)  # fresh run every time, no stale rows
    backtest_db = Database(backtest_db_path)

    symbols = list(settings.symbols)
    replay = ReplayMarketData(source_db, symbols, start_index=MIN_BARS_REQUIRED)

    # The dashboard reads bars, decisions, fills, and equity all from one
    # db per mode -- copy the (immutable) price history and model-version
    # metadata into backtest_db too, so "Backtest" mode is self-contained.
    for symbol in symbols:
        backtest_db.upsert_bars(source_db.get_bars(symbol))
    for v in source_db.model_versions():
        backtest_db.insert_model_version(
            version=v["version"], trained_at=datetime.fromisoformat(v["trained_at"]),
            train_rows=v["train_rows"], eval_rows=v["eval_rows"],
            eval_accuracy=v["eval_accuracy"], eval_auc=v["eval_auc"],
            feature_names=FEATURE_NAMES,
        )

    engine = TradingEngine(
        market_data=replay,
        predictor=predictor,
        decision_maker=ThresholdDecisionService(settings),
        execution=PaperBroker(),
        risk=RiskManager(settings.max_position_size, settings.margin_rate),
        position_sizer=PositionSizer(settings.risk_fraction, settings.stop_loss_fraction),
        db=backtest_db,
        settings=settings,
    )

    for symbol in symbols:
        while True:
            engine.run_cycle([symbol])
            if not replay.advance(symbol):
                break

    return BacktestResult(
        symbols_replayed=len(symbols),
        total_fills=len(backtest_db.all_fills()),
        backtest_db_path=str(backtest_db_path),
    )

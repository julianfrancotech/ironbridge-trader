"""Workflow: run one live paper-trading cycle across the watchlist.

This is where the choice of decision-maker is made -- Claude's
tool-using agent if ANTHROPIC_API_KEY is configured, otherwise the
deterministic threshold service -- and where the engine's collaborators
(model, risk manager, paper broker) get wired together for a real run.
protocols.DecisionMaker means the engine itself never needs to know
which one it got.
"""

from __future__ import annotations

import logging

from ironbridge_trader.adapters.paper_broker import PaperBroker
from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Decision
from ironbridge_trader.engine.trading_engine import TradingEngine
from ironbridge_trader.ml.model import PricePredictor
from ironbridge_trader.protocols import DecisionMaker
from ironbridge_trader.risk.manager import RiskManager
from ironbridge_trader.risk.position_sizer import PositionSizer
from ironbridge_trader.services.threshold_agent import ThresholdDecisionService
from ironbridge_trader.services.trading_agent import ClaudeTradingAgent
from ironbridge_trader.storage.db import Database

logger = logging.getLogger(__name__)


def build_decision_maker(settings: Settings, db: Database) -> DecisionMaker:
    if settings.anthropic_api_key:
        logger.info("using ClaudeTradingAgent (ANTHROPIC_API_KEY is set)")
        return ClaudeTradingAgent(settings, db)
    logger.info("ANTHROPIC_API_KEY not set -- using the deterministic ThresholdDecisionService")
    return ThresholdDecisionService(settings)


def run(db: Database, settings: Settings) -> list[Decision]:
    version = db.latest_model_version()
    if version is None:
        raise SystemExit("no trained model found -- run scripts/train_model.py first")
    predictor = PricePredictor.load(settings.model_dir, version)

    engine = TradingEngine(
        market_data=db,
        predictor=predictor,
        decision_maker=build_decision_maker(settings, db),
        execution=PaperBroker(),
        risk=RiskManager(settings.max_position_size, settings.margin_rate),
        position_sizer=PositionSizer(settings.risk_fraction, settings.stop_loss_fraction),
        db=db,
        settings=settings,
    )
    return engine.run_cycle(list(settings.symbols))

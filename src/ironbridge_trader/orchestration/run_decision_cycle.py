"""Workflow: run one live paper-trading cycle across the watchlist.

This is where the choice of decision-maker is made -- Claude's
tool-using agent if ANTHROPIC_API_KEY is configured, otherwise the
deterministic threshold service -- and where the engine's collaborators
(model, risk manager, broker) get wired together for a real run.
protocols.DecisionMaker / protocols.ExecutionClient mean the engine
itself never needs to know which concrete one it got.

build_execution_client is deliberately NOT the same "auto-switch the
moment credentials exist" pattern as build_decision_maker: it only
picks Alpaca when Settings.execution_provider is explicitly set to
"alpaca_paper". Sending real orders to a real broker's (paper) account
is a bigger step than picking an LLM vs. a threshold rule, and deserves
an explicit choice rather than turning on implicitly because a .env
file happens to have Alpaca keys in it.
"""

from __future__ import annotations

import logging

from ironbridge_trader.adapters.alpaca_broker import AlpacaBroker
from ironbridge_trader.adapters.paper_broker import PaperBroker
from ironbridge_trader.config import Settings
from ironbridge_trader.domain.models import Decision
from ironbridge_trader.engine.trading_engine import TradingEngine
from ironbridge_trader.ml.model import PricePredictor
from ironbridge_trader.protocols import DecisionMaker, ExecutionClient
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


def build_execution_client(settings: Settings) -> ExecutionClient:
    if settings.execution_provider == "alpaca_paper":
        if settings.alpaca_api_key and settings.alpaca_secret_key:
            logger.info("using AlpacaBroker (execution_provider=alpaca_paper)")
            return AlpacaBroker(settings.alpaca_api_key, settings.alpaca_secret_key)
        logger.warning(
            "execution_provider=alpaca_paper but ALPACA_API_KEY/ALPACA_SECRET_KEY aren't "
            "set in .env -- falling back to PaperBroker"
        )
    return PaperBroker(settings.transaction_cost_bps, settings.commission_per_trade)


def run(db: Database, settings: Settings) -> list[Decision]:
    if not settings.trading_enabled:
        # The kill switch. Checked here, not inside TradingEngine, so
        # scripts/backtest.py (which builds its own engine directly,
        # never through this function) keeps working even while live
        # trading is paused.
        logger.warning("trading_enabled=False -- skipping this cycle (kill switch is on)")
        return []

    version = db.latest_model_version()
    if version is None:
        raise SystemExit("no trained model found -- run scripts/train_model.py first")
    predictor = PricePredictor.load(settings.model_dir, version)

    engine = TradingEngine(
        market_data=db,
        predictor=predictor,
        decision_maker=build_decision_maker(settings, db),
        execution=build_execution_client(settings),
        risk=RiskManager(settings.max_position_size, settings.margin_rate),
        position_sizer=PositionSizer(settings.risk_fraction, settings.stop_loss_fraction),
        db=db,
        settings=settings,
    )
    return engine.run_cycle(list(settings.symbols))

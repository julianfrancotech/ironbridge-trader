import logging

from ironbridge_trader.analytics.risk_veto import capture_risk_vetoes

_ENGINE_LOGGER = "ironbridge_trader.engine.trading_engine"


def test_captures_and_counts_by_symbol_and_reason():
    logger = logging.getLogger(_ENGINE_LOGGER)

    with capture_risk_vetoes() as vetoes:
        logger.warning("order for %s blocked: %s: %s", "AAPL", "PositionLimitExceededError", "boom")
        logger.warning("order for %s blocked: %s: %s", "AAPL", "PositionLimitExceededError", "boom")
        logger.warning("order for %s blocked: %s: %s", "MSFT", "InsufficientMarginError", "boom")

    assert vetoes.total == 3
    assert vetoes.by_reason()["PositionLimitExceededError"] == 2
    assert vetoes.by_reason()["InsufficientMarginError"] == 1
    assert vetoes.by_symbol()["AAPL"] == 2
    assert vetoes.by_symbol()["MSFT"] == 1


def test_ignores_warnings_from_other_loggers():
    with capture_risk_vetoes() as vetoes:
        logging.getLogger("some.other.module").warning(
            "order for %s blocked: %s: %s", "AAPL", "X", "boom"
        )

    assert vetoes.total == 0


def test_ignores_the_differently_shaped_skip_warning_from_the_same_logger():
    # engine/trading_engine.py's run_cycle also logs a 3-arg warning for
    # a completely different reason (insufficient history) -- must not
    # be miscounted as a risk-manager veto just because it shares a
    # logger and an arg count.
    logger = logging.getLogger(_ENGINE_LOGGER)

    with capture_risk_vetoes() as vetoes:
        logger.warning("skipping %s this cycle: %s: %s", "AAPL", "TraderError", "not enough history")

    assert vetoes.total == 0


def test_handler_detaches_after_the_context_manager_exits():
    logger = logging.getLogger(_ENGINE_LOGGER)
    handlers_before = list(logger.handlers)

    with capture_risk_vetoes():
        pass

    assert logger.handlers == handlers_before

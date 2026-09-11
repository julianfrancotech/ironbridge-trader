from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pandas as pd
import pytest

from ironbridge_trader.analytics.multi_regime import (
    RegimeWindowResult,
    evaluate_regime_windows,
    split_into_windows,
)
from ironbridge_trader.domain.models import Bar, Fill, Side
from ironbridge_trader.storage.db import Database


def test_split_into_windows_creates_equal_contiguous_spans():
    index = pd.date_range("2024-01-01", "2024-01-11", freq="D")  # 11 days

    windows = split_into_windows(index, num_windows=2)

    assert len(windows) == 2
    assert windows[0][0] == pd.Timestamp("2024-01-01")
    assert windows[0][1] == windows[1][0]  # contiguous -- no gap, no overlap
    assert windows[1][1] == pd.Timestamp("2024-01-11")


def test_split_into_windows_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        split_into_windows(pd.date_range("2024-01-01", "2024-01-05"), num_windows=0)


def test_evaluate_regime_windows_returns_empty_list_for_an_empty_database(tmp_path):
    db = Database(tmp_path / "t.db")
    assert evaluate_regime_windows(db, ["A"], starting_equity=Decimal(1000)) == []


def test_a_strategy_can_lose_one_window_and_win_the_next(tmp_path):
    # Uptrend for 6 days, then downtrend for 6 days. The "strategy" buys
    # at the bottom, sells at the peak (locks in the uptrend's gain,
    # smaller position than the benchmark so it still underperforms
    # during the rise), then sits in cash through the whole decline.
    # This is the exact scenario item 2 exists to catch: one number
    # for the whole span would average these into something misleading.
    db = Database(tmp_path / "t.db")
    start = datetime(2024, 1, 1, tzinfo=UTC)
    closes = [100, 110, 120, 130, 140, 150, 140, 120, 110, 100, 90, 80]
    bars = [
        Bar(
            symbol="A", timestamp=start + timedelta(days=i), open=Decimal(c), high=Decimal(c),
            low=Decimal(c), close=Decimal(c), volume=1000,
        )
        for i, c in enumerate(closes)
    ]
    db.upsert_bars(bars)
    db.insert_fill(Fill(order_id="o1", symbol="A", side=Side.BUY, price=Decimal(100), quantity=5, timestamp=start))
    db.insert_fill(
        Fill(
            order_id="o2", symbol="A", side=Side.SELL, price=Decimal(150), quantity=5,
            timestamp=start + timedelta(days=5),
        )
    )

    windows = evaluate_regime_windows(db, ["A"], starting_equity=Decimal(1000), num_windows=2)

    assert len(windows) == 2
    # Window 1 (the rise): strategy only held 5 of a possible 10 units,
    # so it underperforms the fully-invested benchmark: +25% vs +50%.
    assert windows[0].strategy_return == pytest.approx(0.25)
    assert windows[0].benchmark_return == pytest.approx(0.50)
    assert windows[0].beats_return is False
    # Window 2 (the decline): strategy is flat in cash, benchmark rides
    # the whole decline down: 0% vs -42.9%.
    assert windows[1].strategy_return == pytest.approx(0.0)
    assert windows[1].benchmark_return == pytest.approx(-0.42857, abs=1e-4)
    assert windows[1].beats_return is True


def _window(strategy_sharpe=None, benchmark_sharpe=None, strategy_dd=None, benchmark_dd=None) -> RegimeWindowResult:
    return RegimeWindowResult(
        window_index=0, start=pd.Timestamp("2024-01-01"), end=pd.Timestamp("2024-06-01"),
        strategy_return=0.0, benchmark_return=0.0,
        strategy_sharpe=strategy_sharpe, benchmark_sharpe=benchmark_sharpe,
        strategy_max_drawdown=strategy_dd, benchmark_max_drawdown=benchmark_dd,
    )


def test_beats_sharpe_requires_both_values_and_the_configured_margin():
    assert _window(strategy_sharpe=1.5, benchmark_sharpe=1.0).beats_sharpe is True
    assert _window(strategy_sharpe=0.5, benchmark_sharpe=1.0).beats_sharpe is False
    assert _window(strategy_sharpe=None, benchmark_sharpe=1.0).beats_sharpe is False  # no benefit of the doubt


def test_within_drawdown_budget_allows_the_configured_multiple():
    # benchmark drawdown -10%, budget is 1.25x -> strategy allowed down to -12.5%
    assert _window(strategy_dd=-0.10, benchmark_dd=-0.10).within_drawdown_budget is True
    assert _window(strategy_dd=-0.12, benchmark_dd=-0.10).within_drawdown_budget is True
    assert _window(strategy_dd=-0.20, benchmark_dd=-0.10).within_drawdown_budget is False
    assert _window(strategy_dd=-0.10, benchmark_dd=None).within_drawdown_budget is False

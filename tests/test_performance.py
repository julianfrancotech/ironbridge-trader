from datetime import UTC, datetime, timedelta
from decimal import Decimal

from ironbridge_trader.analytics.performance import (
    compute_buy_and_hold_curve,
    compute_drawdown,
    compute_equity_curve,
    max_drawdown,
    sharpe_ratio,
)
from ironbridge_trader.domain.models import Bar, Fill, Side
from ironbridge_trader.storage.db import Database

CLOSES = [100.0, 110.0, 90.0, 120.0, 130.0]


def _seed_single_symbol(db: Database) -> list[datetime]:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    dates = [start + timedelta(days=i) for i in range(len(CLOSES))]
    bars = [
        Bar(
            symbol="A", timestamp=d, open=Decimal(str(c)), high=Decimal(str(c)),
            low=Decimal(str(c)), close=Decimal(str(c)), volume=1000,
        )
        for d, c in zip(dates, CLOSES)
    ]
    db.upsert_bars(bars)
    # buy 5 units on day 0 at that day's close (100) -- half the $1,000
    # starting equity stays uninvested as cash on purpose, so the buy-
    # and-hold benchmark (fully invested) has something real to beat.
    db.insert_fill(
        Fill(order_id="o1", symbol="A", side=Side.BUY, price=Decimal(100), quantity=5, timestamp=dates[0])
    )
    return dates


def test_equity_curve_tracks_cash_plus_mark_to_market(tmp_path):
    db = Database(tmp_path / "t.db")
    _seed_single_symbol(db)

    equity = compute_equity_curve(db, ["A"], starting_equity=Decimal(1000))

    # cash=500 after the buy, +5 * close on each day
    assert list(equity.round(2)) == [1000.0, 1050.0, 950.0, 1100.0, 1150.0]


def test_equity_curve_subtracts_commission_regardless_of_side(tmp_path):
    db = Database(tmp_path / "t.db")
    dates = _seed_single_symbol(db)
    db.insert_fill(
        Fill(
            order_id="o2", symbol="A", side=Side.SELL, price=Decimal(110), quantity=2,
            timestamp=dates[1], commission=Decimal("1.50"),
        )
    )

    equity = compute_equity_curve(db, ["A"], starting_equity=Decimal(1000))

    # day 1: 500 cash + 220 proceeds - 1.50 commission = 718.50, plus the
    # remaining 3 shares marked at day 1's close (110) = 330 -> 1048.50
    assert float(equity.iloc[1]) == 1048.50


def test_buy_and_hold_fully_invests_and_can_beat_the_strategy(tmp_path):
    db = Database(tmp_path / "t.db")
    _seed_single_symbol(db)

    strategy = compute_equity_curve(db, ["A"], starting_equity=Decimal(1000))
    benchmark = compute_buy_and_hold_curve(db, ["A"], starting_equity=Decimal(1000))

    # benchmark buys 10 shares (all $1,000) on day 0 vs the strategy's 5 --
    # same price series, so benchmark moves twice as far in both directions.
    assert list(benchmark.round(2)) == [1000.0, 1100.0, 900.0, 1200.0, 1300.0]
    assert benchmark.iloc[-1] > strategy.iloc[-1]


def test_max_drawdown_is_negative_and_matches_the_worst_dip(tmp_path):
    db = Database(tmp_path / "t.db")
    _seed_single_symbol(db)
    equity = compute_equity_curve(db, ["A"], starting_equity=Decimal(1000))

    dd = compute_drawdown(equity)
    worst = max_drawdown(equity)

    assert worst < 0
    assert worst == dd.min()
    # day index 2 (value 950) dips from the running peak of 1050 set on day 1
    assert worst == (950.0 / 1050.0 - 1.0)


def test_sharpe_ratio_is_positive_for_a_net_upward_curve(tmp_path):
    db = Database(tmp_path / "t.db")
    _seed_single_symbol(db)
    equity = compute_equity_curve(db, ["A"], starting_equity=Decimal(1000))

    sharpe = sharpe_ratio(equity)

    assert sharpe is not None
    assert sharpe > 0


def test_sharpe_ratio_is_none_for_a_flat_curve():
    import pandas as pd

    flat = pd.Series([1000.0, 1000.0, 1000.0, 1000.0])
    assert sharpe_ratio(flat) is None


def test_empty_database_returns_empty_series(tmp_path):
    db = Database(tmp_path / "t.db")
    assert compute_equity_curve(db, ["A"], starting_equity=Decimal(1000)).empty
    assert compute_buy_and_hold_curve(db, ["A"], starting_equity=Decimal(1000)).empty

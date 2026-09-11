"""Multi-regime backtest analysis: slices one continuous backtest's
equity curve (analytics/performance.py) into several calendar-length
windows and evaluates each independently against
analytics/validation_bar.py's Sharpe and drawdown thresholds. The
point (docs/validation-plan.md, item 2 of the Phase 0 plan): a model
can look good over one continuous multi-year span and still be
worthless in half of it -- one backtest number can't tell you which
you have.

Windows are mechanical, equal-calendar-length splits of the available
history, not hand-picked "the bull period" / "the bear period" -- that
avoids the temptation to choose boundaries that flatter or damn the
result, the same reasoning docs/validation-plan.md gives for keeping
these thresholds out of a Settings field.

Positions and the buy-and-hold benchmark carry through window
boundaries unmodified -- no artificial reset to a fresh starting
balance at each window. This is the realistic version of "how did the
strategy do during this stretch of calendar time," not "what if I
reset to $100k and started fresh here." Sharpe and drawdown are still
computed fresh per window (using that window's own running peak and
volatility), so a strong window can't hide a weak one or vice versa.

Deliberately out of scope here: the permutation test (criterion D in
validation-plan.md). That needs N full retrains, which is heavier
infrastructure than slicing an existing curve -- it belongs with the
model-promotion gate work, not this diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

import pandas as pd

from ironbridge_trader.analytics import validation_bar
from ironbridge_trader.analytics.performance import (
    compute_buy_and_hold_curve,
    compute_drawdown,
    compute_equity_curve,
    sharpe_ratio,
)
from ironbridge_trader.storage.db import Database


@dataclass(frozen=True, slots=True)
class RegimeWindowResult:
    window_index: int
    start: pd.Timestamp
    end: pd.Timestamp
    strategy_return: float
    benchmark_return: float
    strategy_sharpe: float | None
    benchmark_sharpe: float | None
    strategy_max_drawdown: float | None
    benchmark_max_drawdown: float | None

    @property
    def beats_return(self) -> bool:
        """Criterion A (validation-plan.md): no margin, just >=."""
        return self.strategy_return >= self.benchmark_return

    @property
    def beats_sharpe(self) -> bool:
        """Criterion B. False (not "unknown") when either Sharpe is
        undefined -- a flat or too-short window can't demonstrate this
        criterion, so it doesn't get benefit of the doubt.
        """
        if self.strategy_sharpe is None or self.benchmark_sharpe is None:
            return False
        return self.strategy_sharpe >= self.benchmark_sharpe * validation_bar.MIN_SHARPE_RATIO_VS_BENCHMARK

    @property
    def within_drawdown_budget(self) -> bool:
        """Criterion C. Drawdowns are <= 0; "within budget" means not
        more negative than MAX_DRAWDOWN_MULTIPLE_OF_BENCHMARK times the
        benchmark's own worst drawdown in this same window.
        """
        if self.strategy_max_drawdown is None or self.benchmark_max_drawdown is None:
            return False
        budget = self.benchmark_max_drawdown * validation_bar.MAX_DRAWDOWN_MULTIPLE_OF_BENCHMARK
        return self.strategy_max_drawdown >= budget


def _total_return(series: pd.Series) -> float:
    return float(series.iloc[-1] / series.iloc[0] - 1.0)


def split_into_windows(index: pd.DatetimeIndex, num_windows: int) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if num_windows < 1:
        raise ValueError("num_windows must be >= 1")
    boundaries = pd.date_range(index[0], index[-1], periods=num_windows + 1)
    return [(boundaries[i], boundaries[i + 1]) for i in range(num_windows)]


def evaluate_regime_windows(
    db: Database, symbols: list[str], starting_equity: Decimal, num_windows: int = 4,
) -> list[RegimeWindowResult]:
    equity = compute_equity_curve(db, symbols, starting_equity)
    benchmark = compute_buy_and_hold_curve(db, symbols, starting_equity)
    if equity.empty or benchmark.empty:
        return []

    results = []
    for i, (start, end) in enumerate(split_into_windows(equity.index, num_windows)):
        strategy_slice = equity.loc[start:end]
        benchmark_slice = benchmark.loc[start:end]
        if len(strategy_slice) < 2 or len(benchmark_slice) < 2:
            continue  # too few bars in this window to say anything (e.g. a short final slice)

        results.append(
            RegimeWindowResult(
                window_index=i, start=start, end=end,
                strategy_return=_total_return(strategy_slice),
                benchmark_return=_total_return(benchmark_slice),
                strategy_sharpe=sharpe_ratio(strategy_slice),
                benchmark_sharpe=sharpe_ratio(benchmark_slice),
                strategy_max_drawdown=float(compute_drawdown(strategy_slice).min()),
                benchmark_max_drawdown=float(compute_drawdown(benchmark_slice).min()),
            )
        )
    return results

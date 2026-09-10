"""Portfolio-level performance analytics: the actual account equity curve
(cash plus mark-to-market position value, across every symbol combined,
against the SAME shared account_equity pool risk/manager.py checks
against) plus a buy-and-hold benchmark and the two risk metrics --
max drawdown and Sharpe ratio -- the dashboard didn't show before this.

Computed here, at read time, from what's already stored (bars + fills),
rather than written incrementally during the trading loop. The question
"how has my TOTAL account done" needs every symbol's fills and prices
reconciled against one shared cash balance; storage/db.py's
equity_curve table tracks each symbol's position independently and only
on days it actually filled an order -- the right shape for the
per-symbol Portfolio-tab charts, but not for one combined curve.
"""

from __future__ import annotations

from decimal import Decimal

import pandas as pd

from ironbridge_trader.storage.db import Database

# Standard equities convention. An approximation for a watchlist that
# also holds BTC-USD, which trades 365 days/year -- see README.
TRADING_DAYS_PER_YEAR = 252


def _price_matrix(db: Database, symbols: list[str]) -> pd.DataFrame:
    """One row per date (union of every symbol's bar dates), one column
    per symbol, forward-filled close price. Forward-fill matters because
    the watchlist mixes trading calendars -- on a day AAPL's market is
    closed, an existing AAPL position still needs a last-known price to
    be marked to market against.
    """
    series = {}
    for symbol in symbols:
        bars = db.get_bars(symbol)
        if bars:
            series[symbol] = pd.Series({b.timestamp: float(b.close) for b in bars})
    if not series:
        return pd.DataFrame()
    return pd.DataFrame(series).sort_index().ffill()


def compute_equity_curve(db: Database, symbols: list[str], starting_equity: Decimal) -> pd.Series:
    """cash + sum(qty[s] * price[s][t]) at every date, replaying every
    stored fill across every symbol in chronological order against one
    shared cash balance that starts at `starting_equity`.
    """
    prices = _price_matrix(db, symbols)
    if prices.empty:
        return pd.Series(dtype=float)

    fills = db.all_fills()  # every symbol, already ORDER BY ts
    fill_idx = 0
    cash = float(starting_equity)
    qty = dict.fromkeys(prices.columns, 0.0)
    equity_values = []

    for date in prices.index:
        while fill_idx < len(fills) and pd.Timestamp(fills[fill_idx]["ts"]) <= date:
            f = fills[fill_idx]
            notional = f["price"] * f["quantity"]
            if f["side"] == "BUY":
                cash -= notional
                qty[f["symbol"]] = qty.get(f["symbol"], 0.0) + f["quantity"]
            else:
                cash += notional
                qty[f["symbol"]] = qty.get(f["symbol"], 0.0) - f["quantity"]
            fill_idx += 1

        position_value = sum(
            qty[s] * prices.loc[date, s]
            for s in qty
            if s in prices.columns and pd.notna(prices.loc[date, s])
        )
        equity_values.append(cash + position_value)

    return pd.Series(equity_values, index=prices.index, name="equity")


def compute_buy_and_hold_curve(db: Database, symbols: list[str], starting_equity: Decimal) -> pd.Series:
    """Equal-dollar allocation across every symbol, bought once at each
    symbol's own first available price and held -- zero decisions, zero
    trades, zero exposure to any mistake covered in the README's Trading
    theory section. The benchmark every active strategy has to clear.
    """
    prices = _price_matrix(db, symbols)
    if prices.empty:
        return pd.Series(dtype=float)

    allocation = float(starting_equity) / len(prices.columns)
    first_prices = prices.bfill().iloc[0]  # each column's own first available price
    contributions = pd.DataFrame(
        {
            s: prices[s] * (allocation / first_prices[s])
            for s in prices.columns
            if first_prices[s] > 0
        }
    )
    return contributions.sum(axis=1, skipna=True).rename("buy_and_hold")


def compute_drawdown(equity: pd.Series) -> pd.Series:
    """Fraction below the running peak, at every point in time -- e.g.
    -0.20 means the account is currently 20% below its highest-ever
    value. This is the number a trader watching a real account actually
    feels; a chart of raw account value alone hides it.
    """
    if equity.empty:
        return equity
    return equity / equity.cummax() - 1.0


def max_drawdown(equity: pd.Series) -> float | None:
    dd = compute_drawdown(equity)
    return None if dd.empty else float(dd.min())


def sharpe_ratio(equity: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float | None:
    """Mean daily return divided by its own volatility, annualized -- the
    standard way to ask "was this good, or did it just take a lot of risk
    to get lucky." NOT adjusted for a risk-free rate (a simplification --
    a textbook Sharpe subtracts it, but it rarely changes the conclusion
    at this scale). None when there isn't enough history, or the curve
    never moved at all (dividing by a zero standard deviation is
    undefined, not zero).
    """
    daily_returns = equity.pct_change().dropna()
    std = daily_returns.std()
    if len(daily_returns) < 2 or std == 0 or pd.isna(std):
        return None
    return float(daily_returns.mean() / std * (periods_per_year**0.5))

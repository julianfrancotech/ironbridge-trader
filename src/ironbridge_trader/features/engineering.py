"""Technical-indicator feature engineering.

Every feature here is a *relative* or *normalized* quantity (a ratio, a
z-score, a percent change) rather than a raw price. Raw closing prices
carry no signal a tree model can generalize from (AAPL at $180 today
tells it nothing about AAPL at $50 five years ago), but the *shape* of
recent price action -- momentum, whether short-term average is above
long-term average, how volatile things have been -- repeats across
price levels and across symbols. That's what lets one model, trained
on years of history, stay useful as prices drift.
"""

from __future__ import annotations

import pandas as pd

from ironbridge_trader.domain.models import Bar

FEATURE_NAMES = [
    "ret_1", "ret_5", "ret_10",
    "sma_ratio", "ema_ratio", "rsi_14",
    "volatility_10", "volume_z", "range_pct",
]

MIN_BARS_REQUIRED = 30  # slowest indicator (SMA20) plus a little slack


def _bars_to_frame(bars: list[Bar]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "close": [float(b.close) for b in bars],
            "high": [float(b.high) for b in bars],
            "low": [float(b.low) for b in bars],
            "volume": [float(b.volume) for b in bars],
        }
    )


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss.replace(0, 1e-9)
    return 100 - (100 / (1 + rs))


def _feature_frame(bars: list[Bar]) -> pd.DataFrame:
    """All indicators, row-aligned with `bars`. Early rows are NaN until
    enough history has accumulated for the slowest window.
    """
    df = _bars_to_frame(bars)
    close = df["close"]

    out = pd.DataFrame(index=df.index)
    out["ret_1"] = close.pct_change(1)
    out["ret_5"] = close.pct_change(5)
    out["ret_10"] = close.pct_change(10)

    sma5 = close.rolling(5).mean()
    sma20 = close.rolling(20).mean()
    out["sma_ratio"] = sma5 / sma20 - 1

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    out["ema_ratio"] = ema12 / ema26 - 1

    out["rsi_14"] = _rsi(close, 14) / 100.0  # scaled to [0, 1]

    daily_ret = close.pct_change()
    out["volatility_10"] = daily_ret.rolling(10).std()

    vol_mean20 = df["volume"].rolling(20).mean()
    vol_std20 = df["volume"].rolling(20).std().replace(0, 1e-9)
    out["volume_z"] = (df["volume"] - vol_mean20) / vol_std20

    out["range_pct"] = (df["high"] - df["low"]) / close

    return out[FEATURE_NAMES]


def compute_features(bars: list[Bar]) -> dict[str, float] | None:
    """Feature snapshot as of the most recent bar. None if not enough history."""
    if len(bars) < MIN_BARS_REQUIRED:
        return None
    row = _feature_frame(bars).iloc[-1]
    if row.isna().any():
        return None
    return {k: float(v) for k, v in row.items()}


def build_training_frame(bars: list[Bar]) -> pd.DataFrame:
    """Features + label for every bar that has enough trailing history AND
    a known next-bar outcome (so the very last bar, with no "next" yet,
    is dropped). label = 1 if the next bar's close is higher, else 0.
    """
    features = _feature_frame(bars)
    close = pd.Series([float(b.close) for b in bars])
    label = (close.shift(-1) > close).astype(float)

    frame = features.copy()
    frame["label"] = label
    frame["timestamp"] = [b.timestamp for b in bars]
    frame = frame.dropna()
    return frame.reset_index(drop=True)

import pandas as pd
import pytest
import requests

from ironbridge_trader.adapters import market_data
from ironbridge_trader.domain.exceptions import MarketDataError


def _fake_history_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {"Open": [1.0], "High": [1.5], "Low": [0.5], "Close": [1.2], "Volume": [100]},
        index=pd.DatetimeIndex(["2024-01-02"], tz="UTC"),
    )


class _FakeTicker:
    """Stands in for yf.Ticker so tests exercise the real retry-decorated
    _download_history, not a monkeypatched replacement that would bypass
    the decorator entirely.
    """

    def __init__(self, history_fn):
        self._history_fn = history_fn

    def history(self, period, interval, auto_adjust):
        return self._history_fn()


def test_fetch_bars_retries_transient_network_errors_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 2:
            raise requests.exceptions.ConnectionError("dropped")
        return _fake_history_frame()

    monkeypatch.setattr(market_data.yf, "Ticker", lambda symbol: _FakeTicker(flaky))

    bars = market_data.fetch_bars("TEST", years=1)

    assert calls["n"] == 2
    assert len(bars) == 1
    assert bars[0].symbol == "TEST"


def test_fetch_bars_raises_market_data_error_on_empty_result(monkeypatch):
    monkeypatch.setattr(market_data.yf, "Ticker", lambda symbol: _FakeTicker(pd.DataFrame))

    with pytest.raises(MarketDataError):
        market_data.fetch_bars("NOSUCHSYMBOL", years=1)

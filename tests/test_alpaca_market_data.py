from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from ironbridge_trader.adapters import alpaca_market_data
from ironbridge_trader.domain.exceptions import MarketDataError


class _FakeAlpacaBar:
    def __init__(self, timestamp, price=100.0, volume=1000):
        self.timestamp = timestamp
        self.open = price
        self.high = price
        self.low = price
        self.close = price
        self.volume = volume


class _FakeBarSet:
    def __init__(self, data: dict):
        self.data = data


class _FakeStockClient:
    """Stands in for StockHistoricalDataClient. respond_fn(request) -> BarSet
    or raises, so tests exercise the real retry-decorated _download_bars.
    """

    def __init__(self, respond_fn):
        self._respond_fn = respond_fn
        self.init_args: tuple = ()
        self.requests: list = []

    def get_stock_bars(self, request):
        self.requests.append(request)
        return self._respond_fn(request)


class _FakeCryptoClient:
    def __init__(self, respond_fn):
        self._respond_fn = respond_fn
        self.requests: list = []

    def get_crypto_bars(self, request):
        self.requests.append(request)
        return self._respond_fn(request)


class _FakeHTTPError:
    def __init__(self, status_code):
        self.response = SimpleNamespace(status_code=status_code)


def _api_error(status_code):
    from alpaca.common.exceptions import APIError

    return APIError("boom", _FakeHTTPError(status_code))


def test_fetch_bars_uses_stock_client_for_an_equity_symbol(monkeypatch):
    ts = datetime(2024, 6, 1, tzinfo=UTC)
    client = _FakeStockClient(lambda req: _FakeBarSet({"AAPL": [_FakeAlpacaBar(ts)]}))
    monkeypatch.setattr(alpaca_market_data, "StockHistoricalDataClient", lambda *a, **kw: client)

    bars = alpaca_market_data.fetch_bars("AAPL", years=5, api_key="k", secret_key="s")

    assert len(bars) == 1
    assert bars[0].symbol == "AAPL"
    assert client.requests[0].symbol_or_symbols == "AAPL"


def test_fetch_bars_uses_crypto_client_and_translates_symbol_format(monkeypatch):
    ts = datetime(2024, 6, 1, tzinfo=UTC)
    client = _FakeCryptoClient(lambda req: _FakeBarSet({"BTC/USD": [_FakeAlpacaBar(ts)]}))
    monkeypatch.setattr(alpaca_market_data, "CryptoHistoricalDataClient", lambda *a, **kw: client)

    bars = alpaca_market_data.fetch_bars("BTC-USD", years=5, api_key="k", secret_key="s")

    assert len(bars) == 1
    assert bars[0].symbol == "BTC-USD"  # translated back to this app's convention
    assert client.requests[0].symbol_or_symbols == "BTC/USD"  # but sent to Alpaca as a pair


def test_fetch_bars_passes_start_through_for_incremental_fetch(monkeypatch):
    last_stored = datetime(2024, 6, 1, tzinfo=UTC)
    client = _FakeStockClient(lambda req: _FakeBarSet({"AAPL": [_FakeAlpacaBar(last_stored)]}))
    monkeypatch.setattr(alpaca_market_data, "StockHistoricalDataClient", lambda *a, **kw: client)

    alpaca_market_data.fetch_bars("AAPL", years=5, start=last_stored, api_key="k", secret_key="s")

    # Alpaca's request model stores `start` as a naive datetime (its own
    # normalization, not this adapter's doing) -- same wall-clock value,
    # since window_start is always constructed in UTC.
    assert client.requests[0].start == last_stored.replace(tzinfo=None)


def test_fetch_bars_returns_empty_list_instead_of_raising_when_nothing_new_since_start(monkeypatch):
    client = _FakeStockClient(lambda req: _FakeBarSet({}))
    monkeypatch.setattr(alpaca_market_data, "StockHistoricalDataClient", lambda *a, **kw: client)

    bars = alpaca_market_data.fetch_bars(
        "AAPL", years=5, start=datetime(2024, 6, 1, tzinfo=UTC), api_key="k", secret_key="s"
    )

    assert bars == []


def test_fetch_bars_raises_market_data_error_on_empty_full_window_result(monkeypatch):
    client = _FakeStockClient(lambda req: _FakeBarSet({}))
    monkeypatch.setattr(alpaca_market_data, "StockHistoricalDataClient", lambda *a, **kw: client)

    with pytest.raises(MarketDataError):
        alpaca_market_data.fetch_bars("NOSUCHSYMBOL", years=5, api_key="k", secret_key="s")


def test_fetch_bars_rejects_a_non_daily_interval():
    with pytest.raises(MarketDataError):
        alpaca_market_data.fetch_bars("AAPL", years=5, interval="1h", api_key="k", secret_key="s")


def test_fetch_bars_retries_a_rate_limit_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    ts = datetime(2024, 6, 1, tzinfo=UTC)
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _api_error(429)
        return _FakeBarSet({"AAPL": [_FakeAlpacaBar(ts)]})

    client = _FakeStockClient(flaky)
    monkeypatch.setattr(alpaca_market_data, "StockHistoricalDataClient", lambda *a, **kw: client)

    bars = alpaca_market_data.fetch_bars("AAPL", years=5, api_key="k", secret_key="s")

    assert calls["n"] == 2
    assert len(bars) == 1


def test_fetch_bars_does_not_retry_a_permanent_api_error(monkeypatch):
    from alpaca.common.exceptions import APIError

    calls = {"n": 0}

    def always_rejected(req):
        calls["n"] += 1
        raise _api_error(422)  # e.g. invalid symbol -- not transient

    client = _FakeStockClient(always_rejected)
    monkeypatch.setattr(alpaca_market_data, "StockHistoricalDataClient", lambda *a, **kw: client)

    with pytest.raises(APIError):
        alpaca_market_data.fetch_bars("AAPL", years=5, api_key="k", secret_key="s")

    assert calls["n"] == 1  # failed once, immediately -- not retried into a slower identical failure

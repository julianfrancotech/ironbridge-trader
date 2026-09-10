from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace

import pytest
from alpaca.common.exceptions import APIError
from alpaca.trading.enums import OrderStatus as AlpacaOrderStatus

from ironbridge_trader.adapters import alpaca_broker
from ironbridge_trader.domain.exceptions import BrokerError
from ironbridge_trader.domain.models import Order, Side


def _alpaca_order(status, filled_avg_price="150.25", filled_qty="10", filled_at=None):
    return SimpleNamespace(
        status=status, filled_avg_price=filled_avg_price, filled_qty=filled_qty,
        filled_at=filled_at or datetime(2024, 6, 1, tzinfo=UTC),
    )


def _order(symbol="AAPL", order_id="AAPL-abc123", side=Side.BUY, quantity=10) -> Order:
    return Order.market_order(
        order_id=order_id, symbol=symbol, side=side, quantity=quantity,
        reference_price=Decimal(150), as_of=datetime(2024, 6, 1, tzinfo=UTC),
    )


class _FakeHTTPError:
    def __init__(self, status_code):
        self.response = SimpleNamespace(status_code=status_code)


def _api_error(status_code):
    return APIError("boom", _FakeHTTPError(status_code))


class _FakeTradingClient:
    def __init__(self, submit_fn, get_order_fn=None):
        self._submit_fn = submit_fn
        self._get_order_fn = get_order_fn
        self.submitted_requests: list = []
        self.status_checks = 0

    def submit_order(self, request):
        self.submitted_requests.append(request)
        return self._submit_fn(request)

    def get_order_by_client_id(self, client_id):
        self.status_checks += 1
        return self._get_order_fn(client_id)


def _make_broker(client: _FakeTradingClient) -> alpaca_broker.AlpacaBroker:
    broker = alpaca_broker.AlpacaBroker.__new__(alpaca_broker.AlpacaBroker)
    broker._client = client
    return broker


def test_place_order_returns_fill_immediately_when_submit_response_already_filled():
    client = _FakeTradingClient(lambda req: _alpaca_order(AlpacaOrderStatus.FILLED))
    broker = _make_broker(client)

    fill = broker.place_order(_order())

    assert fill.price == Decimal("150.25")
    assert fill.quantity == 10
    assert client.status_checks == 0  # no extra calls needed -- the common case


def test_place_order_sends_our_order_id_as_client_order_id_for_idempotency():
    client = _FakeTradingClient(lambda req: _alpaca_order(AlpacaOrderStatus.FILLED))
    broker = _make_broker(client)

    broker.place_order(_order(order_id="AAPL-xyz789"))

    assert client.submitted_requests[0].client_order_id == "AAPL-xyz789"


def test_place_order_translates_crypto_symbol_and_uses_gtc():
    client = _FakeTradingClient(lambda req: _alpaca_order(AlpacaOrderStatus.FILLED))
    broker = _make_broker(client)

    broker.place_order(_order(symbol="BTC-USD"))

    request = client.submitted_requests[0]
    assert request.symbol == "BTC/USD"
    assert request.time_in_force.value == "gtc"


def test_place_order_keeps_equity_symbol_and_uses_day():
    client = _FakeTradingClient(lambda req: _alpaca_order(AlpacaOrderStatus.FILLED))
    broker = _make_broker(client)

    broker.place_order(_order(symbol="AAPL"))

    request = client.submitted_requests[0]
    assert request.symbol == "AAPL"
    assert request.time_in_force.value == "day"


def test_place_order_polls_until_filled_when_not_immediately_terminal(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    statuses = iter([AlpacaOrderStatus.NEW, AlpacaOrderStatus.FILLED])
    client = _FakeTradingClient(
        submit_fn=lambda req: _alpaca_order(AlpacaOrderStatus.NEW),
        get_order_fn=lambda cid: _alpaca_order(next(statuses)),
    )
    broker = _make_broker(client)

    fill = broker.place_order(_order())

    assert fill.quantity == 10
    assert client.status_checks == 2


def test_place_order_raises_broker_error_immediately_on_rejected_status():
    client = _FakeTradingClient(lambda req: _alpaca_order(AlpacaOrderStatus.REJECTED))
    broker = _make_broker(client)

    with pytest.raises(BrokerError):
        broker.place_order(_order())

    assert client.status_checks == 0  # a terminal bad status needs no polling


def test_place_order_raises_broker_error_after_bounded_wait_exhausted(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    client = _FakeTradingClient(
        submit_fn=lambda req: _alpaca_order(AlpacaOrderStatus.NEW),
        get_order_fn=lambda cid: _alpaca_order(AlpacaOrderStatus.NEW),  # never resolves
    )
    broker = _make_broker(client)

    with pytest.raises(BrokerError):
        broker.place_order(_order())

    assert client.status_checks == alpaca_broker._POLL_ATTEMPTS  # bounded, not open-ended


def test_place_order_raises_broker_error_not_a_raw_api_error_on_permanent_submit_failure():
    # Regression test for the ExecutionClient contract: TradingEngine only
    # catches TraderError, so a raw APIError escaping here would break
    # per-symbol fault isolation in run_cycle (Phase B).
    def always_rejected(req):
        raise _api_error(422)

    client = _FakeTradingClient(always_rejected)
    broker = _make_broker(client)

    with pytest.raises(BrokerError):
        broker.place_order(_order())


def test_place_order_retries_a_transient_submit_error_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    def flaky(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise _api_error(503)
        return _alpaca_order(AlpacaOrderStatus.FILLED)

    client = _FakeTradingClient(flaky)
    broker = _make_broker(client)

    fill = broker.place_order(_order())

    assert calls["n"] == 2
    assert fill.quantity == 10


def test_place_order_does_not_retry_a_permanent_submit_error(monkeypatch):
    calls = {"n": 0}

    def always_rejected(req):
        calls["n"] += 1
        raise _api_error(422)

    client = _FakeTradingClient(always_rejected)
    broker = _make_broker(client)

    with pytest.raises(BrokerError):
        broker.place_order(_order())

    assert calls["n"] == 1

import pytest

from ironbridge_trader.resilience import retry_with_backoff


class FlakyError(Exception):
    pass


class PermanentError(Exception):
    pass


def test_retries_until_success(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=3, retry_on=(FlakyError,))
    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise FlakyError("not yet")
        return "ok"

    assert flaky() == "ok"
    assert calls["n"] == 3


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=3, retry_on=(FlakyError,))
    def always_fails():
        calls["n"] += 1
        raise FlakyError("still broken")

    with pytest.raises(FlakyError):
        always_fails()
    assert calls["n"] == 3


def test_does_not_retry_exceptions_outside_retry_on(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda _: None)
    calls = {"n": 0}

    @retry_with_backoff(max_attempts=3, retry_on=(FlakyError,))
    def wrong_error():
        calls["n"] += 1
        raise PermanentError("not transient")

    with pytest.raises(PermanentError):
        wrong_error()
    assert calls["n"] == 1  # never retried


def test_succeeds_on_first_try_without_sleeping(monkeypatch):
    def fail_if_called(_):
        raise AssertionError("should not sleep when the first attempt succeeds")

    monkeypatch.setattr("time.sleep", fail_if_called)

    @retry_with_backoff(max_attempts=3, retry_on=(FlakyError,))
    def works():
        return 42

    assert works() == 42

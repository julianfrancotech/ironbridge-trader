"""Retry-with-backoff for calls to systems this app doesn't control
(Yahoo Finance, the Anthropic API). Every external call here is
synchronous, so this is a plain function decorator -- the same idea
ironbridge-capital's cqg/decorators.py uses for its async websocket
client (retry only exceptions the caller names as transient, exponential
backoff, give up loudly after max_attempts), adapted to a sync,
batch/script context instead of a long-lived connection.

retry_on is deliberately explicit at every call site: retrying a
permanent failure (a malformed request, a bad API key) just delays an
inevitable failure and hides the real error behind pointless waiting.
"""

from __future__ import annotations

import functools
import logging
import time
from collections.abc import Callable
from typing import ParamSpec, TypeVar

logger = logging.getLogger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_seconds: float = 1.0,
    retry_on: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            attempt = 1
            while True:
                try:
                    return func(*args, **kwargs)
                except retry_on as exc:
                    if attempt >= max_attempts:
                        logger.error(
                            "%s: giving up after %d attempts (%s)",
                            func.__qualname__, attempt, exc,
                        )
                        raise
                    wait = backoff_seconds * (2 ** (attempt - 1))
                    logger.warning(
                        "%s: attempt %d/%d failed (%s), retrying in %.1fs",
                        func.__qualname__, attempt, max_attempts, exc, wait,
                    )
                    time.sleep(wait)
                    attempt += 1

        return wrapper

    return decorator

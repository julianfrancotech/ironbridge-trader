"""Bounded thread-pool helper for the handful of places this app maps
one blocking I/O call (yfinance/Alpaca REST, the Anthropic SDK) over a
list of symbols. Threads, not asyncio: every call being pooled is
already a blocking synchronous call, so a thread pool is the minimal
primitive that fits -- a full asyncio rewrite would touch every
adapter and service for no benefit at this scale.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")
R = TypeVar("R")


def run_bounded(items: Iterable[T], fn: Callable[[T], R], max_workers: int) -> list[Future[R]]:
    """Run fn(item) for every item, at most max_workers at a time, one
    task per item -- never two tasks for the same item, which is what
    keeps this safe to use over per-symbol work that isn't otherwise
    synchronized: different items' work interleaves, but nothing about
    a single item's own sequence changes.

    Returns one Future per item, in the same order as `items`, all
    already resolved (the pool is fully drained before this returns).
    Callers call .result() on each exactly as they would wrap a
    sequential call in try/except -- a failed item's exception is
    raised from .result(), not swallowed here, so error handling stays
    the caller's decision.
    """
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(fn, item) for item in items]
    return futures

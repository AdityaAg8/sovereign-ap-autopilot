"""In-memory token-bucket throttle for `POST /auth/login` (REVIEW L-02, docs/security.md B8): each
(client IP, email) pair gets `burst` attempts that refill at a steady rate, so password guessing
against one account is slowed to a crawl. Process-local by design: the API is one local process.
"""

from __future__ import annotations

import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

DEFAULT_MAX_KEYS: Final[int] = 50_000
"""Most (IP, email) buckets kept; beyond it the least recently used bucket is dropped, so a
spray of distinct emails cannot grow memory without bound."""


@dataclass
class _Bucket:
    tokens: float
    updated: float


class LoginRateLimiter:
    """Token bucket per key. `acquire` spends one token and returns 0.0, or returns the seconds
    until the next token without spending anything. Not thread-safe; the API calls it from its
    single event loop, where it never awaits."""

    def __init__(
        self,
        *,
        burst: int,
        refill_per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = DEFAULT_MAX_KEYS,
    ) -> None:
        if burst < 1 or refill_per_minute <= 0 or max_keys < 1:
            raise ValueError("burst, refill_per_minute and max_keys must be positive")
        self._burst = float(burst)
        self._rate = refill_per_minute / 60.0
        self._clock = clock
        self._max_keys = max_keys
        self._buckets: OrderedDict[tuple[str, str], _Bucket] = OrderedDict()

    def acquire(self, client_ip: str, email: str) -> float:
        key = (client_ip, email.strip().casefold())
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=self._burst, updated=now)
            self._buckets[key] = bucket
            while len(self._buckets) > self._max_keys:
                self._buckets.popitem(last=False)
        else:
            elapsed = max(0.0, now - bucket.updated)
            bucket.tokens = min(self._burst, bucket.tokens + elapsed * self._rate)
            bucket.updated = now
            self._buckets.move_to_end(key)
        if bucket.tokens >= 1.0:
            bucket.tokens -= 1.0
            return 0.0
        return (1.0 - bucket.tokens) / self._rate

    def __len__(self) -> int:
        return len(self._buckets)


def retry_after_header(seconds: float) -> str:
    """`Retry-After` value: whole seconds, rounded up, at least 1."""
    return str(max(1, math.ceil(seconds)))

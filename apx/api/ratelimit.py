"""In-memory token-bucket throttle for `POST /auth/login` (REVIEW L-02 and N-04, docs/security.md
B8): an account bucket per normalised email slows password guessing against one account, and a
coarse per-client-IP bucket with a much higher limit caps one source's total attempts. Keys are
sha256 digests in capped maps, so memory stays bounded whatever the usernames look like; a refused
request never creates an account bucket, and eviction drops the fullest old bucket, so a spray of
usernames cannot flush a drained account's bucket back to full (REVIEW.md P3-03).
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final

MAX_EMAIL_LENGTH: Final[int] = 320
"""Longest login email accepted (RFC 5321: 64-octet local part, `@`, 255-octet domain). The
route rejects anything longer with 422 before the limiter or argon2 sees it."""

DEFAULT_MAX_KEYS: Final[int] = 50_000
"""Most account buckets kept; beyond it the least recently used one is dropped. Each key is a
32-byte digest, so the cap bounds memory to a few MB whatever usernames are sent."""

EVICTION_WINDOW: Final[int] = 64
"""How many of the least recently used buckets eviction compares; it drops the one holding the
most tokens (the oldest among equals), so a drained bucket outlives a full one."""

DEFAULT_MAX_IP_KEYS: Final[int] = 10_000
"""Most client-IP buckets kept (LRU, like the account buckets)."""

DEFAULT_IP_FACTOR: Final[int] = 20
"""The per-IP bucket's burst and refill rate, as a multiple of the per-account ones: coarse
enough that many users behind one address (every user, behind the gateway) are not throttled
together in normal use, tight enough to cap a spray across many emails."""


def normalise_email(email: str) -> str:
    """The form of an email that identifies an account for throttling: trimmed and casefolded."""
    return email.strip().casefold()


def _digest(kind: str, value: str) -> bytes:
    return hashlib.sha256(f"{kind}\x00{value}".encode()).digest()


@dataclass
class _Bucket:
    tokens: float
    updated: float


class _BucketMap:
    """Token buckets keyed by digest, least recently used first, at most `max_keys` of them."""

    def __init__(self, *, burst: float, rate_per_s: float, max_keys: int) -> None:
        self.burst = burst
        self.rate = rate_per_s
        self._max_keys = max_keys
        self._buckets: OrderedDict[bytes, _Bucket] = OrderedDict()

    def refilled(self, key: bytes, now: float) -> _Bucket:
        """The bucket for `key` with tokens refilled up to `now`, created full if absent and
        marked most recently used; creating one may evict another (`_evict`)."""
        bucket = self.existing(key, now)
        if bucket is None:
            bucket = _Bucket(tokens=self.burst, updated=now)
            self._buckets[key] = bucket
            while len(self._buckets) > self._max_keys:
                self._evict(now, keep=key)
        return bucket

    def existing(self, key: bytes, now: float) -> _Bucket | None:
        """The bucket for `key` refilled up to `now` and marked most recently used, or None
        when there is none (nothing is created)."""
        bucket = self._buckets.get(key)
        if bucket is None:
            return None
        elapsed = max(0.0, now - bucket.updated)
        bucket.tokens = min(self.burst, bucket.tokens + elapsed * self.rate)
        bucket.updated = now
        self._buckets.move_to_end(key)
        return bucket

    def _evict(self, now: float, *, keep: bytes) -> None:
        """Drops, among the `EVICTION_WINDOW` least recently used buckets other than `keep` (the
        one just created, which the caller is about to debit), the one that holds the most
        tokens at `now` (the oldest among equals). A full bucket is identical to no bucket, so
        dropping it loses nothing; a drained one is kept while fuller ones exist."""
        victim: bytes | None = None
        most = -1.0
        for index, (key, bucket) in enumerate(self._buckets.items()):
            if index >= EVICTION_WINDOW:
                break
            if key == keep:
                continue
            tokens = min(self.burst, bucket.tokens + max(0.0, now - bucket.updated) * self.rate)
            if tokens > most:
                victim, most = key, tokens
        if victim is not None:
            del self._buckets[victim]

    def wait_s(self, bucket: _Bucket) -> float:
        return 0.0 if bucket.tokens >= 1.0 else (1.0 - bucket.tokens) / self.rate

    def refund(self, key: bytes) -> None:
        bucket = self._buckets.get(key)
        if bucket is not None:
            bucket.tokens = min(self.burst, bucket.tokens + 1.0)

    def keys(self) -> list[bytes]:
        return list(self._buckets)

    def __len__(self) -> int:
        return len(self._buckets)


class LoginRateLimiter:
    """An account bucket per normalised email and a coarse bucket per client IP. `acquire`
    spends one token from each and returns 0.0, or, if either is empty, spends nothing and
    returns the seconds until both have a token. `refund` gives both tokens back after a
    successful login. Not thread-safe; the API calls it from its single event loop, where it
    never awaits. Process-local by design: the API is one local process.

    Trade-off (REVIEW N-04). Every request reaches the API from the gateway container, so the
    client IP cannot tell a local attacker from the real user; keying the account bucket on
    (IP, email) protected nothing and still let any local page lock an account out. So the
    account bucket is keyed on the email alone, and a successful login refunds its token, so
    the real user never pays for signing in. Someone who can reach the endpoint can still keep
    one account's bucket empty, but only while they keep sending failed attempts at the refill
    rate: no lockout state outlives the attack, and the account is usable again one refill
    interval (12 s at the default 5/min) after it stops. The IP bucket is only a coarse flood
    cap (`ip_factor` times the account limits); a spray over many emails can empty it for
    everyone behind the gateway, with the same bound. Telling clients apart for real needs the
    gateway to forward a trusted client address, which is outside this module."""

    def __init__(
        self,
        *,
        burst: int,
        refill_per_minute: float,
        clock: Callable[[], float] = time.monotonic,
        max_keys: int = DEFAULT_MAX_KEYS,
        ip_factor: int = DEFAULT_IP_FACTOR,
        max_ip_keys: int = DEFAULT_MAX_IP_KEYS,
    ) -> None:
        if burst < 1 or refill_per_minute <= 0 or max_keys < 1 or max_ip_keys < 1:
            raise ValueError("burst, refill_per_minute and the key caps must be positive")
        if ip_factor < 1:
            raise ValueError("ip_factor must be at least 1")
        rate = refill_per_minute / 60.0
        self._clock = clock
        self._accounts = _BucketMap(burst=float(burst), rate_per_s=rate, max_keys=max_keys)
        self._ips = _BucketMap(
            burst=float(burst * ip_factor), rate_per_s=rate * ip_factor, max_keys=max_ip_keys
        )

    def acquire(self, client_ip: str, email: str) -> float:
        now = self._clock()
        account_key = _digest("email", normalise_email(email))
        source = self._ips.refilled(_digest("ip", client_ip), now)
        # An absent account bucket would be full, so it is only created for an admitted
        # request: refused ones (cheap, no argon2) never add keys that could evict others.
        account = self._accounts.existing(account_key, now)
        account_wait = 0.0 if account is None else self._accounts.wait_s(account)
        wait = max(account_wait, self._ips.wait_s(source))
        if wait > 0:
            return wait
        if account is None:
            account = self._accounts.refilled(account_key, now)
        account.tokens -= 1.0
        source.tokens -= 1.0
        return 0.0

    def refund(self, client_ip: str, email: str) -> None:
        """Returns the tokens a successful login spent, so signing in never counts against the
        account or the address (only failures do)."""
        self._accounts.refund(_digest("email", normalise_email(email)))
        self._ips.refund(_digest("ip", client_ip))

    def stored_keys(self) -> list[bytes]:
        """Every key held, account and IP buckets alike (for memory-bound tests)."""
        return self._accounts.keys() + self._ips.keys()

    @property
    def ip_bucket_count(self) -> int:
        return len(self._ips)

    def __len__(self) -> int:
        """Number of account buckets held."""
        return len(self._accounts)


def retry_after_header(seconds: float) -> str:
    """`Retry-After` value: whole seconds, rounded up, at least 1."""
    return str(max(1, math.ceil(seconds)))

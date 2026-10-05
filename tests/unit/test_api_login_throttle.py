"""Login hardening (REVIEW L-02, N-04): the per-email bucket and the coarse per-IP bucket throttle
failed attempts with a 429 and `Retry-After`, refill over time and keep memory bounded, over-long
usernames are refused with 422, and an unknown email still costs one full argon2 verify.
The database is replaced by a stub session; the limiter and the route are the real ones.
"""

from __future__ import annotations

import gc
import tracemalloc
import uuid
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apx.api import security
from apx.api import settings as settings_module
from apx.api.app import create_app
from apx.api.deps import get_db_session
from apx.api.ratelimit import MAX_EMAIL_LENGTH, LoginRateLimiter, retry_after_header


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _NoUsersSession:
    """Every email lookup finds nobody."""

    async def scalar(self, _stmt: Any) -> None:
        return None


# ------------------------------------------------------------------------------------------
# LoginRateLimiter
# ------------------------------------------------------------------------------------------


def test_burst_attempts_pass_then_the_next_one_waits() -> None:
    clock = _Clock()
    limiter = LoginRateLimiter(burst=3, refill_per_minute=6, clock=clock)
    assert [limiter.acquire("1.2.3.4", "a@x.test") for _ in range(3)] == [0.0, 0.0, 0.0]
    assert limiter.acquire("1.2.3.4", "a@x.test") == pytest.approx(10.0)  # 6/min -> 10 s


def test_tokens_refill_over_time() -> None:
    clock = _Clock()
    limiter = LoginRateLimiter(burst=1, refill_per_minute=6, clock=clock)
    assert limiter.acquire("ip", "a@x.test") == 0.0
    assert limiter.acquire("ip", "a@x.test") > 0
    clock.now += 10.0
    assert limiter.acquire("ip", "a@x.test") == 0.0


def test_the_account_bucket_is_per_email_across_ips_and_case_insensitive() -> None:
    # REVIEW N-04: behind the gateway every client has the same IP, so the account bucket is
    # keyed on the normalised email alone; another address does not buy fresh attempts.
    limiter = LoginRateLimiter(burst=1, refill_per_minute=1, clock=_Clock())
    assert limiter.acquire("ip1", "a@x.test") == 0.0
    assert limiter.acquire("ip1", " A@X.test ") > 0  # same account, other spelling
    assert limiter.acquire("ip2", "a@x.test") > 0  # same account, other address
    assert limiter.acquire("ip1", "b@x.test") == 0.0


def test_the_ip_bucket_is_coarse_and_caps_a_spray_over_many_emails() -> None:
    clock = _Clock()
    limiter = LoginRateLimiter(burst=2, refill_per_minute=6, clock=clock, ip_factor=10)
    # 20 attempts (10 x burst) across distinct emails pass; the 21st hits the IP cap.
    assert all(limiter.acquire("gw", f"u{i}@x.test") == 0.0 for i in range(20))
    wait = limiter.acquire("gw", "fresh@x.test")
    assert wait == pytest.approx(1.0)  # 60/min at the IP level -> 1 s
    # A different address is unaffected.
    assert limiter.acquire("other", "fresh@x.test") == 0.0


def test_a_throttled_attempt_spends_nothing_from_the_other_bucket() -> None:
    limiter = LoginRateLimiter(burst=1, refill_per_minute=6, clock=_Clock(), ip_factor=1)
    assert limiter.acquire("gw", "a@x.test") == 0.0  # IP bucket now empty
    assert limiter.acquire("gw", "b@x.test") > 0  # refused by the IP bucket...
    assert limiter.acquire("gw2", "b@x.test") == 0.0  # ...without spending b's token


def test_a_successful_login_is_refunded_so_only_failures_count() -> None:
    limiter = LoginRateLimiter(burst=2, refill_per_minute=1, clock=_Clock())
    for _ in range(10):
        assert limiter.acquire("gw", "a@x.test") == 0.0
        limiter.refund("gw", "a@x.test")
    assert limiter.acquire("gw", "a@x.test") == 0.0
    assert limiter.acquire("gw", "a@x.test") == 0.0
    assert limiter.acquire("gw", "a@x.test") > 0


def test_a_refund_never_overfills_a_bucket() -> None:
    limiter = LoginRateLimiter(burst=1, refill_per_minute=1, clock=_Clock())
    limiter.refund("gw", "a@x.test")  # unknown key: a no-op
    assert limiter.acquire("gw", "a@x.test") == 0.0
    limiter.refund("gw", "a@x.test")
    limiter.refund("gw", "a@x.test")
    assert limiter.acquire("gw", "a@x.test") == 0.0
    assert limiter.acquire("gw", "a@x.test") > 0


def test_the_lockout_ends_one_refill_interval_after_the_attack_stops() -> None:
    clock = _Clock()
    limiter = LoginRateLimiter(burst=5, refill_per_minute=5, clock=clock)
    for _ in range(50):  # an attacker hammering the victim's account
        limiter.acquire("gw", "victim@x.test")
        clock.now += 1.0
    assert limiter.acquire("gw", "victim@x.test") > 0
    clock.now += 12.0  # one token at 5/min
    assert limiter.acquire("gw", "victim@x.test") == 0.0


def test_memory_is_bounded_by_max_keys() -> None:
    limiter = LoginRateLimiter(
        burst=1, refill_per_minute=1, clock=_Clock(), max_keys=100, max_ip_keys=10
    )
    for i in range(1000):
        limiter.acquire(f"ip{i}", f"user{i}@x.test")
    assert len(limiter) == 100
    assert limiter.ip_bucket_count == 10


def test_a_refused_attempt_creates_no_account_bucket() -> None:
    """P3-03: a request the IP bucket refuses costs the server nothing (no argon2), so it must
    not add account keys either; otherwise a cheap spray churns the account map."""
    limiter = LoginRateLimiter(burst=1, refill_per_minute=1, clock=_Clock(), ip_factor=1)
    assert limiter.acquire("gw", "first@x.test") == 0.0
    for i in range(100):
        assert limiter.acquire("gw", f"spray{i}@x.test") > 0
    assert len(limiter) == 1


def test_a_username_spray_cannot_flush_a_drained_account_bucket() -> None:
    """P3-03: the attacker drains the target, then sends attempts at many other usernames to
    push its bucket out of the capped map. It must stay drained rather than come back full."""
    clock = _Clock()
    limiter = LoginRateLimiter(
        burst=5, refill_per_minute=5, clock=clock, max_keys=10, ip_factor=10_000
    )
    target = "victim@x.test"
    for _ in range(5):
        assert limiter.acquire("attacker", target) == 0.0
    assert limiter.acquire("attacker", target) > 0
    for i in range(1_000):
        limiter.acquire(f"ip{i % 7}", f"spray{i}@x.test")
    assert limiter.acquire("attacker", target) > 0, "the drained bucket was evicted and refilled"
    assert len(limiter) == 10


def test_keys_are_fixed_size_digests_whatever_the_username() -> None:
    limiter = LoginRateLimiter(burst=1, refill_per_minute=1, clock=_Clock())
    limiter.acquire("gw", "x" * 1_000_000)
    assert all(isinstance(k, bytes) and len(k) == 32 for k in limiter.stored_keys())


def test_memory_stays_bounded_under_a_spray_of_maximal_usernames() -> None:
    # REVIEW N-04 measured ~1 MB per bucket with 1 MB usernames. Now each bucket holds a
    # 32-byte digest: 2,000 buckets from 320-char emails stay far under 1 MB in total.
    # A large IP factor admits every attempt from "gw": only admitted attempts create account
    # buckets (P3-03), and this test is about the cap on those.
    limiter = LoginRateLimiter(
        burst=1, refill_per_minute=1, clock=_Clock(), max_keys=2_000, ip_factor=20_000
    )
    tracemalloc.start()
    try:
        before, _ = tracemalloc.get_traced_memory()
        for i in range(10_000):
            limiter.acquire("gw", f"{i:0>308}@example.test")
        gc.collect()
        after, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert len(limiter) == 2_000
    assert after - before < 1_000_000


def test_invalid_configuration_is_refused() -> None:
    with pytest.raises(ValueError):
        LoginRateLimiter(burst=0, refill_per_minute=1)


def test_retry_after_is_whole_seconds_rounded_up() -> None:
    assert retry_after_header(0.2) == "1"
    assert retry_after_header(9.01) == "10"


# ------------------------------------------------------------------------------------------
# POST /auth/login
# ------------------------------------------------------------------------------------------


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setenv("API_JWT_SECRET", "x" * 48)
    monkeypatch.setenv("API_LOGIN_BURST", "3")
    monkeypatch.setenv("API_LOGIN_REFILL_PER_MINUTE", "1")
    settings_module.get_settings.cache_clear()
    app = create_app()

    async def _session() -> AsyncIterator[_NoUsersSession]:
        yield _NoUsersSession()

    app.dependency_overrides[get_db_session] = _session
    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        settings_module.get_settings.cache_clear()


def _login(client: TestClient, email: str) -> int:
    response = client.post("/auth/login", data={"username": email, "password": "guess"})
    if response.status_code == 429:
        assert int(response.headers["Retry-After"]) >= 1
    return response.status_code


def test_login_is_throttled_after_the_configured_burst(client: TestClient) -> None:
    assert [_login(client, "victim@x.test") for _ in range(4)] == [401, 401, 401, 429]
    # Another account from the same client is not locked out by the first one's bucket.
    assert _login(client, "other@x.test") == 401


def test_unknown_email_still_runs_a_full_argon2_verify(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified: list[str] = []
    real_verify = security.verify_password

    def spy(password_hash: str, password: str) -> bool:
        verified.append(password_hash)
        return real_verify(password_hash, password)

    monkeypatch.setattr(security, "verify_password", spy)
    assert _login(client, "nobody@x.test") == 401
    assert verified == [security.dummy_hash()]
    assert verified[0].startswith("$argon2")


def test_verify_login_never_accepts_a_missing_hash() -> None:
    assert security.verify_login(None, "") is False
    assert security.verify_login(None, "anything") is False
    good = security.hash_password("pw")
    assert security.verify_login(good, "pw") is True
    assert security.verify_login(good, "nope") is False


def test_the_limiter_is_one_per_app(client: TestClient) -> None:
    _login(client, "a@x.test")
    app = client.app
    limiter = app.state.login_limiter  # type: ignore[attr-defined]
    assert isinstance(limiter, LoginRateLimiter)
    assert len(limiter) == 1


def test_an_over_long_username_is_422_before_the_limiter_or_argon2(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    verified: list[str] = []

    def spy(password_hash: str, _password: str) -> bool:
        verified.append(password_hash)
        return False

    monkeypatch.setattr(security, "verify_password", spy)
    email = "a" * (MAX_EMAIL_LENGTH - len("@x.test") + 1) + "@x.test"
    assert len(email) == MAX_EMAIL_LENGTH + 1
    response = client.post("/auth/login", data={"username": email, "password": "guess"})
    assert response.status_code == 422
    assert verified == []
    assert getattr(client.app.state, "login_limiter", None) is None  # type: ignore[attr-defined]
    # Exactly the RFC 5321 maximum is still an ordinary (failed) login.
    assert _login(client, email[1:]) == 401


class _User:
    def __init__(self, password: str) -> None:
        self.id = uuid.uuid4()
        self.email = "approver@x.test"
        self.is_active = True
        self.password_hash = security.hash_password(password)


class _Rows:
    def scalars(self) -> _Rows:
        return self

    def all(self) -> list[str]:
        return []


class _OneUserSession:
    def __init__(self, user: _User) -> None:
        self._user = user

    async def scalar(self, _stmt: Any) -> _User:
        return self._user

    async def execute(self, _stmt: Any) -> _Rows:
        return _Rows()


def test_successful_logins_do_not_drain_the_account_bucket(client: TestClient) -> None:
    user = _User("right")
    app = client.app

    async def _session() -> AsyncIterator[_OneUserSession]:
        yield _OneUserSession(user)

    app.dependency_overrides[get_db_session] = _session  # type: ignore[attr-defined]
    ok = {"username": user.email, "password": "right"}
    # Burst is 3: ten successful sign-ins in a row are never throttled.
    assert [client.post("/auth/login", data=ok).status_code for _ in range(10)] == [200] * 10
    bad = {"username": user.email, "password": "wrong"}
    assert [client.post("/auth/login", data=bad).status_code for _ in range(4)] == [
        401,
        401,
        401,
        429,
    ]


def test_a_new_bucket_is_never_the_one_evicted_to_make_room() -> None:
    # REVIEW.md P4-05: with fewer keys than the eviction window, the bucket just created was the
    # fullest in the window, so it was dropped and the caller debited an orphan (no throttle).
    from apx.api.ratelimit import _BucketMap

    buckets = _BucketMap(burst=5, rate_per_s=1.0, max_keys=2)
    for key in (b"a", b"b", b"c"):
        bucket = buckets.refilled(key, now=0.0)
        bucket.tokens -= 1
        assert buckets.existing(key, now=0.0) is bucket

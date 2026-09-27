"""Login hardening (REVIEW L-02): the per-(IP, email) token bucket throttles repeated attempts with
a 429 and `Retry-After`, refills over time, keeps its memory bounded, and an unknown email still
costs one full argon2 verify, so response timing does not reveal which emails are registered.
The database is replaced by a stub session; the limiter and the route are the real ones.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from apx.api import security
from apx.api import settings as settings_module
from apx.api.app import create_app
from apx.api.deps import get_db_session
from apx.api.ratelimit import LoginRateLimiter, retry_after_header


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


def test_buckets_are_per_ip_and_per_email_and_email_is_case_insensitive() -> None:
    limiter = LoginRateLimiter(burst=1, refill_per_minute=1, clock=_Clock())
    assert limiter.acquire("ip1", "a@x.test") == 0.0
    assert limiter.acquire("ip1", " A@X.test ") > 0  # same account, other spelling
    assert limiter.acquire("ip1", "b@x.test") == 0.0
    assert limiter.acquire("ip2", "a@x.test") == 0.0


def test_memory_is_bounded_by_max_keys() -> None:
    limiter = LoginRateLimiter(burst=1, refill_per_minute=1, clock=_Clock(), max_keys=100)
    for i in range(1000):
        limiter.acquire("ip", f"user{i}@x.test")
    assert len(limiter) == 100


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

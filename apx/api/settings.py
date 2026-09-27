"""API configuration read from environment variables via pydantic-settings: the JWT signing
secret, token lifetime, CORS origin, login throttle and Temporal address (PRD §8, §10 M5.3). Every
variable is documented here and must be mirrored in `.env.example`.
"""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from typing import Any

from pydantic import Field, PrivateAttr, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from apx.policy.approval_limit import DEFAULT_APPROVAL_LIMIT, load_approval_limit

MIN_JWT_SECRET_CHARS = 32


class ApiSettings(BaseSettings):
    """Process-wide API settings. `API_JWT_SECRET` is required and has no default so a
    production deployment cannot start with a guessable HS256 key."""

    model_config = SettingsConfigDict(env_prefix="API_", extra="ignore")

    jwt_secret: SecretStr = Field(min_length=MIN_JWT_SECRET_CHARS)
    """`API_JWT_SECRET`: HS256 signing key for access tokens. Required and at least 32 characters
    (RFC 7518 §3.2: an HS256 key must be at least 256 bits), so an empty or placeholder value
    fails at start-up instead of signing forgeable tokens. `make up` generates one into `.env`
    (scripts/init_env.py). Never committed (§8)."""

    jwt_algorithm: str = "HS256"
    """`API_JWT_ALGORITHM`: JWT signing algorithm. HS256 only; the API never accepts `alg=none`
    or an asymmetric algorithm it was not configured for."""

    access_token_minutes: int = Field(default=15, gt=0)
    """`API_ACCESS_TOKEN_MINUTES`: access token lifetime in minutes. Kept short so a leaked
    token self-expires quickly; there is no refresh token in M5.3."""

    cors_origin: str = "http://localhost:3000"
    """`API_CORS_ORIGIN`: the single web origin allowed to call this API (`web/`, PRD §10
    M5.4). CORS is locked to exactly this origin, never `*`."""

    _approval_limit: Decimal = PrivateAttr(default=DEFAULT_APPROVAL_LIMIT)

    login_burst: int = Field(default=5, ge=1)
    """`API_LOGIN_BURST`: login attempts one client IP may make for one email before it is
    throttled (token-bucket capacity, REVIEW L-02)."""

    login_refill_per_minute: float = Field(default=5.0, gt=0)
    """`API_LOGIN_REFILL_PER_MINUTE`: attempts regained per minute once throttled."""

    sod_cooldown_days: int = Field(default=30, ge=0)
    """`API_SOD_COOLDOWN_DAYS`: days after a bank-detail change during which its author cannot
    approve that vendor's payments (P2, §6.3)."""

    temporal_address: str | None = None
    """`API_TEMPORAL_ADDRESS`: Temporal frontend (e.g. `localhost:7233`) used to signal a
    reviewer's decision to the invoice's workflow. Unset means review decisions fail loudly
    with a configuration error instead of leaving the workflow waiting (PRD §10 M5.3)."""

    def model_post_init(self, context: Any, /) -> None:
        self._approval_limit = load_approval_limit().amount

    @property
    def approval_limit(self) -> Decimal:
        """Amount above which an approval needs the approver role's `approve_above_limit`
        capability (P3, §6.3). Read once from `APPROVAL_LIMIT`, the one setting the worker's
        routing also reads (`apx.policy.approval_limit`, REVIEW L-03)."""
        return self._approval_limit


@lru_cache(maxsize=1)
def get_settings() -> ApiSettings:
    """Process-wide settings, created once and reused. Tests call `get_settings.cache_clear()`
    after `monkeypatch.setenv` to pick up a fresh environment."""
    return ApiSettings()  # type: ignore[call-arg]  # populated from the environment, not here

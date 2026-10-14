"""Unit tests for password hashing and JWT access tokens: argon2 verification,
token round-tripping, expiry and tampering. No database or network needed."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
import pytest
from pydantic import ValidationError as PydanticValidationError

from apx.api.security import (
    JwtTokenVerifier,
    create_access_token,
    hash_password,
    verify_password,
)
from apx.api.settings import ApiSettings
from apx.common.errors import PolicyDeniedError
from apx.common.types import Role, UserId


def _settings(**overrides: object) -> ApiSettings:
    overrides.setdefault("jwt_secret", "unit-test-secret-value-long-enough-for-hs256")
    return ApiSettings(**overrides)  # type: ignore[arg-type]


def test_hash_password_round_trips_through_verify_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password(hashed, "correct horse battery staple") is True


def test_verify_password_rejects_a_wrong_password() -> None:
    hashed = hash_password("correct horse battery staple")
    assert verify_password(hashed, "wrong password") is False


def test_verify_password_rejects_a_malformed_hash_without_raising() -> None:
    assert verify_password("not-an-argon2-hash", "anything") is False


def test_access_token_round_trips_sub_and_roles() -> None:
    settings = _settings()
    roles = frozenset({Role.REVIEWER, Role.APPROVER})
    token, expires_at = create_access_token(user_id=UserId("u1"), roles=roles, settings=settings)

    claims = JwtTokenVerifier(settings).verify(token)
    assert claims.sub == "u1"
    assert claims.roles == roles
    assert abs((claims.exp - expires_at).total_seconds()) < 1


def test_verify_rejects_a_token_signed_with_a_different_secret() -> None:
    settings = _settings()
    token, _ = create_access_token(user_id=UserId("u1"), roles=frozenset(), settings=settings)

    other_settings = _settings(jwt_secret="a-completely-different-secret-of-hs256-length")
    with pytest.raises(PolicyDeniedError):
        JwtTokenVerifier(other_settings).verify(token)


def test_verify_rejects_an_expired_token() -> None:
    settings = _settings()
    expired = jwt.encode(
        {
            "sub": "u1",
            "roles": ["reviewer"],
            "exp": datetime.now(UTC) - timedelta(minutes=1),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(PolicyDeniedError):
        JwtTokenVerifier(settings).verify(expired)


def test_verify_rejects_an_unknown_role_claim() -> None:
    settings = _settings()
    bad_token = jwt.encode(
        {
            "sub": "u1",
            "roles": ["superuser"],
            "exp": datetime.now(UTC) + timedelta(minutes=5),
        },
        settings.jwt_secret.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )
    with pytest.raises(PolicyDeniedError):
        JwtTokenVerifier(settings).verify(bad_token)


@pytest.mark.parametrize("secret", ["", "too-short-for-hs256"])
def test_settings_reject_an_empty_or_short_jwt_secret(secret: str) -> None:
    with pytest.raises(PydanticValidationError, match="jwt_secret"):
        _settings(jwt_secret=secret)

"""Password hashing (argon2) and JWT access tokens (PyJWT, HS256): issues short-lived tokens at
login and verifies them on every request via `JwtTokenVerifier` (PRD §10 M5.3, apx.api.interfaces
.TokenVerifier).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHash, VerificationError, VerifyMismatchError

from apx.api.interfaces import TokenClaims
from apx.api.settings import ApiSettings
from apx.common.errors import PolicyDeniedError
from apx.common.types import Role, UserId

_password_hasher = PasswordHasher()


def hash_password(password: str) -> str:
    """Hash a plaintext password with argon2id for storage in `users.password_hash`."""
    return _password_hasher.hash(password)


def verify_password(password_hash: str, password: str) -> bool:
    """Check `password` against a stored argon2 hash. Never raises on a wrong password or a
    malformed hash; both simply fail authentication."""
    try:
        return _password_hasher.verify(password_hash, password)
    except (VerifyMismatchError, VerificationError, InvalidHash):
        return False


def create_access_token(
    *, user_id: UserId, roles: frozenset[Role], settings: ApiSettings
) -> tuple[str, datetime]:
    """Issue a short-lived HS256 access token carrying `sub` and `roles`. Returns the encoded
    token and its expiry, so the login endpoint can report `expires_in` without decoding it
    back."""
    expires_at = datetime.now(UTC) + timedelta(minutes=settings.access_token_minutes)
    payload = {
        "sub": str(user_id),
        "roles": sorted(role.value for role in roles),
        "exp": expires_at,
    }
    token = jwt.encode(
        payload, settings.jwt_secret.get_secret_value(), algorithm=settings.jwt_algorithm
    )
    return token, expires_at


class JwtTokenVerifier:
    """`apx.api.interfaces.TokenVerifier` backed by PyJWT. Verification never touches the
    database: a token's `roles` claim is authoritative for the lifetime of that (short-lived)
    token, so authorisation checks below never need a per-request database round trip for
    identity."""

    def __init__(self, settings: ApiSettings) -> None:
        self._settings = settings

    def verify(self, token: str) -> TokenClaims:
        try:
            payload = jwt.decode(
                token,
                self._settings.jwt_secret.get_secret_value(),
                algorithms=[self._settings.jwt_algorithm],
            )
        except jwt.InvalidTokenError as exc:
            raise PolicyDeniedError(f"invalid or expired token: {exc}") from exc

        try:
            return TokenClaims(
                sub=UserId(str(payload["sub"])),
                roles=frozenset(Role(r) for r in payload["roles"]),
                exp=datetime.fromtimestamp(payload["exp"], tz=UTC),
            )
        except (KeyError, ValueError) as exc:
            raise PolicyDeniedError(f"malformed token claims: {exc}") from exc

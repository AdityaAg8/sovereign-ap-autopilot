"""`POST /auth/login`: verifies an email/password pair against `users`/`user_roles` and issues a
short-lived HS256 JWT carrying the user's roles (PRD §10 M5.3, docs/security.md B8). Attempts are
throttled per (client IP, email) and an unknown email costs a full argon2 verify, so neither
guessing nor timing reveals which accounts exist (REVIEW L-02)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from apx.api.deps import DbSession, SettingsDep
from apx.api.interfaces import TokenResponse
from apx.api.ratelimit import LoginRateLimiter, retry_after_header
from apx.api.security import create_access_token, verify_login
from apx.api.settings import ApiSettings
from apx.common.types import Role, UserId
from apx.db.models import RoleDef, User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])

LoginForm = Annotated[OAuth2PasswordRequestForm, Depends()]


def login_limiter(request: Request, settings: ApiSettings) -> LoginRateLimiter:
    """The app's one limiter, created on first use from its settings and kept on `app.state`,
    so each app instance (and each test's app) has its own buckets."""
    limiter = getattr(request.app.state, "login_limiter", None)
    if not isinstance(limiter, LoginRateLimiter):
        limiter = LoginRateLimiter(
            burst=settings.login_burst, refill_per_minute=settings.login_refill_per_minute
        )
        request.app.state.login_limiter = limiter
    return limiter


@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form: LoginForm,
    session: DbSession,
    settings: SettingsDep,
) -> TokenResponse:
    client_ip = request.client.host if request.client is not None else "unknown"
    wait_s = login_limiter(request, settings).acquire(client_ip, form.username)
    if wait_s > 0:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="too many login attempts; try again later",
            headers={"Retry-After": retry_after_header(wait_s)},
        )

    user = await session.scalar(select(User).where(User.email == form.username))
    # Always one argon2 verify, known email or not, active or not (constant work, L-02).
    password_ok = verify_login(None if user is None else user.password_hash, form.password)
    if user is None or not user.is_active or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role_stmt = (
        select(RoleDef.name)
        .join(UserRole, UserRole.role_id == RoleDef.id)
        .where(UserRole.user_id == user.id)
    )
    role_names = (await session.execute(role_stmt)).scalars().all()
    roles = frozenset(Role(name) for name in role_names)

    token, _expires_at = create_access_token(
        user_id=UserId(user.id), roles=roles, settings=settings
    )
    return TokenResponse(access_token=token, expires_in=settings.access_token_minutes * 60)

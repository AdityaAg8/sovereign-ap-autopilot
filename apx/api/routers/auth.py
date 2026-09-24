"""`POST /auth/login`: verifies an email/password pair against `users`/`user_roles` and issues a
short-lived HS256 JWT carrying the user's roles (PRD §10 M5.3, docs/security.md B8)."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from apx.api.deps import DbSession, SettingsDep
from apx.api.interfaces import TokenResponse
from apx.api.security import create_access_token, verify_password
from apx.common.types import Role, UserId
from apx.db.models import RoleDef, User, UserRole

router = APIRouter(prefix="/auth", tags=["auth"])

LoginForm = Annotated[OAuth2PasswordRequestForm, Depends()]


@router.post("/login", response_model=TokenResponse)
async def login(
    form: LoginForm,
    session: DbSession,
    settings: SettingsDep,
) -> TokenResponse:
    user = await session.scalar(select(User).where(User.email == form.username))
    if user is None or not user.is_active or not verify_password(user.password_hash, form.password):
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

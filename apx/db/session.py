"""Async SQLAlchemy engine and session factory for Postgres, driven by the `DATABASE_URL`
environment variable (asyncpg only; psycopg is not used, PRD §7 apx/db, §10 M5.1)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apx.common.errors import ConfigError

DEFAULT_DATABASE_URL = "postgresql+asyncpg://apx:apx-dev-only@localhost:5432/apx"
"""Matches the local Compose `postgres` service's default `POSTGRES_PASSWORD`
(infra/compose.yaml); overridden in every other environment by the `DATABASE_URL` env var."""


def get_database_url() -> str:
    """Reads `DATABASE_URL` from the environment, falling back to the local Compose default.

    Raises `apx.common.errors.ConfigError` if the URL is set but does not use the `asyncpg`
    driver (psycopg is blocked on this host; the async engine requires an async driver).
    """
    url = os.environ.get("DATABASE_URL", DEFAULT_DATABASE_URL)
    if not url.startswith("postgresql+asyncpg://"):
        raise ConfigError(f"DATABASE_URL must use the postgresql+asyncpg:// driver, got: {url!r}")
    return url


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    """Process-wide async engine, created once and reused (PRD §8: no per-request engines)."""
    return create_async_engine(get_database_url(), pool_pre_ping=True)


@lru_cache(maxsize=1)
def _session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(bind=get_engine(), expire_on_commit=False)


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """Yields an `AsyncSession`, committing on clean exit and rolling back on error."""
    session = _session_factory()()
    try:
        yield session
        await session.commit()
    except BaseException:
        await session.rollback()
        raise
    finally:
        await session.close()

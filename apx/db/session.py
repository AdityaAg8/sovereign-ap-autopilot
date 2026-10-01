"""Async SQLAlchemy engine and session factory for Postgres, driven by the `DATABASE_URL`
environment variable (asyncpg only; psycopg is not used, PRD §7 apx/db, §10 M5.1)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache
from urllib.parse import quote

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apx.common.errors import ConfigError

MISSING_URL_MESSAGE = (
    "no database configured: set DATABASE_URL (postgresql+asyncpg://...) or POSTGRES_PASSWORD "
    "(generated into .env by `make env`); there is no built-in default password"
)


def get_database_url() -> str:
    """Reads `DATABASE_URL` from the environment, else builds it from the Compose `POSTGRES_*`
    settings. There is no baked-in default credential (REVIEW C-02).

    Raises `apx.common.errors.ConfigError` when neither `DATABASE_URL` nor `POSTGRES_PASSWORD`
    is set, or when the URL does not use the `asyncpg` driver (psycopg is blocked on this host;
    the async engine requires an async driver).
    """
    url = os.environ.get("DATABASE_URL") or _compose_url()
    if not url.startswith("postgresql+asyncpg://"):
        raise ConfigError(f"DATABASE_URL must use the postgresql+asyncpg:// driver, got: {url!r}")
    return url


def _compose_url() -> str:
    """The Compose `postgres` service, from the same `POSTGRES_USER`/`POSTGRES_PASSWORD`/
    `POSTGRES_DB` overrides Compose reads from `.env`, so every process sees the database Compose
    actually created. `POSTGRES_HOST` is `postgres` inside the `app` network; on the host it
    defaults to localhost (the gateway's passthrough)."""
    if not os.environ.get("POSTGRES_PASSWORD"):
        raise ConfigError(MISSING_URL_MESSAGE)
    user = quote(os.environ.get("POSTGRES_USER") or "apx", safe="")
    password = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    database = quote(os.environ.get("POSTGRES_DB") or "apx", safe="")
    host = os.environ.get("POSTGRES_HOST") or "localhost"
    return f"postgresql+asyncpg://{user}:{password}@{host}:5432/{database}"


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

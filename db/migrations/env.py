"""Alembic async migration runner for the Postgres schema in `apx/db/models.py`. Resolves the
connection URL like the app does (`apx.db.session.get_database_url`, else a non-empty
`alembic.ini` value) and runs migrations through an `asyncpg` engine so the same driver is used
in tests and production."""

from __future__ import annotations

import asyncio
import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from apx.db.base import Base
from apx.db.models import *  # noqa: F403 -- registers every table on Base.metadata
from apx.db.session import get_database_url

# Alembic Config object; provides access to values in alembic.ini.
config = context.config

# The app's own resolution (`DATABASE_URL`, else the Compose `POSTGRES_*` settings) wins over
# the static `alembic.ini` value, so migrations reach the same database the app does -- a
# throwaway test container, the host via the gateway, or `postgres` inside the Compose network.
# `alembic.ini` ships an empty URL, so with nothing configured `get_database_url` raises a typed
# `ConfigError` instead of connecting with a baked-in password.
if (
    os.environ.get("DATABASE_URL")
    or os.environ.get("POSTGRES_PASSWORD")
    or not config.get_main_option("sqlalchemy.url")
):
    config.set_main_option("sqlalchemy.url", get_database_url())

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (`alembic upgrade head --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Creates an asyncpg engine and runs the sync migration context over it via
    `AsyncConnection.run_sync` (Alembic's async cookbook pattern)."""
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

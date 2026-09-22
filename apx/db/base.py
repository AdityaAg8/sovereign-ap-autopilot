"""Declarative base and shared column mixins for every ORM model in `apx.db.models`. A fixed
naming convention keeps constraint and index names stable across Alembic autogenerate runs
(PRD §7 apx/db, §10 M5.1).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Deterministic names for constraints Postgres would otherwise name automatically. Required
# so a hand-written `downgrade()` can drop a named CHECK/FK without guessing its name, and so
# repeated `alembic upgrade head` / `downgrade base` cycles are reproducible.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    """Shared declarative base; `apx.db.models` defines every mapped table on top of this."""

    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class CreatedAtMixin:
    """Adds a server-side `created_at` timestamp set at insert time."""

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

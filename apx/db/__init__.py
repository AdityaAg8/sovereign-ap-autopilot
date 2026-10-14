"""Postgres persistence layer: SQLAlchemy 2.0 typed models, an async session factory and the
declarative base migrated by `db/migrations`."""

from __future__ import annotations

from apx.db.base import Base

__all__ = ["Base"]

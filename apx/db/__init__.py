"""Postgres persistence layer: SQLAlchemy 2.0 typed models, an async session factory and the
declarative base migrated by `db/migrations` (PRD §7 apx/db, §10 M5.1)."""

from __future__ import annotations

from apx.db.base import Base

__all__ = ["Base"]

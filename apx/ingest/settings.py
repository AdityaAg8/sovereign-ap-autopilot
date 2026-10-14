"""Environment-driven configuration for `apx.ingest`: the watch-folder and content-addressed
store locations, and the upload hardening limits every ingest path enforces.
`max_bytes`/`max_pages` default to PRD §10 M5.3's 20 MB / 30 pages, the only upload limits
(the API upload route and the inbox watcher both enforce them through `Ingestor`)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestSettings(BaseSettings):
    """Process-wide ingest settings for the watcher and (via `apx.api.deps`) the API."""

    model_config = SettingsConfigDict(env_prefix="INGEST_", extra="ignore")

    inbox_dir: Path = Path("data/inbox")
    """`INGEST_INBOX_DIR`: folder watched (via `watchfiles`) for newly dropped documents."""

    store_dir: Path = Path("data/documents")
    """`INGEST_STORE_DIR`: content-addressed store for every ingested document, keyed by
    sha256."""

    max_bytes: int = Field(default=20 * 1024 * 1024, gt=0)
    """`INGEST_MAX_BYTES`: maximum accepted document size in bytes."""

    max_pages: int = Field(default=30, gt=0)
    """`INGEST_MAX_PAGES`: maximum accepted PDF page count."""


@lru_cache(maxsize=1)
def get_ingest_settings() -> IngestSettings:
    """Process-wide settings, created once and reused. Tests call `get_ingest_settings.
    cache_clear()` after `monkeypatch.setenv` to pick up a fresh environment."""
    return IngestSettings()

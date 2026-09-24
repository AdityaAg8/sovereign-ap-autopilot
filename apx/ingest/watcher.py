"""`data/inbox/` watcher (watchfiles): ingests every file dropped into the watch folder through
the same `Ingestor.ingest` function the API upload path calls (PRD §10 M5.5), so a document that
lands in the inbox and is also uploaded (or is re-saved by an editor, or copied in twice) starts
its `InvoiceWorkflow` exactly once.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from watchfiles import Change, awatch

from apx.common.errors import ApxError
from apx.ingest.interfaces import Ingestor

logger = logging.getLogger(__name__)

_DEFAULT_SETTLE_CHECKS = 3
_DEFAULT_SETTLE_INTERVAL_S = 0.1


async def _read_when_stable(
    path: Path,
    *,
    checks: int = _DEFAULT_SETTLE_CHECKS,
    interval_s: float = _DEFAULT_SETTLE_INTERVAL_S,
) -> bytes | None:
    """Polls `path`'s size up to `checks` times, `interval_s` apart, and reads it as soon as two
    consecutive polls agree -- so a file still being copied or written into the inbox is never
    ingested half-written. Returns `None` if the file disappears, or never settles within the
    poll budget (a later filesystem event will retry it)."""
    previous_size: int | None = None
    for _ in range(checks):
        try:
            current_size = path.stat().st_size
        except FileNotFoundError:
            return None
        if current_size == previous_size:
            try:
                return path.read_bytes()
            except FileNotFoundError:
                return None
        previous_size = current_size
        await asyncio.sleep(interval_s)
    return None


async def watch_inbox(inbox_dir: Path, ingestor: Ingestor) -> None:
    """Runs until cancelled, ingesting (with `source="watch"`) every file added to or modified
    in `inbox_dir`. A document that fails ingest hardening (unsupported type, oversized, too
    many pages, encrypted) is logged and left in place rather than crashing the watcher; every
    other failure propagates so the process supervisor can restart it."""
    inbox_dir.mkdir(parents=True, exist_ok=True)
    async for changes in awatch(inbox_dir):
        for change, raw_path in changes:
            if change is Change.deleted:
                continue
            path = Path(raw_path)
            if not path.is_file():
                continue
            data = await _read_when_stable(path)
            if data is None:
                continue
            try:
                await ingestor.ingest(data, path.name, "watch")
            except ApxError:
                logger.exception("failed to ingest %s from the inbox watcher", path)

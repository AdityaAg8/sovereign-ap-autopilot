"""Unit tests for `apx.ingest.watcher`: the `data/inbox/` watcher reads a dropped file only once
its size has settled, ingests it through the injected `Ingestor` with `source="watch"`, and
never crashes the watch loop on a hardening rejection (PRD §10 M5.5). No database is used: the
`Ingestor` collaborator is a fake that only records calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from apx.common.errors import IngestError
from apx.ingest.interfaces import IngestedDocument, IngestResult
from apx.ingest.watcher import _read_when_stable, watch_inbox


class _FakeIngestor:
    def __init__(self, *, fail: bool = False) -> None:
        self.calls: list[tuple[bytes, str, str]] = []
        self._fail = fail

    async def ingest(
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestResult:
        self.calls.append((data, filename, source))
        if self._fail:
            raise IngestError("boom")
        document = IngestedDocument(
            sha256="0" * 64,
            filename=filename,
            media_type="application/pdf",
            size_bytes=len(data),
            received_at=datetime.now(UTC),
            source=source,
        )
        return IngestResult(document=document, duplicate=False, workflow_id="wf-1")


async def _run_until(
    predicate: Callable[[], bool], *, timeout_s: float = 5.0, interval_s: float = 0.05
) -> None:
    async def _poll() -> None:
        while not predicate():
            await asyncio.sleep(interval_s)

    await asyncio.wait_for(_poll(), timeout=timeout_s)


# ------------------------------------------------------------------------------------------
# _read_when_stable
# ------------------------------------------------------------------------------------------


async def test_read_when_stable_returns_bytes_of_an_already_settled_file(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"%PDF-1.4 hello")
    assert await _read_when_stable(path, checks=2, interval_s=0.01) == b"%PDF-1.4 hello"


async def test_read_when_stable_returns_none_for_a_missing_file(tmp_path: Path) -> None:
    assert await _read_when_stable(tmp_path / "missing.pdf", checks=2, interval_s=0.01) is None


async def test_read_when_stable_waits_out_a_growing_file(tmp_path: Path) -> None:
    path = tmp_path / "invoice.pdf"
    path.write_bytes(b"partial")

    async def _grow() -> None:
        # Much shorter than `interval_s` below, so the growth reliably lands inside the first
        # poll window regardless of event-loop scheduling jitter between the two sleeps.
        await asyncio.sleep(0.005)
        path.write_bytes(b"partial-more-bytes-now-final")

    grower = asyncio.create_task(_grow())
    try:
        data = await _read_when_stable(path, checks=6, interval_s=0.05)
    finally:
        await grower
    assert data == b"partial-more-bytes-now-final"


# ------------------------------------------------------------------------------------------
# watch_inbox
# ------------------------------------------------------------------------------------------


async def test_watch_inbox_ingests_a_dropped_file_with_source_watch(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    ingestor = _FakeIngestor()
    task = asyncio.create_task(watch_inbox(inbox, ingestor))
    try:
        await _run_until(lambda: inbox.is_dir(), timeout_s=5.0)
        (inbox / "invoice.pdf").write_bytes(b"%PDF-1.4 fake invoice bytes")
        await _run_until(lambda: len(ingestor.calls) == 1, timeout_s=10.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    data, filename, source = ingestor.calls[0]
    assert data == b"%PDF-1.4 fake invoice bytes"
    assert filename == "invoice.pdf"
    assert source == "watch"


async def test_watch_inbox_logs_and_continues_on_a_hardening_rejection(tmp_path: Path) -> None:
    inbox = tmp_path / "inbox"
    ingestor = _FakeIngestor(fail=True)
    task = asyncio.create_task(watch_inbox(inbox, ingestor))
    try:
        await _run_until(lambda: inbox.is_dir(), timeout_s=5.0)
        (inbox / "bad.pdf").write_bytes(b"not really a pdf")
        await _run_until(lambda: len(ingestor.calls) == 1, timeout_s=10.0)
        # The watcher must still be running afterwards, not have crashed on the typed error.
        await asyncio.sleep(0.1)
        assert not task.done()
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


async def test_watch_inbox_ingests_files_already_waiting_at_startup(tmp_path: Path) -> None:
    """Files dropped while the worker was down (a crash, a restart) raise no filesystem event
    once it is back, so the watcher sweeps the inbox when it starts; ingest dedupes by sha256,
    so a file that was already ingested does not start a second workflow."""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    (inbox / "a.pdf").write_bytes(b"%PDF-1.4 waiting a")
    (inbox / "b.pdf").write_bytes(b"%PDF-1.4 waiting b")
    (inbox / "sub").mkdir()
    ingestor = _FakeIngestor()
    task = asyncio.create_task(watch_inbox(inbox, ingestor))
    try:
        await _run_until(lambda: len(ingestor.calls) == 2, timeout_s=10.0)
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    assert sorted((name, source) for _data, name, source in ingestor.calls) == [
        ("a.pdf", "watch"),
        ("b.pdf", "watch"),
    ]

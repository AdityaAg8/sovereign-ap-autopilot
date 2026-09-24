"""Unit tests for `apx.asp_mock.repository`: both the in-memory and SQLite-backed
`AspRepository` implementations return the same record for a replayed key, raise on a
conflicting payload, and build a record exactly once under 20 concurrent duplicate calls for the
same key (PRD §3 apx/asp_mock, §6.4 idempotency, §10 M4.4)."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Awaitable, Callable
from datetime import UTC, date, datetime
from pathlib import Path

import pytest

from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt
from apx.asp_mock.repository import (
    AspRecord,
    AspRepository,
    InMemoryAspRepository,
    SqliteAspRepository,
)
from apx.common.errors import IdempotencyConflictError
from apx.common.types import IdempotencyKey, Sha256Hex

_KEY: IdempotencyKey = "inv-1:asp_send"


def _sha(payload: bytes) -> Sha256Hex:
    return hashlib.sha256(payload).hexdigest()


def _record_builder(
    xml_sha256: Sha256Hex, *, calls: list[int], delay: float = 0.0
) -> Callable[[], Awaitable[AspRecord]]:
    """A `build` callable that counts its invocations and, when `delay` > 0, actually suspends
    partway through -- so concurrent callers racing on the same key genuinely interleave inside
    the critical section instead of one call running to completion before the next starts."""

    async def build() -> AspRecord:
        calls.append(1)
        if delay:
            await asyncio.sleep(delay)
        now = datetime.now(UTC)
        receipt = AspReceipt(
            receipt_id=f"receipt-{len(calls)}",
            idempotency_key=_KEY,
            status="accepted",
            received_at=now,
            message="accepted",
        )
        archive = ArchiveRecord(
            idempotency_key=_KEY,
            xml_sha256=xml_sha256,
            archived_at=now,
            retain_until=date(now.year + 5, now.month, now.day),
        )
        return AspRecord(
            idempotency_key=_KEY, xml_sha256=xml_sha256, receipt=receipt, archive=archive
        )

    return build


async def _repository_returns_same_record_for_replay(repository: AspRepository) -> None:
    sha = _sha(b"payload-a")
    calls: list[int] = []
    first = await repository.get_or_create(_KEY, sha, _record_builder(sha, calls=calls))
    second = await repository.get_or_create(_KEY, sha, _record_builder(sha, calls=calls))
    assert first == second
    assert len(calls) == 1, "the second call must not invoke build() again"
    assert await repository.get(_KEY) == first


async def _repository_conflicts_on_different_payload(repository: AspRepository) -> None:
    sha_a, sha_b = _sha(b"payload-a"), _sha(b"payload-b")
    calls: list[int] = []
    await repository.get_or_create(_KEY, sha_a, _record_builder(sha_a, calls=calls))
    with pytest.raises(IdempotencyConflictError):
        await repository.get_or_create(_KEY, sha_b, _record_builder(sha_b, calls=calls))


async def _repository_builds_once_under_concurrent_duplicates(repository: AspRepository) -> None:
    sha = _sha(b"payload-a")
    calls: list[int] = []
    build = _record_builder(sha, calls=calls, delay=0.01)
    results = await asyncio.gather(*(repository.get_or_create(_KEY, sha, build) for _ in range(20)))
    assert len(calls) == 1, f"build() ran {len(calls)} times under concurrent duplicates"
    assert len({r.receipt.receipt_id for r in results}) == 1
    assert len({r.archive.archived_at for r in results}) == 1


async def test_in_memory_repository_returns_same_record_for_replay() -> None:
    await _repository_returns_same_record_for_replay(InMemoryAspRepository())


async def test_in_memory_repository_conflicts_on_different_payload() -> None:
    await _repository_conflicts_on_different_payload(InMemoryAspRepository())


async def test_in_memory_repository_builds_once_under_concurrent_duplicates() -> None:
    await _repository_builds_once_under_concurrent_duplicates(InMemoryAspRepository())


async def test_sqlite_repository_returns_same_record_for_replay(tmp_path: Path) -> None:
    repo = SqliteAspRepository(tmp_path / "archive.sqlite3")
    try:
        await _repository_returns_same_record_for_replay(repo)
    finally:
        repo.close()


async def test_sqlite_repository_conflicts_on_different_payload(tmp_path: Path) -> None:
    repo = SqliteAspRepository(tmp_path / "archive.sqlite3")
    try:
        await _repository_conflicts_on_different_payload(repo)
    finally:
        repo.close()


async def test_sqlite_repository_builds_once_under_concurrent_duplicates(tmp_path: Path) -> None:
    repo = SqliteAspRepository(tmp_path / "archive.sqlite3")
    try:
        await _repository_builds_once_under_concurrent_duplicates(repo)
    finally:
        repo.close()


async def test_sqlite_repository_persists_to_a_second_instance(tmp_path: Path) -> None:
    db_path = tmp_path / "archive.sqlite3"
    sha = _sha(b"payload-a")
    calls: list[int] = []
    repo_a = SqliteAspRepository(db_path)
    try:
        stored = await repo_a.get_or_create(_KEY, sha, _record_builder(sha, calls=calls))
    finally:
        repo_a.close()

    repo_b = SqliteAspRepository(db_path)
    try:
        reread = await repo_b.get(_KEY)
    finally:
        repo_b.close()
    assert reread == stored

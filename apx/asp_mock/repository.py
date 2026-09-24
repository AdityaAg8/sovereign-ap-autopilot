"""Small storage interface behind the mock ASP: one `AspRecord` (receipt + archive) per
idempotency key, created at most once even under concurrent duplicate calls for the same key.
Ships an in-memory implementation for tests and a SQLite-backed implementation as the default,
file-persisted store (PRD §3 apx/asp_mock, §6.4 idempotency)."""

from __future__ import annotations

import asyncio
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Protocol

from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt
from apx.common.errors import IdempotencyConflictError
from apx.common.types import Frozen, IdempotencyKey, Sha256Hex


class AspRecord(Frozen):
    """The one stored row for an idempotency key: the receipt and archive built together the
    first time that key was seen, plus the payload identity used to detect a conflicting
    replay."""

    idempotency_key: IdempotencyKey
    xml_sha256: Sha256Hex
    receipt: AspReceipt
    archive: ArchiveRecord


class AspRepository(Protocol):
    async def get(self, key: IdempotencyKey) -> AspRecord | None:
        """Returns the stored record for `key`, or None if it has never been created."""
        ...

    async def get_or_create(
        self,
        key: IdempotencyKey,
        xml_sha256: Sha256Hex,
        build: Callable[[], Awaitable[AspRecord]],
    ) -> AspRecord:
        """Returns the existing record for `key` if one exists and its `xml_sha256` matches;
        raises `IdempotencyConflictError` if one exists with a *different* `xml_sha256`.
        Otherwise calls `build()` exactly once and stores its result, even when many callers
        race on the same key concurrently (exactly-once under concurrent duplicate sends,
        PRD §6.4)."""
        ...


class InMemoryAspRepository:
    """Process-local `AspRepository` for tests: a dict guarded by one lock per key so concurrent
    duplicate calls for the same key serialise on that key alone."""

    def __init__(self) -> None:
        self._records: dict[IdempotencyKey, AspRecord] = {}
        self._locks: dict[IdempotencyKey, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    async def _lock_for(self, key: IdempotencyKey) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(key)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[key] = lock
            return lock

    async def get(self, key: IdempotencyKey) -> AspRecord | None:
        return self._records.get(key)

    async def get_or_create(
        self,
        key: IdempotencyKey,
        xml_sha256: Sha256Hex,
        build: Callable[[], Awaitable[AspRecord]],
    ) -> AspRecord:
        lock = await self._lock_for(key)
        async with lock:
            existing = self._records.get(key)
            if existing is not None:
                if existing.xml_sha256 != xml_sha256:
                    raise IdempotencyConflictError(
                        f"idempotency key {key!r} was already used with a different payload"
                    )
                return existing
            record = await build()
            self._records[key] = record
            return record


class SqliteAspRepository:
    """File-backed default `AspRepository`: one SQLite table keyed by idempotency key, storing
    the receipt and archive as canonical JSON. All access is serialised by one `asyncio.Lock` so
    that, within the single-process mock ASP, concurrent duplicate calls for the same key still
    build the record exactly once (PRD §6.4); `sqlite3`'s own PRIMARY KEY constraint guards the
    on-disk table against a second, independent process writing the same key concurrently."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS asp_records (
                idempotency_key TEXT PRIMARY KEY,
                xml_sha256 TEXT NOT NULL,
                receipt_json TEXT NOT NULL,
                archive_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()
        self._lock = asyncio.Lock()

    def _row_to_record(self, row: tuple[str, str, str, str]) -> AspRecord:
        key, xml_sha256, receipt_json, archive_json = row
        # Sha256Hex/IdempotencyKey are Annotated[str, ...] aliases, not constructible types;
        # AspRecord's own field validation re-checks the pattern on assignment.
        return AspRecord(
            idempotency_key=key,
            xml_sha256=xml_sha256,
            receipt=AspReceipt.model_validate_json(receipt_json),
            archive=ArchiveRecord.model_validate_json(archive_json),
        )

    def _select(self, key: IdempotencyKey) -> AspRecord | None:
        cursor = self._conn.execute(
            "SELECT idempotency_key, xml_sha256, receipt_json, archive_json "
            "FROM asp_records WHERE idempotency_key = ?",
            (key,),
        )
        row = cursor.fetchone()
        return None if row is None else self._row_to_record(row)

    async def get(self, key: IdempotencyKey) -> AspRecord | None:
        async with self._lock:
            return self._select(key)

    async def get_or_create(
        self,
        key: IdempotencyKey,
        xml_sha256: Sha256Hex,
        build: Callable[[], Awaitable[AspRecord]],
    ) -> AspRecord:
        async with self._lock:
            existing = self._select(key)
            if existing is not None:
                if existing.xml_sha256 != xml_sha256:
                    raise IdempotencyConflictError(
                        f"idempotency key {key!r} was already used with a different payload"
                    )
                return existing
            record = await build()
            try:
                self._conn.execute(
                    "INSERT INTO asp_records "
                    "(idempotency_key, xml_sha256, receipt_json, archive_json) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        record.idempotency_key,
                        record.xml_sha256,
                        record.receipt.model_dump_json(),
                        record.archive.model_dump_json(),
                    ),
                )
                self._conn.commit()
            except sqlite3.IntegrityError as exc:
                # Another process won the race on the on-disk PRIMARY KEY between our SELECT and
                # INSERT; defer to whatever it stored rather than raising or losing data.
                self._conn.rollback()
                winner = self._select(key)
                if winner is None:
                    raise IdempotencyConflictError(
                        f"idempotency key {key!r} conflicted but no record could be read back"
                    ) from exc
                if winner.xml_sha256 != xml_sha256:
                    raise IdempotencyConflictError(
                        f"idempotency key {key!r} was already used with a different payload"
                    ) from exc
                return winner
            return record

    def close(self) -> None:
        self._conn.close()

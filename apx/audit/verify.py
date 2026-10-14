"""Audit chain verifier: `ChainWalker` / `verify_chain` is the pure, DB-free walk that recomputes
every hash and reports the first broken link (edit, delete, reorder or insert all break the
chain); `iter_chain_entries` streams `audit_log` in id-ordered pages from one snapshot, so
`PgAuditVerifier` runs in bounded memory however long the chain grows
(I4)."""

from __future__ import annotations

from collections.abc import AsyncGenerator, AsyncIterable, Iterable
from contextlib import aclosing
from typing import Final

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apx.audit.interfaces import GENESIS_HASH, AuditEntry, CanonicalValue, VerifyResult, entry_hash
from apx.db.models import AuditLog as AuditLogRow
from apx.db.session import get_session

CHAIN_PAGE_SIZE: Final[int] = 1000
"""Rows fetched per keyset page while streaming the chain; memory stays O(page), not O(chain)."""


async def use_snapshot(session: AsyncSession) -> None:
    """Makes `session`'s transaction REPEATABLE READ, so every page of a streamed walk (and any
    other query in it, such as the anchors) sees one consistent snapshot of the chain. Must be
    the transaction's first statement."""
    await session.connection(execution_options={"isolation_level": "REPEATABLE READ"})


async def iter_chain_entries(
    session: AsyncSession, *, page_size: int = CHAIN_PAGE_SIZE
) -> AsyncGenerator[AuditEntry, None]:
    """Yields every `audit_log` row, oldest first, as `AuditEntry`, fetching `page_size` rows at
    a time by keyset (`id > last id`). Selects plain columns, never ORM entities, so nothing
    accumulates in the session's identity map."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    columns = (
        AuditLogRow.id,
        AuditLogRow.ts,
        AuditLogRow.actor,
        AuditLogRow.action,
        AuditLogRow.entity,
        AuditLogRow.payload_json,
        AuditLogRow.prev_hash,
        AuditLogRow.hash,
    )
    last_id: int | None = None
    while True:
        stmt = select(*columns).order_by(AuditLogRow.id.asc()).limit(page_size)
        if last_id is not None:
            stmt = stmt.where(AuditLogRow.id > last_id)
        rows = (await session.execute(stmt)).all()
        for row in rows:
            yield AuditEntry(
                id=row.id,
                ts=row.ts,
                actor=row.actor,
                action=row.action,
                entity=row.entity,
                payload=row.payload_json,
                prev_hash=row.prev_hash,
                hash=row.hash,
            )
        if len(rows) < page_size:
            return
        last_id = rows[-1].id


class ChainWalker:
    """Incremental form of `verify_chain`: `feed` entries in chain order, then read `result()`.
    Holds only the previous hash and a counter. Entries fed after the first bad one are ignored,
    so the result is exactly what `verify_chain` reports for the same sequence."""

    def __init__(self) -> None:
        self._expected_prev_hash = GENESIS_HASH
        self._checked = 0
        self._first_bad_id: int | None = None

    @property
    def broken(self) -> bool:
        return self._first_bad_id is not None

    def feed(self, entry: AuditEntry) -> None:
        if self._first_bad_id is not None:
            return
        self._checked += 1
        if entry.prev_hash != self._expected_prev_hash:
            self._first_bad_id = entry.id
            return
        row_without_hash: dict[str, CanonicalValue] = {
            "id": entry.id,
            "ts": entry.ts,
            "actor": entry.actor,
            "action": entry.action,
            "entity": entry.entity,
            "payload_json": dict(entry.payload),
            "prev_hash": entry.prev_hash,
        }
        if entry_hash(entry.prev_hash, row_without_hash) != entry.hash:
            self._first_bad_id = entry.id
            return
        self._expected_prev_hash = entry.hash

    def result(self) -> VerifyResult:
        return VerifyResult(
            ok=self._first_bad_id is None,
            entries_checked=self._checked,
            first_bad_id=self._first_bad_id,
        )


def verify_chain(entries: Iterable[AuditEntry]) -> VerifyResult:
    """Walks `entries` in the given order. For each one, checks that `prev_hash` equals the
    previous entry's `hash` (genesis for the first) and that recomputing `entry_hash` from its
    own fields reproduces its stored `hash`. Stops and reports the first id that fails either
    check -- an edited row fails the second check at its own id; a deleted, reordered or
    inserted row breaks the link and fails the first check at the id right after the break."""
    walker = ChainWalker()
    for entry in entries:
        walker.feed(entry)
        if walker.broken:
            break
    return walker.result()


async def verify_chain_stream(entries: AsyncIterable[AuditEntry]) -> VerifyResult:
    """`verify_chain` over an async stream (e.g. `iter_chain_entries`), stopping at the first
    broken link without reading the rest."""
    walker = ChainWalker()
    async for entry in entries:
        walker.feed(entry)
        if walker.broken:
            break
    return walker.result()


class PgAuditVerifier:
    """`apx.audit.interfaces.AuditVerifier` implementation backed by Postgres, streaming the
    chain `page_size` rows at a time from one REPEATABLE READ snapshot."""

    def __init__(self, page_size: int = CHAIN_PAGE_SIZE) -> None:
        self._page_size = page_size

    async def verify(self) -> VerifyResult:
        async with get_session() as session:
            await use_snapshot(session)
            async with aclosing(iter_chain_entries(session, page_size=self._page_size)) as rows:
                return await verify_chain_stream(rows)

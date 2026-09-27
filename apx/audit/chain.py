"""Postgres-backed `AuditLog`: appends to `audit_log` under a transaction-scoped advisory lock so
concurrent writers extend the hash chain one at a time and it can never fork (PRD §6.2 I4,
§6.4, §10 M10.4). `append_in` (and `append_in_transaction` for any `AuditLog`) writes the event
inside the caller's transaction, so a state change and its audit entry commit or roll back
together: no phantom entry for a rolled-back write, no committed write without its entry
(REVIEW M-02)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final, Protocol, runtime_checkable

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from apx.audit.interfaces import (
    GENESIS_HASH,
    AuditEntry,
    AuditEvent,
    AuditLog,
    CanonicalValue,
    entry_hash,
)
from apx.common.errors import AuditChainError
from apx.db.models import AuditLog as AuditLogRow
from apx.db.session import get_session

_CHAIN_LOCK_KEY: Final[int] = 0x415044_5F4C4F47
"""Arbitrary, stable `pg_advisory_xact_lock` key naming the single `audit_log` chain-head
lock (spells "APD_LOG" in hex); every appender takes this lock before reading the head, so at
most one transaction extends the chain at a time (I4)."""


@runtime_checkable
class TransactionalAuditLog(Protocol):
    """An `AuditLog` that can also append inside a caller-owned database transaction."""

    async def append(self, event: AuditEvent) -> AuditEntry: ...

    async def append_in(self, session: AsyncSession, event: AuditEvent) -> AuditEntry: ...


async def append_in_transaction(
    audit: AuditLog, session: AsyncSession, event: AuditEvent
) -> AuditEntry:
    """Appends `event` inside `session`'s transaction when `audit` supports it (`PgAuditLog`
    always does), so it commits or rolls back with the caller's state writes. An `AuditLog`
    that is not backed by this database (an in-memory log in tests) has no transaction to
    join and appends on its own, exactly as `append` always did."""
    if isinstance(audit, TransactionalAuditLog):
        return await audit.append_in(session, event)
    return await audit.append(event)


class PgAuditLog:
    """`apx.audit.interfaces.AuditLog` implementation backed by Postgres (asyncpg driver)."""

    async def append(self, event: AuditEvent) -> AuditEntry:
        """Append in a transaction of its own, under `_CHAIN_LOCK_KEY`. Raises
        `AuditChainError`; never drops an event."""
        try:
            async with get_session() as session:
                entry = await self.append_in(session, event)
        except AuditChainError:
            raise
        except Exception as exc:
            raise AuditChainError(f"audit log append failed: {exc}") from exc
        return entry

    async def append_in(self, session: AsyncSession, event: AuditEvent) -> AuditEntry:
        """Append inside `session`'s open transaction, under `_CHAIN_LOCK_KEY`. The entry is
        durable only when the caller commits, and vanishes if it rolls back. The chain lock is
        held until then, so callers append last, after their row locks, and commit promptly.
        Raises `AuditChainError`."""
        try:
            # `pg_advisory_xact_lock` blocks until acquired and releases automatically at
            # commit/rollback, so it serialises appenders without needing an existing row
            # to lock (unlike `SELECT ... FOR UPDATE`, which cannot lock a chain of zero
            # rows for the very first append).
            await session.execute(
                text("SELECT pg_advisory_xact_lock(:key)"), {"key": _CHAIN_LOCK_KEY}
            )

            head_hash = (
                await session.execute(
                    select(AuditLogRow.hash).order_by(AuditLogRow.id.desc()).limit(1)
                )
            ).scalar_one_or_none()
            prev_hash = head_hash if head_hash is not None else GENESIS_HASH

            # Reserve the row's id up front (works for both `SERIAL` and `IDENTITY`
            # columns) so it can be included in the hashed row, matching the other
            # columns' meaning of "assigned once, immutable thereafter" (§6.4).
            next_id = int(
                (
                    await session.execute(
                        text("SELECT nextval(pg_get_serial_sequence('audit_log', 'id'))")
                    )
                ).scalar_one()
            )

            ts = datetime.now(UTC)
            payload = dict(event.payload)
            row_without_hash: dict[str, CanonicalValue] = {
                "id": next_id,
                "ts": ts,
                "actor": event.actor,
                "action": event.action,
                "entity": event.entity,
                "payload_json": payload,
                "prev_hash": prev_hash,
            }
            hash_ = entry_hash(prev_hash, row_without_hash)

            session.add(
                AuditLogRow(
                    id=next_id,
                    ts=ts,
                    actor=event.actor,
                    action=event.action,
                    entity=event.entity,
                    payload_json=payload,
                    prev_hash=prev_hash,
                    hash=hash_,
                )
            )
            await session.flush()
        except AuditChainError:
            raise
        except Exception as exc:
            raise AuditChainError(f"audit log append failed: {exc}") from exc

        return AuditEntry(
            id=next_id,
            ts=ts,
            actor=event.actor,
            action=event.action,
            entity=event.entity,
            payload=event.payload,
            prev_hash=prev_hash,
            hash=hash_,
        )

"""Postgres-backed `AuditLog`: appends to `audit_log` under a session-scoped advisory lock so
concurrent writers extend the hash chain one at a time and it can never fork (PRD §6.2 I4,
§6.4, §10 M10.4)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Final

from sqlalchemy import select, text

from apx.audit.interfaces import GENESIS_HASH, AuditEntry, AuditEvent, CanonicalValue, entry_hash
from apx.common.errors import AuditChainError
from apx.db.models import AuditLog as AuditLogRow
from apx.db.session import get_session

_CHAIN_LOCK_KEY: Final[int] = 0x415044_5F4C4F47
"""Arbitrary, stable `pg_advisory_xact_lock` key naming the single `audit_log` chain-head
lock (spells "APD_LOG" in hex); every appender takes this lock before reading the head, so at
most one transaction extends the chain at a time (I4)."""


class PgAuditLog:
    """`apx.audit.interfaces.AuditLog` implementation backed by Postgres (asyncpg driver)."""

    async def append(self, event: AuditEvent) -> AuditEntry:
        """Append under `_CHAIN_LOCK_KEY`. Raises `AuditChainError`; never drops an event."""
        try:
            async with get_session() as session:
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

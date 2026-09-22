"""Audit chain verifier: `verify_chain` is the pure, DB-free walk that recomputes every hash
and reports the first broken link (edit, delete, reorder or insert all break the chain); `Pg
AuditVerifier` runs it against the real `audit_log` table (PRD §6.2 I4, §6.4, §10 M10.4)."""

from __future__ import annotations

from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from apx.audit.interfaces import GENESIS_HASH, AuditEntry, CanonicalValue, VerifyResult, entry_hash
from apx.db.models import AuditLog as AuditLogRow
from apx.db.session import get_session


async def load_chain_entries(session: AsyncSession) -> list[AuditEntry]:
    """Loads every `audit_log` row, oldest first, as `AuditEntry`. Shared by `PgAuditVerifier`
    and `apx.audit.anchor.PgAuditFullVerifier`, which also needs the raw entries (not just a
    pass/fail) to cross-check against head anchors."""
    rows = (
        (await session.execute(select(AuditLogRow).order_by(AuditLogRow.id.asc()))).scalars().all()
    )
    return [
        AuditEntry(
            id=row.id,
            ts=row.ts,
            actor=row.actor,
            action=row.action,
            entity=row.entity,
            payload=row.payload_json,
            prev_hash=row.prev_hash,
            hash=row.hash,
        )
        for row in rows
    ]


def verify_chain(entries: Sequence[AuditEntry]) -> VerifyResult:
    """Walks `entries` in the given order. For each one, checks that `prev_hash` equals the
    previous entry's `hash` (genesis for the first) and that recomputing `entry_hash` from its
    own fields reproduces its stored `hash`. Stops and reports the first id that fails either
    check -- an edited row fails the second check at its own id; a deleted, reordered or
    inserted row breaks the link and fails the first check at the id right after the break."""
    expected_prev_hash = GENESIS_HASH
    checked = 0
    for entry in entries:
        checked += 1
        if entry.prev_hash != expected_prev_hash:
            return VerifyResult(ok=False, entries_checked=checked, first_bad_id=entry.id)

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
            return VerifyResult(ok=False, entries_checked=checked, first_bad_id=entry.id)

        expected_prev_hash = entry.hash

    return VerifyResult(ok=True, entries_checked=checked, first_bad_id=None)


class PgAuditVerifier:
    """`apx.audit.interfaces.AuditVerifier` implementation backed by Postgres."""

    async def verify(self) -> VerifyResult:
        async with get_session() as session:
            entries = await load_chain_entries(session)
        return verify_chain(entries)

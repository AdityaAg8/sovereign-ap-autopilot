"""Interfaces for the hash-chained audit log: entry model, canonical JSON, hash computation,
append and verification (PRD §6.2 I4, §6.4)."""

from __future__ import annotations

from datetime import datetime
from typing import Final, Protocol

from pydantic import JsonValue

from apx.common.types import Frozen, Sha256Hex

GENESIS_HASH: Final[str] = "0" * 64
"""prev_hash of the first entry."""


class AuditEvent(Frozen):
    """An event to append; id, ts and hashes are assigned by the log."""

    actor: str
    action: str
    entity: str
    payload: dict[str, JsonValue]


class AuditEntry(Frozen):
    id: int
    ts: datetime
    actor: str
    action: str
    entity: str
    payload: dict[str, JsonValue]
    prev_hash: Sha256Hex
    hash: Sha256Hex


class VerifyResult(Frozen):
    ok: bool
    entries_checked: int
    first_bad_id: int | None


class AuditLog(Protocol):
    async def append(self, event: AuditEvent) -> AuditEntry:
        """Append under a serialising lock. Raises AuditChainError; never drops an event."""
        ...


class AuditVerifier(Protocol):
    async def verify(self) -> VerifyResult: ...


def canonical_json(value: JsonValue) -> bytes:
    """Deterministic JSON: sorted keys, no whitespace, UTF-8, ISO-8601 timestamps."""
    raise NotImplementedError


def entry_hash(prev_hash: str, row_without_hash: dict[str, JsonValue]) -> Sha256Hex:
    """sha256(prev_hash || canonical_json(row without hash)) as lowercase hex (§6.4)."""
    raise NotImplementedError

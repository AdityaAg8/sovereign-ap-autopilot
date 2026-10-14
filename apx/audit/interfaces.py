"""Interfaces for the hash-chained audit log: entry model, canonical JSON, hash computation,
append and verification (I4)."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from typing import Final, Protocol

from pydantic import JsonValue

from apx.common.types import Frozen, Sha256Hex

GENESIS_HASH: Final[str] = "0" * 64
"""prev_hash of the first entry."""

type CanonicalValue = (
    JsonValue | Decimal | datetime | Sequence[CanonicalValue] | Mapping[str, CanonicalValue]
)
"""Anything `canonical_json` can encode: plain JSON values, plus `Decimal` and timezone-aware
`datetime` (normalised to strings), nested in any mix of mappings/sequences. A superset of
`JsonValue` so a row's already-JSON `payload_json` sits alongside its raw `ts`/`Decimal`
columns in one dict without a cast."""


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


def canonical_json(value: CanonicalValue) -> bytes:
    """Deterministic JSON encoding used as the audit hash pre-image: object keys sorted
    at every level, no whitespace, UTF-8 bytes. `Decimal` is rendered fixed-point (`format(x,
    "f")`, never scientific notation) and a timezone-aware `datetime` is converted to UTC and
    rendered `isoformat()`; both become JSON strings so the result is stable across equal
    values with different in-memory representations (e.g. `Decimal("1.50")` vs `Decimal("1.5")`
    still differ deliberately -- callers must normalise scale themselves before hashing money).

    Raises `ValueError` for a naive `datetime` or a non-finite `float` (`nan`/`inf`), and
    `TypeError` for any other type -- both would make the hash irreproducible."""
    return json.dumps(
        _normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def _normalize(value: CanonicalValue) -> JsonValue:
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, Decimal):
        return format(value, "f")
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError(f"canonical_json requires a timezone-aware datetime, got {value!r}")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"canonical_json cannot encode a non-finite float: {value!r}")
        return value
    if isinstance(value, Mapping):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, Sequence):
        return [_normalize(item) for item in value]
    raise TypeError(f"canonical_json cannot encode a {type(value).__name__}")


def entry_hash(prev_hash: str, row_without_hash: Mapping[str, CanonicalValue]) -> Sha256Hex:
    """sha256(prev_hash || canonical_json(row without hash)) as lowercase hex."""
    body = canonical_json(dict(row_without_hash))
    return hashlib.sha256(prev_hash.encode("ascii") + body).hexdigest()

"""Interfaces for the mock Access Service Provider: exactly-once submission receipts and archive
records keyed by idempotency key (PRD §3, §6.4)."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Protocol

from apx.common.types import Frozen, IdempotencyKey, Sha256Hex
from apx.einvoice.interfaces import EInvoiceDocument


class AspSubmission(Frozen):
    idempotency_key: IdempotencyKey
    document: EInvoiceDocument


class AspReceipt(Frozen):
    receipt_id: str
    idempotency_key: IdempotencyKey
    status: Literal["accepted", "rejected"]
    received_at: datetime
    message: str


class ArchiveRecord(Frozen):
    idempotency_key: IdempotencyKey
    xml_sha256: Sha256Hex
    archived_at: datetime
    retain_until: date


class AspClient(Protocol):
    async def submit(self, submission: AspSubmission) -> AspReceipt:
        """Replaying the same key returns the original receipt; a different payload under the
        same key raises IdempotencyConflictError. Raises AspError."""
        ...


class Archive(Protocol):
    async def store(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        """Exactly-once by key, same semantics as AspClient.submit."""
        ...

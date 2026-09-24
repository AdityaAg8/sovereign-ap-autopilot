"""Concrete mock ASP: implements `AspClient` and `Archive` (apx.asp_mock.interfaces) on top of
an `AspRepository`, validating each document against the runtime PINT AE schematron gate before
it is accepted, and stamping every accepted archive record with 5-year retention metadata
(PRD §3 apx/asp_mock, §6.4 idempotency, §10 M4.4)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, date, datetime
from uuid import uuid4

from apx.asp_mock.errors import DocumentRejectedError
from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt, AspSubmission
from apx.asp_mock.repository import AspRecord, AspRepository
from apx.common.types import IdempotencyKey
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument, SchematronValidator

_RETENTION_YEARS = 5
"""Minimum e-invoice archive retention period (PRD §3 apx/asp_mock)."""


def retain_until(archived_at: datetime, *, years: int = _RETENTION_YEARS) -> date:
    """`archived_at`'s date plus `years`; 29 Feb rolls back to 28 Feb in a non-leap target year
    rather than raising."""
    d = archived_at.date()
    try:
        return d.replace(year=d.year + years)
    except ValueError:
        return d.replace(month=2, day=28, year=d.year + years)


class MockAsp:
    """Exactly-once, in-process mock of an Access Service Provider. `submit` and `store` both
    key off the same idempotency key -> `AspRecord` mapping in `repository`, so a request that
    goes through `submit` (validate + accept) makes the matching archive record retrievable via
    `store`/`AspRepository.get` under that same key without a second schematron run."""

    def __init__(
        self,
        repository: AspRepository,
        *,
        validator_factory: Callable[[DocumentTypeCode], SchematronValidator],
    ) -> None:
        self._repository = repository
        self._validator_factory = validator_factory
        self._validators: dict[DocumentTypeCode, SchematronValidator] = {}
        self._validators_lock = asyncio.Lock()

    async def submit(self, submission: AspSubmission) -> AspReceipt:
        """Replaying `submission.idempotency_key` returns the original receipt; the same key
        with a different document raises `IdempotencyConflictError`. Raises
        `DocumentRejectedError` (an `AspError`) with the failed schematron rule IDs when the
        document does not pass validation."""
        await self._validate(submission.document)
        record = await self._get_or_create(submission.idempotency_key, submission.document)
        return record.receipt

    async def store(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        """Exactly-once by key, same semantics as `submit`. Does not re-validate: an archive
        entry for a document that was never accepted by `submit` is out of scope for the mock
        ASP's single ingress path but the method itself has no such precondition."""
        record = await self._get_or_create(key, document)
        return record.archive

    async def _validate(self, document: EInvoiceDocument) -> None:
        validator = await self._validator_for(document.type_code)
        result = validator.validate(document.xml)
        if not result.passed:
            raise DocumentRejectedError(tuple(f for f in result.failures if f.flag == "fatal"))

    async def _validator_for(self, type_code: DocumentTypeCode) -> SchematronValidator:
        async with self._validators_lock:
            validator = self._validators.get(type_code)
            if validator is None:
                validator = self._validator_factory(type_code)
                self._validators[type_code] = validator
            return validator

    async def _get_or_create(self, key: IdempotencyKey, document: EInvoiceDocument) -> AspRecord:
        async def build() -> AspRecord:
            now = datetime.now(UTC)
            receipt = AspReceipt(
                receipt_id=str(uuid4()),
                idempotency_key=key,
                status="accepted",
                received_at=now,
                message="accepted",
            )
            archive = ArchiveRecord(
                idempotency_key=key,
                xml_sha256=document.xml_sha256,
                archived_at=now,
                retain_until=retain_until(now),
            )
            return AspRecord(
                idempotency_key=key,
                xml_sha256=document.xml_sha256,
                receipt=receipt,
                archive=archive,
            )

        return await self._repository.get_or_create(key, document.xml_sha256, build)

    def close(self) -> None:
        """Best-effort teardown for validators that hold native resources (for example a live
        Saxon processor); safe to call even when none were ever built."""
        for validator in self._validators.values():
            close = getattr(validator, "close", None)
            if callable(close):
                close()


__all__ = ["MockAsp", "retain_until"]

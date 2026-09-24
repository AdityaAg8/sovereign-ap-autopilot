"""Unit tests for `apx.asp_mock.service.MockAsp`: submit/store are idempotent by key, a
rejected document raises `DocumentRejectedError` carrying the schematron rule IDs, and 20
concurrent duplicate submits for the same key yield exactly one archive record (PRD §3
apx/asp_mock, §6.4 idempotency, §10 M4.4). Uses a fake `SchematronValidator` throughout: the
real PINT AE validator is exercised end to end in `test_asp_mock_app.py`."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, date, datetime

import pytest

from apx.asp_mock.errors import DocumentRejectedError
from apx.asp_mock.interfaces import AspReceipt, AspSubmission
from apx.asp_mock.repository import InMemoryAspRepository
from apx.asp_mock.service import MockAsp, retain_until
from apx.common.errors import IdempotencyConflictError
from apx.common.types import IdempotencyKey, InvoiceId
from apx.einvoice.interfaces import (
    DocumentTypeCode,
    EInvoiceDocument,
    SchematronFailure,
    SchematronResult,
)

_PASSING = SchematronResult(passed=True, failures=(), rules_version="1.0.4")
_FAILING = SchematronResult(
    passed=False,
    failures=(
        SchematronFailure(rule_id="ibr-002", flag="fatal", location="/Invoice", message="no ID"),
        SchematronFailure(
            rule_id="ibr-cl-25", flag="warning", location="/Invoice", message="advisory"
        ),
    ),
    rules_version="1.0.4",
)


class _FakeValidator:
    """Structurally satisfies `apx.einvoice.interfaces.SchematronValidator`."""

    def __init__(self, result: SchematronResult) -> None:
        self.result = result
        self.calls = 0
        self.closed = False

    def validate(self, xml: bytes) -> SchematronResult:
        self.calls += 1
        return self.result

    def close(self) -> None:
        self.closed = True


def _document(xml: bytes = b"<Invoice/>", invoice_id: str = "inv-1") -> EInvoiceDocument:
    return EInvoiceDocument(
        invoice_id=InvoiceId(invoice_id),
        type_code=DocumentTypeCode.INVOICE,
        xml=xml,
        xml_sha256=hashlib.sha256(xml).hexdigest(),
        spec_version="1.0.4",
    )


def _service(result: SchematronResult = _PASSING) -> tuple[MockAsp, InMemoryAspRepository]:
    repository = InMemoryAspRepository()
    service = MockAsp(repository, validator_factory=lambda _tc: _FakeValidator(result))
    return service, repository


async def test_close_closes_every_validator_built_so_far() -> None:
    built: list[_FakeValidator] = []

    def factory(_tc: DocumentTypeCode) -> _FakeValidator:
        validator = _FakeValidator(_PASSING)
        built.append(validator)
        return validator

    service = MockAsp(InMemoryAspRepository(), validator_factory=factory)
    key: IdempotencyKey = "inv-1:asp_send"
    await service.submit(AspSubmission(idempotency_key=key, document=_document()))
    assert len(built) == 1 and not built[0].closed

    service.close()
    assert built[0].closed


async def test_submit_returns_an_accepted_receipt() -> None:
    service, _ = _service()
    key: IdempotencyKey = "inv-1:asp_send"
    receipt = await service.submit(AspSubmission(idempotency_key=key, document=_document()))
    assert receipt.status == "accepted"
    assert receipt.idempotency_key == key
    assert receipt.receipt_id


async def test_duplicate_submit_returns_the_same_receipt() -> None:
    service, _ = _service()
    key: IdempotencyKey = "inv-1:asp_send"
    submission = AspSubmission(idempotency_key=key, document=_document())
    first = await service.submit(submission)
    second = await service.submit(submission)
    assert first == second


async def test_submit_with_different_payload_under_same_key_conflicts() -> None:
    service, _ = _service()
    key: IdempotencyKey = "inv-1:asp_send"
    await service.submit(AspSubmission(idempotency_key=key, document=_document(xml=b"<Invoice/>")))
    with pytest.raises(IdempotencyConflictError):
        await service.submit(
            AspSubmission(idempotency_key=key, document=_document(xml=b"<Invoice>x</Invoice>"))
        )


async def test_submit_rejects_a_failing_document_with_rule_ids() -> None:
    service, repository = _service(_FAILING)
    key: IdempotencyKey = "inv-1:asp_send"
    with pytest.raises(DocumentRejectedError) as excinfo:
        await service.submit(AspSubmission(idempotency_key=key, document=_document()))
    rule_ids = {f.rule_id for f in excinfo.value.failures}
    assert rule_ids == {"ibr-002"}, "only fatal failures are carried, not warnings"
    assert await repository.get(key) is None, "a rejected document must not be archived"


async def test_store_archives_with_5_year_retention_metadata() -> None:
    service, _ = _service()
    key: IdempotencyKey = "inv-1:archive"
    document = _document()
    archive = await service.store(key, document)
    assert archive.idempotency_key == key
    assert archive.xml_sha256 == document.xml_sha256
    assert archive.retain_until == retain_until(archive.archived_at)
    assert archive.retain_until.year == archive.archived_at.year + 5


async def test_duplicate_store_returns_the_same_archive_record() -> None:
    service, _ = _service()
    key: IdempotencyKey = "inv-1:archive"
    document = _document()
    first = await service.store(key, document)
    second = await service.store(key, document)
    assert first == second


@pytest.mark.parametrize(
    ("archived_at", "expected"),
    [
        (datetime(2024, 2, 29, tzinfo=UTC), date(2029, 2, 28)),
        (datetime(2020, 1, 15, tzinfo=UTC), date(2025, 1, 15)),
    ],
)
def test_retain_until_handles_leap_day(archived_at: datetime, expected: date) -> None:
    assert retain_until(archived_at) == expected


async def test_concurrent_duplicate_submits_yield_one_archive_record() -> None:
    service, repository = _service()
    key: IdempotencyKey = "inv-1:asp_send"
    submission = AspSubmission(idempotency_key=key, document=_document())

    async def submit_with_jitter() -> AspReceipt:
        await asyncio.sleep(0)
        return await service.submit(submission)

    receipts = await asyncio.gather(*(submit_with_jitter() for _ in range(20)))
    receipt_ids = {r.receipt_id for r in receipts}
    assert len(receipt_ids) == 1

    record = await repository.get(key)
    assert record is not None
    assert record.receipt.receipt_id == receipt_ids.pop()

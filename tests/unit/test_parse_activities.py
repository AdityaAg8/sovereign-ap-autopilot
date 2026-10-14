"""Unit tests for the parse-handoff activities of `DefaultInvoiceActivities`: `parse` stores the
document and returns only a ref; `extract`, `trust_gate` and `fraud` load the full parse (page
PNGs included) from the store; a wiped store is audited and re-parsed, and a re-parse that does
not reproduce the ref, or a corrupt entry, raises a typed error."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

import pytest

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.common.errors import ParsedArtifactIntegrityError
from apx.common.invoice import Invoice
from apx.common.types import InvoiceId
from apx.db.enums import DbInvoiceStatus
from apx.extract.interfaces import ExtractionResult, FieldName
from apx.fraud.interfaces import FraudFlag, FraudReport
from apx.ingest.interfaces import IngestedDocument
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.parse.store import FileParsedDocumentStore
from apx.policy.interfaces import RoutingDecision
from apx.trust.interfaces import FieldFeatures, TrustDecision
from apx.workflows.activities import DefaultInvoiceActivities, InvoiceActivityDeps
from tests.unit import _payload_fixtures as fx

_UNUSED: Any = None


class _Documents:
    def get(self, sha256: str) -> bytes:
        assert sha256 == fx.DOC_SHA
        return b"%PDF-1.7 fake"

    def put(  # pragma: no cover
        self, data: bytes, filename: str, source: Literal["watch", "upload"]
    ) -> IngestedDocument:
        raise NotImplementedError

    def exists(self, sha256: str) -> bool:  # pragma: no cover
        return True


class _Parser:
    def __init__(self, document: ParsedDocument) -> None:
        self.document = document
        self.calls = 0

    async def parse(self, document_sha256: str, data: bytes) -> ParsedDocument:
        self.calls += 1
        return self.document


class _Extraction:
    def __init__(self) -> None:
        self.seen: ParsedDocument | None = None

    async def run(self, document: ParsedDocument) -> ExtractionResult:
        self.seen = document
        return fx.extraction_result()


class _Features:
    def __init__(self) -> None:
        self.seen: ParsedDocument | None = None

    def features(
        self, result: ExtractionResult, document: ParsedDocument
    ) -> dict[FieldName, FieldFeatures]:
        self.seen = document
        return {}


class _Gate:
    def decide(self, features: Mapping[FieldName, FieldFeatures]) -> TrustDecision:
        return fx.trust_decision()


class _Fraud:
    def __init__(self) -> None:
        self.page_images: list[bytes] = []

    async def assess(self, invoice: Invoice, page_images: Sequence[bytes]) -> FraudReport:
        self.page_images = list(page_images)
        return FraudReport(flags=(), max_score=0.0)


class _Records:
    async def flags(self, invoice_id: InvoiceId, flags: Sequence[FraudFlag]) -> None:
        return None

    async def save(self, invoice: Invoice, result: ExtractionResult) -> None:  # pragma: no cover
        raise NotImplementedError

    async def set_status(  # pragma: no cover
        self, invoice_id: InvoiceId, status: DbInvoiceStatus
    ) -> None:
        raise NotImplementedError

    async def routed(  # pragma: no cover
        self, invoice_id: InvoiceId, routing: RoutingDecision, trust: TrustDecision
    ) -> None:
        raise NotImplementedError


class _Audit:
    def __init__(self) -> None:
        self.actions: list[str] = []
        self.payloads: dict[str, dict[str, Any]] = {}

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.actions.append(event.action)
        self.payloads[event.action] = dict(event.payload)
        return AuditEntry.model_validate(
            {
                **event.model_dump(),
                "id": len(self.actions),
                "ts": "2026-09-23T00:00:00Z",
                "prev_hash": "0" * 64,
                "hash": f"{len(self.actions):064x}",
            }
        )


@pytest.fixture(scope="module")
def document() -> ParsedDocument:
    return fx.parsed_document()


class _Harness:
    def __init__(self, root: Path, document: ParsedDocument) -> None:
        self.store = FileParsedDocumentStore(root)
        self.parser = _Parser(document)
        self.extraction = _Extraction()
        self.features = _Features()
        self.fraud = _Fraud()
        self.audit = _Audit()
        self.activities = DefaultInvoiceActivities(
            InvoiceActivityDeps(
                documents=_Documents(),
                scanner=_UNUSED,
                parser=self.parser,
                parsed=self.store,
                extraction=self.extraction,
                invoices=_UNUSED,
                feature_builder=self.features,
                trust_gate=_Gate(),
                validator=_UNUSED,
                fraud_engine=self.fraud,
                vendors=_UNUSED,
                policy_engine=_UNUSED,
                ubl_builder=_UNUSED,
                schematron=_UNUSED,
                asp_client=_UNUSED,
                archive=_UNUSED,
                audit=self.audit,
                records=_Records(),
                session_factory=_UNUSED,
                idempotency=_UNUSED,
            )
        )


async def test_parse_returns_ref_and_consumers_load_the_full_parse(
    tmp_path: Path, document: ParsedDocument
) -> None:
    h = _Harness(tmp_path, document)

    ref = await h.activities.parse(fx.workflow_input())
    await h.activities.extract(ref)
    await h.activities.trust_gate(fx.extraction_result(), ref)
    await h.activities.fraud(fx.invoice(), ref)

    assert isinstance(ref, ParsedDocumentRef)
    assert ref.page_count == fx.PAGES
    assert h.extraction.seen == document
    assert h.features.seen == document
    assert h.fraud.page_images == [page.image_png for page in document.pages]
    assert h.parser.calls == 1
    assert "activity:parse_cache_miss" not in h.audit.actions


async def test_parse_is_retry_safe(tmp_path: Path, document: ParsedDocument) -> None:
    h = _Harness(tmp_path, document)

    first = await h.activities.parse(fx.workflow_input())
    retried = await h.activities.parse(fx.workflow_input())

    assert first == retried
    assert h.store.load(first) == document


async def test_wiped_store_is_audited_and_reparsed(
    tmp_path: Path, document: ParsedDocument
) -> None:
    h = _Harness(tmp_path / "parsed", document)
    ref = await h.activities.parse(fx.workflow_input())
    for path in sorted((tmp_path / "parsed").rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()

    await h.activities.extract(ref)

    assert h.extraction.seen == document
    assert h.parser.calls == 2
    assert h.audit.actions.count("activity:parse_cache_miss") == 1
    assert h.store.load(ref) == document


async def test_reparse_that_differs_from_the_ref_is_an_integrity_error(
    tmp_path: Path, document: ParsedDocument
) -> None:
    h = _Harness(tmp_path / "parsed", document)
    ref = await h.activities.parse(fx.workflow_input())
    (tmp_path / "parsed" / ref.document_sha256 / ref.content_sha256 / "document.json").unlink()
    h.parser.document = document.model_copy(update={"parser_version": "paddleocr-vl-1.7/1"})

    with pytest.raises(ParsedArtifactIntegrityError):
        await h.activities.extract(ref)
    assert h.extraction.seen is None


async def test_corrupt_entry_is_not_repaired_by_reparsing(
    tmp_path: Path, document: ParsedDocument
) -> None:
    h = _Harness(tmp_path, document)
    ref = await h.activities.parse(fx.workflow_input())
    (tmp_path / ref.document_sha256 / ref.content_sha256 / "page-0000.png").write_bytes(b"x")

    with pytest.raises(ParsedArtifactIntegrityError):
        await h.activities.fraud(fx.invoice(), ref)
    assert h.parser.calls == 1
    assert "activity:parse_cache_miss" not in h.audit.actions


async def test_a_complete_parse_audits_no_truncated_page_and_leaves_the_gate_alone(
    tmp_path: Path, document: ParsedDocument
) -> None:
    h = _Harness(tmp_path, document)
    ref = await h.activities.parse(fx.workflow_input())

    decision = await h.activities.trust_gate(fx.extraction_result(), ref)

    assert h.audit.payloads["activity:parse"]["truncated_pages"] == 0
    assert decision == fx.trust_decision()


async def test_a_truncated_page_is_audited_and_routes_to_review(
    tmp_path: Path, document: ParsedDocument
) -> None:
    """ADR-0013 amendment 1 (13e9e6552e94: the OCR budget cut pages 2 and 3 and 13 of 39 lines
    were lost): a parse with a page the OCR service cut short is counted in `activity:parse`, and
    the trust gate accepts no field, so the invoice routes to review."""
    pages = list(document.pages)
    pages[1] = pages[1].model_copy(update={"truncated": True})
    h = _Harness(tmp_path, document.model_copy(update={"pages": tuple(pages)}))
    ref = await h.activities.parse(fx.workflow_input())

    decision = await h.activities.trust_gate(fx.extraction_result(), ref)

    assert h.audit.payloads["activity:parse"]["truncated_pages"] == 1
    assert not decision.all_accepted
    assert not any(f.accepted for f in decision.fields)
    assert decision.refusal == f"parse_truncated: OCR cut short on page(s) 2 of {fx.PAGES}"
    assert h.audit.payloads["activity:trust_gate"]["refusal"] == decision.refusal

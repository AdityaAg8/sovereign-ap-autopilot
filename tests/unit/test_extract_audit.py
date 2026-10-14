"""The `extract` activity's audit payload: besides the disputed fields and whether an arbiter
ran, it records each extractor's validity and finish reason, so an invalid Extractor B output
(every field disputed, trust gate rule H1) is visible in the audit trail. Runs the real activity
with a fake extraction pipeline and an in-memory audit log."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from apx.audit.interfaces import AuditEntry, AuditEvent
from apx.extract.extractor import invalid_output
from apx.extract.interfaces import ExtractionResult, ExtractorId
from apx.parse.interfaces import ParsedDocument
from apx.parse.store import FileParsedDocumentStore
from apx.workflows.activities import (
    DefaultInvoiceActivities,
    ExtractAudit,
    InvoiceActivityDeps,
    extract_audit,
)
from tests.unit import _payload_fixtures as fx


class _Audit:
    def __init__(self) -> None:
        self.events: list[AuditEvent] = []

    async def append(self, event: AuditEvent) -> AuditEntry:
        self.events.append(event)
        return AuditEntry.model_validate(
            {
                **event.model_dump(),
                "id": len(self.events),
                "ts": "2026-09-29T00:00:00Z",
                "prev_hash": "0" * 64,
                "hash": f"{len(self.events):064x}",
            }
        )


class _Extraction:
    def __init__(self, result: ExtractionResult) -> None:
        self._result = result

    async def run(self, document: ParsedDocument) -> ExtractionResult:
        return self._result


def _invalid_b() -> ExtractionResult:
    result = fx.extraction_result()
    a = result.a.model_copy(update={"finish_reason": "stop"})
    return result.model_copy(
        update={"a": a, "b": invalid_output(ExtractorId.B, "extractor-b", "length")}
    )


async def test_the_extract_audit_shows_an_invalid_extractor_b_and_why(tmp_path: Path) -> None:
    unused: Any = None
    parsed = FileParsedDocumentStore(tmp_path / "parsed")
    ref = parsed.put(fx.parsed_document())
    audit = _Audit()
    deps = InvoiceActivityDeps(
        documents=unused,
        scanner=unused,
        parser=unused,
        parsed=parsed,
        extraction=_Extraction(_invalid_b()),
        invoices=unused,
        feature_builder=unused,
        trust_gate=unused,
        validator=unused,
        fraud_engine=unused,
        vendors=unused,
        policy_engine=unused,
        ubl_builder=unused,
        schematron=unused,
        asp_client=unused,
        archive=unused,
        audit=audit,
        records=unused,
        session_factory=unused,
        idempotency=unused,
    )
    await DefaultInvoiceActivities(deps).extract(ref)
    (event,) = [e for e in audit.events if e.action == "activity:extract"]
    assert event.payload == {
        "disputed": ["iban", "total"],
        "has_arbiter": True,
        "a_valid": True,
        "b_valid": False,
        "a_finish_reason": "stop",
        "b_finish_reason": "length",
    }
    shutil.rmtree(tmp_path / "parsed")


def test_the_payload_is_typed_and_a_missing_finish_reason_is_null() -> None:
    audit = extract_audit(fx.extraction_result())
    assert isinstance(audit, ExtractAudit)
    assert set(ExtractAudit.model_fields) == {
        "disputed",
        "has_arbiter",
        "a_valid",
        "b_valid",
        "a_finish_reason",
        "b_finish_reason",
    }
    dumped = audit.model_dump(mode="json")
    assert dumped["a_valid"] is True and dumped["b_valid"] is True
    assert dumped["a_finish_reason"] is None and dumped["b_finish_reason"] is None

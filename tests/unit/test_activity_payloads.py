"""Every type that crosses the `InvoiceWorkflow` Temporal boundary (each registered activity's
arguments and result, plus the workflow input, state and review signal) must round-trip
through the worker's real `pydantic_data_converter` and stay far below Temporal's ~2 MB payload
limit. Regression for the live-e2e failure where `parse` returned page PNGs as raw bytes, and
for an adversarial PDF whose uncapped scan flags alone approach that limit."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import pytest
from temporalio import activity
from temporalio.contrib.pydantic import pydantic_data_converter

from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt
from apx.common.invoice import Invoice, InvoiceKind, VatCategory
from apx.common.types import Route
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument
from apx.extract.interfaces import ExtractionResult
from apx.fraud.interfaces import FraudReport
from apx.parse.interfaces import ParsedDocument, ParsedDocumentRef
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.interfaces import (
    PayeeMatch,
    PolicyThresholds,
    RoutedPayee,
    RoutingDecision,
    RoutingInput,
    payee_iban_digest,
)
from apx.scan.cap import MAX_FLAGS_PER_KIND
from apx.scan.detector import PdfHiddenContentDetector
from apx.scan.interfaces import ScanFlagKind, ScanReport
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import ValidationReport
from apx.workflows.activities import DefaultInvoiceActivities
from apx.workflows.components import DetectorScanner
from apx.workflows.corrections import LineCorrection, ReviewCorrections
from apx.workflows.interfaces import InvoiceWorkflowInput, InvoiceWorkflowState
from apx.workflows.review_guard import ReviewGuardActivities
from apx.workflows.workflow import ReviewDecision
from tests.fixtures.scan import pdf_builder
from tests.unit import _payload_fixtures as fx
from tests.unit import _policy_factories as pf

PAYLOAD_BUDGET_BYTES = 256 * 1024
"""Well under Temporal's 2 MB per-payload hard limit, leaving room for history growth."""

FIXTURES: dict[type[Any], Callable[[], object]] = {
    InvoiceWorkflowInput: fx.workflow_input,
    InvoiceWorkflowState: fx.workflow_state,
    ReviewDecision: lambda: ReviewDecision(approved=True),
    ReviewCorrections: lambda: ReviewCorrections(
        kind=InvoiceKind.INVOICE,
        currency="AED",
        lines=(LineCorrection(index=1, vat_category=VatCategory.STANDARD, vat_rate=Decimal("5")),),
    ),
    ReviewCorrections | None: lambda: ReviewCorrections(
        kind=InvoiceKind.CREDIT_NOTE,
        currency="AED",
        lines=(LineCorrection(index=1, vat_category=VatCategory.STANDARD, vat_rate=Decimal("5")),),
        supplier_address="Office 12, Al Quoz, Dubai",
    ),
    ScanReport: fx.scan_report,
    ParsedDocument: fx.parsed_document,
    ParsedDocumentRef: fx.parsed_ref,
    ExtractionResult: fx.extraction_result,
    TrustDecision: fx.trust_decision,
    Invoice: fx.invoice,
    ValidationReport: fx.validation_report,
    FraudReport: fx.fraud_report,
    RoutingDecision: fx.routing_decision,
    RoutedPayee | None: lambda: RoutedPayee(
        masked_iban="****3456",
        match=PayeeMatch.MATCHED,
        verified_count=2,
        iban_sha256=payee_iban_digest("AE070331234567890123456"),
    ),
    EInvoiceDocument: fx.einvoice_document,
    AspReceipt: fx.asp_receipt,
    ArchiveRecord: fx.archive_record,
    str: fx.idempotency_key,
    bool: lambda: True,
    Route: lambda: Route.REVIEW,
    type(None): lambda: None,
}


def _activity_definitions() -> list[Any]:
    definitions = [
        activity._Definition.from_callable(member)
        for cls in (DefaultInvoiceActivities, ReviewGuardActivities)
        for member in vars(cls).values()
        if callable(member)
    ]
    return [d for d in definitions if d is not None]


def _boundary_types() -> list[tuple[str, type[Any]]]:
    types: list[tuple[str, type[Any]]] = [
        ("workflow input", InvoiceWorkflowInput),
        ("workflow result/status query", InvoiceWorkflowState),
        ("review signal", ReviewDecision),
    ]
    for definition in _activity_definitions():
        for i, arg_type in enumerate(definition.arg_types or ()):
            types.append((f"{definition.name} arg {i}", arg_type))
        types.append((f"{definition.name} result", definition.ret_type))
    return types


def test_every_registered_activity_is_covered() -> None:
    names = {d.name for d in _activity_definitions()}
    assert names == {
        "scan",
        "parse",
        "extract",
        "trust_gate",
        "build_invoice",
        "validate",
        "fraud",
        "route",
        "review_hold",
        "review_note",
        "confirm_review",
        "apply_review",
        "build_einvoice",
        "send_to_asp",
        "archive",
        "mark_failed",
    }
    missing = {label: t for label, t in _boundary_types() if t not in FIXTURES}
    assert not missing, f"add a realistic fixture for {missing}"


@pytest.mark.parametrize(("label", "boundary_type"), _boundary_types(), ids=lambda v: str(v))
def test_boundary_type_round_trips_under_budget(label: str, boundary_type: type[Any]) -> None:
    value = FIXTURES[boundary_type]()
    converter = pydantic_data_converter.payload_converter

    payloads = converter.to_payloads([value])
    (restored,) = converter.from_payloads(payloads, [boundary_type])

    assert restored == value, label
    size = sum(p.ByteSize() for p in payloads)
    assert size < PAYLOAD_BUDGET_BYTES, f"{label}: {size} bytes"


def test_parse_never_crosses_the_boundary_as_a_document() -> None:
    # The 5-page parse itself is neither JSON-safe (raw PNG bytes) nor small; only its ref may
    # travel between activities.
    boundary = {t for _label, t in _boundary_types()}
    assert ParsedDocument not in boundary
    assert ParsedDocumentRef in boundary


def test_einvoice_xml_bytes_survive_any_encoding() -> None:
    # UBL XML is normally UTF-8, but a Latin-1 or UTF-16 document must not break (or be
    # silently altered in) the send_to_asp/archive payloads.
    xml = "<?xml version='1.0' encoding='ISO-8859-1'?><Invoice>café</Invoice>".encode("latin-1")
    document = fx.einvoice_document().model_copy(
        update={"xml": xml, "type_code": DocumentTypeCode.INVOICE}
    )
    converter = pydantic_data_converter.payload_converter

    (restored,) = converter.from_payloads(converter.to_payloads([document]), [EInvoiceDocument])

    assert restored.xml == xml


def _route(scan: ScanReport) -> RoutingDecision:
    thresholds = PolicyThresholds(
        approval_limit=pf.money("10000.00"),
        fraud_review=0.5,
        scan_review=0.5,
        attack_block=0.9,
        sod_cooldown_days=30,
    )
    vendor = pf.make_vendor()
    return DefaultPolicyEngine(thresholds).route(
        RoutingInput(
            invoice=pf.make_invoice(vendor_id=vendor.id),
            vendor=vendor,
            trust=pf.make_trust_decision(),
            validation=pf.make_validation_report(),
            fraud=pf.make_fraud_report(),
            scan=scan,
            at=datetime(2026, 9, 23, tzinfo=UTC),
            bank_changes=(),
        )
    )


@pytest.mark.parametrize(
    ("hidden_attack", "expected_route"),
    [(True, Route.BLOCK), (False, Route.REVIEW)],
    ids=["flood-plus-hidden-instruction", "flood-only"],
)
def test_zero_width_flood_scan_is_capped_under_budget_and_routes_unchanged(
    hidden_attack: bool, expected_route: Route
) -> None:
    # 5,000 zero-width characters: uncapped, the real detector emits ~1.9 MB of flags (one per
    # character), which is at Temporal's 2 MB limit and is sent twice (scan result, route arg).
    pdf = pdf_builder.zero_width_flood_pdf(5000, hidden_attack=hidden_attack)
    found = PdfHiddenContentDetector().detect(pdf)
    report = DetectorScanner(PdfHiddenContentDetector(), guard=None).scan(fx.DOC_SHA, pdf)
    converter = pydantic_data_converter.payload_converter

    payloads = converter.to_payloads([report])
    (restored,) = converter.from_payloads(payloads, [ScanReport])

    zero_width = [f for f in report.flags if f.kind is ScanFlagKind.ZERO_WIDTH]
    assert len(zero_width) == MAX_FLAGS_PER_KIND
    assert all(f.bbox is not None and f.page == 0 for f in zero_width)
    assert report.suppressed[ScanFlagKind.ZERO_WIDTH] == 5000 - MAX_FLAGS_PER_KIND
    assert restored == report
    assert sum(p.ByteSize() for p in payloads) < PAYLOAD_BUDGET_BYTES
    uncapped = report.model_copy(update={"flags": tuple(found), "suppressed": {}})
    assert max(f.score for f in report.flags) == max(f.score for f in found)
    assert _route(report) == _route(uncapped)
    assert _route(report).route is expected_route

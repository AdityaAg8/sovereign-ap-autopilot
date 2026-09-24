"""Tests for the M5.6 production adapters in `apx.workflows.components` and the pure parts of
`apx.workflows.sql`: grounded fields to `Invoice`, scan scoring, rules over a history snapshot,
UBL paying only the verified IBAN, and the choice of that IBAN."""

from __future__ import annotations

import io
import random
from datetime import UTC, date, datetime
from decimal import Decimal

import pytest
from PIL import Image

from apx.common.errors import EInvoiceError, IngestError, ModelOutputError, TrustGateError
from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.master import BankAccount, BankDetailChange, Vendor
from apx.common.types import (
    FieldValue,
    InvoiceId,
    Language,
    Money,
    Sha256Hex,
    UserId,
    VendorId,
)
from apx.einvoice.interfaces import DocumentTypeCode
from apx.extract.interfaces import (
    ExtractionResult,
    ExtractorId,
    ExtractorOutput,
    FieldName,
    InvoiceTypeCode,
    RawLineItem,
)
from apx.fraud.rules import BankDetailChangeRule, DuplicateInvoiceRule
from apx.scan.interfaces import ScanFlag, ScanFlagKind
from apx.scan.text import build_evidence
from apx.trust.interfaces import FieldFeatures
from apx.workflows.activities import verified_payment_iban
from apx.workflows.components import (
    AgreementScore,
    DetectorScanner,
    GroundedInvoiceBuilder,
    PintAeUblBuilder,
    RuleFraudEngine,
    StoredPageSource,
)
from apx.workflows.interfaces import InvoiceWorkflowInput
from apx.workflows.sql import HistorySnapshot

SHA = Sha256Hex("b" * 64)
VENDOR = VendorId("V-001")
SUPPLIER_TRN = "100000000000003"
VERIFIED_IBAN = "AE070331234567890123456"
CLAIMED_IBAN = "AE460090000000123456789"
INP = InvoiceWorkflowInput(invoice_id=InvoiceId(SHA), document_sha256=SHA, filename="inv.pdf")


def _fields(**overrides: str | None) -> dict[FieldName, FieldValue]:
    values: dict[FieldName, str | None] = {
        FieldName.INVOICE_NUMBER: "INV-7",
        FieldName.ISSUE_DATE: "05/03/2026",
        FieldName.DUE_DATE: "2026-04-04",
        FieldName.SUPPLIER_NAME: "Gulf Steel Trading LLC",
        FieldName.SUPPLIER_TRN: SUPPLIER_TRN,
        FieldName.SUPPLIER_ADDRESS: "Building 4, Al Quoz, Dubai, United Arab Emirates",
        FieldName.BUYER_NAME: "Sovereign Holdings",
        FieldName.CURRENCY: "AED",
        FieldName.SUBTOTAL: "1,000.00",
        FieldName.VAT_TOTAL: "50.00",
        FieldName.TOTAL: "1,050.00",
        FieldName.IBAN: CLAIMED_IBAN,
    }
    values.update({FieldName(k): v for k, v in overrides.items()})
    return {
        name: FieldValue(name=name.value, raw=raw, normalized=raw) for name, raw in values.items()
    }


def _result(
    grounded: dict[FieldName, FieldValue], lines: tuple[RawLineItem, ...] = ()
) -> ExtractionResult:
    out = ExtractorOutput(
        extractor=ExtractorId.A,
        model="a",
        fields=(),
        line_items=lines,
        logprobs={},
        valid_json=True,
        invoice_type_code=InvoiceTypeCode.INVOICE,
    )
    return ExtractionResult(
        document_sha256=SHA, a=out, b=out, arbiter=None, disputed=frozenset(), grounded=grounded
    )


class Resolver:
    async def vendor_id_for_trn(self, trn: str) -> VendorId | None:
        return VENDOR if trn == SUPPLIER_TRN else None


async def test_builder_maps_grounded_fields_and_resolves_the_vendor() -> None:
    lines = (
        RawLineItem(
            description="Steel beams",
            quantity="10",
            unit_price="100.00",
            net_amount="1,000.00",
            vat_rate="5%",
            vat_category=VatCategory.STANDARD,
        ),
        RawLineItem(description=None, quantity="1", unit_price="1", net_amount="1", vat_rate="5"),
    )
    invoice = await GroundedInvoiceBuilder(Resolver()).build(_result(_fields(), lines), INP)
    assert invoice.vendor_id == VENDOR
    assert invoice.issue_date == date(2026, 3, 5)
    assert invoice.total == Money(amount=Decimal("1050.00"), currency="AED")
    assert invoice.language is Language.EN
    assert [ln.vat_rate for ln in invoice.lines] == [Decimal("5")]
    assert invoice.lines[0].vat_category is VatCategory.STANDARD
    assert invoice.kind is InvoiceKind.INVOICE


async def test_builder_detects_arabic_and_unknown_vendor() -> None:
    fields = _fields(supplier_name="شركة الخليج للصلب", supplier_trn="100000000000999")
    invoice = await GroundedInvoiceBuilder(Resolver()).build(_result(fields), INP)
    assert invoice.language is Language.AR
    assert invoice.vendor_id is None
    assert invoice.supplier.name == "شركة الخليج للصلب"


@pytest.mark.parametrize("missing", ["invoice_number", "total", "issue_date"])
async def test_builder_refuses_missing_required_fields(missing: str) -> None:
    with pytest.raises(ModelOutputError, match=missing):
        await GroundedInvoiceBuilder(Resolver()).build(_result(_fields(**{missing: None})), INP)


async def test_builder_refuses_unreadable_amounts() -> None:
    with pytest.raises(ModelOutputError, match="unreadable"):
        await GroundedInvoiceBuilder(Resolver()).build(_result(_fields(total="lots")), INP)


class FixedDetector:
    def __init__(self, flags: list[ScanFlag]) -> None:
        self.flags = flags

    def detect(self, pdf_bytes: bytes) -> list[ScanFlag]:
        return self.flags


def _flag(kind: ScanFlagKind, score: float) -> ScanFlag:
    return ScanFlag(kind=kind, page=0, score=score, evidence=build_evidence(kind, "x"))


def test_scanner_scores_only_attack_kinds() -> None:
    scanner = DetectorScanner(
        FixedDetector([_flag(ScanFlagKind.PDF_METADATA, 0.9), _flag(ScanFlagKind.HIDDEN_TEXT, 0.7)])
    )
    report = scanner.scan(SHA, b"%PDF-1.7 ...")
    assert report.injection_score == 0.7
    assert len(report.flags) == 2
    assert (
        DetectorScanner(FixedDetector([_flag(ScanFlagKind.HIDDEN_TEXT, 1.0)]))
        .scan(SHA, b"\x89PNG")
        .flags
        == ()
    )


def _invoice(
    invoice_id: str = SHA, *, iban: str | None = CLAIMED_IBAN, lines: bool = True
) -> Invoice:
    party = Party(name="Gulf Steel", trn=SUPPLIER_TRN, address="Al Quoz, Dubai")
    net = Money(amount=Decimal("1000.00"))
    return Invoice(
        id=InvoiceId(invoice_id),
        document_sha256=Sha256Hex(invoice_id),
        language=Language.EN,
        number="INV-7",
        issue_date=date(2026, 3, 5),
        due_date=None,
        supplier=party,
        buyer=Party(name="Sovereign Holdings", trn="100000000000011", address="Marina, Dubai"),
        vendor_id=VENDOR,
        po_number=None,
        lines=(
            InvoiceLine(
                description="Steel beams",
                quantity=Decimal("10"),
                unit_price=Money(amount=Decimal("100.00")),
                net_amount=net,
                vat_category=VatCategory.STANDARD,
                vat_rate=Decimal("5"),
            ),
        )
        if lines
        else (),
        subtotal=net,
        vat_total=Money(amount=Decimal("50.00")),
        total=Money(amount=Decimal("1050.00")),
        iban=iban,
    )


class FixedHistory:
    def __init__(self, snapshot: HistorySnapshot) -> None:
        self.snapshot_value = snapshot

    async def snapshot(self, invoice: Invoice) -> HistorySnapshot:
        return self.snapshot_value


async def test_fraud_engine_runs_rules_over_the_snapshot_excluding_the_subject() -> None:
    subject = _invoice()
    twin = _invoice("c" * 64)
    change = BankDetailChange(
        vendor_id=VENDOR,
        changed_by=UserId("u1"),
        changed_at=datetime(2026, 3, 1, tzinfo=UTC),
        old_iban=None,
        new_iban=CLAIMED_IBAN,
    )
    snapshot = HistorySnapshot(subject, (subject, twin), (change,), (VENDOR,), None)
    engine = RuleFraudEngine(
        (DuplicateInvoiceRule(), BankDetailChangeRule()), FixedHistory(snapshot)
    )
    report = await engine.assess(subject, [])
    assert {f.rule_id for f in report.flags} == {"duplicate_invoice", "bank_detail_change"}
    assert report.max_score == max(f.score for f in report.flags)
    alone = HistorySnapshot(subject, (subject,), (), (), None)
    clean = await RuleFraudEngine((DuplicateInvoiceRule(),), FixedHistory(alone)).assess(
        subject, []
    )
    assert clean.flags == () and clean.max_score == 0.0


def test_ubl_pays_the_verified_iban_never_the_claimed_one() -> None:
    document = PintAeUblBuilder(buyer_trade_license="TL-1").build(
        _invoice(), DocumentTypeCode.INVOICE, verified_payment_iban=VERIFIED_IBAN
    )
    assert VERIFIED_IBAN.encode() in document.xml
    assert CLAIMED_IBAN.encode() not in document.xml
    assert document.invoice_id == SHA
    with pytest.raises(EInvoiceError):
        PintAeUblBuilder().build(_invoice(), DocumentTypeCode.INVOICE, verified_payment_iban=None)


def _vendor(*accounts: tuple[str, bool]) -> Vendor:
    return Vendor(
        id=VENDOR,
        name="Gulf Steel",
        trn=SUPPLIER_TRN,
        bank_accounts=tuple(
            BankAccount(
                iban=iban,
                verified=ok,
                verified_by=None,
                verified_at=None,
                source="admin_callback" if ok else "unverified",
            )
            for iban, ok in accounts
        ),
    )


def test_verified_payment_iban_choice() -> None:
    assert verified_payment_iban(None, CLAIMED_IBAN) is None
    one = _vendor((VERIFIED_IBAN, True), (CLAIMED_IBAN, False))
    assert verified_payment_iban(one, CLAIMED_IBAN) == VERIFIED_IBAN
    two = _vendor((VERIFIED_IBAN, True), (CLAIMED_IBAN, True))
    assert verified_payment_iban(two, CLAIMED_IBAN) == CLAIMED_IBAN
    assert verified_payment_iban(two, None) is None
    assert verified_payment_iban(_vendor((CLAIMED_IBAN, False)), CLAIMED_IBAN) is None


def test_agreement_score_is_an_unfitted_interim_score() -> None:
    features = FieldFeatures(
        agreement=0.5,
        a_mean_logprob=-0.1,
        a_min_logprob=-0.2,
        b_mean_logprob=-0.1,
        b_min_logprob=-0.2,
        format_valid=True,
        ocr_support=0.8,
        arbiter_vote=None,
    )
    score = AgreementScore()
    assert score.predict([(FieldName.TOTAL, features)]) == [pytest.approx(0.4)]
    assert score.version.startswith("uncalibrated")
    with pytest.raises(TrustGateError):
        score.fit([])


class DictStore:
    """In-memory document reader keyed by sha256."""

    def __init__(self, docs: dict[str, bytes]) -> None:
        self.docs = docs

    def get(self, sha256: Sha256Hex) -> bytes:
        if sha256 not in self.docs:
            raise IngestError(f"missing {sha256}")
        return self.docs[sha256]


def _png(seed: int) -> bytes:
    rng = random.Random(seed)
    image = Image.new("L", (64, 64))
    image.putdata([rng.randrange(256) for _ in range(64 * 64)])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


async def test_near_duplicate_pages_flag_a_resubmitted_document() -> None:
    subject = _invoice()
    original = _invoice("c" * 64)
    missing = _invoice("f" * 64)
    snapshot = HistorySnapshot(subject, (subject, original, missing), (), (), None)
    engine = RuleFraudEngine(
        (), FixedHistory(snapshot), pages=StoredPageSource(DictStore({"c" * 64: _png(1)}))
    )
    report = await engine.assess(subject, [_png(1)])
    assert [f.rule_id for f in report.flags] == ["near_duplicate_page"]
    assert report.flags[0].related_invoices == ("c" * 64,)
    unrelated = await engine.assess(subject, [_png(2)])
    assert unrelated.flags == ()


def test_stored_page_source_renders_pdf_pages() -> None:
    buffer = io.BytesIO()
    Image.new("RGB", (200, 280), "white").save(buffer, format="PDF")
    pages = StoredPageSource(DictStore({"a" * 64: buffer.getvalue()})).pages("a" * 64)
    assert len(pages) == 1 and pages[0].startswith(b"\x89PNG")

"""Production adapters that turn the pipeline modules into the invoice workflow's ports (PRD §10
M5.6): grounded fields into a typed `Invoice`, the hidden-content detector into a `Scanner`, the
fraud rules into a `FraudEngine`, `Invoice` into PINT AE UBL, and the interim trust score used
until a calibrator is fitted on real review labels."""

from __future__ import annotations

import asyncio
import io
import logging
import re
from collections.abc import Mapping, Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from hashlib import sha256
from typing import Protocol

import pypdfium2 as pdfium

from apx.common.errors import (
    ConfigError,
    EInvoiceError,
    IngestError,
    ModelOutputError,
    TrustGateError,
)
from apx.common.fields import InvoiceFields, LineItemFields
from apx.common.invoice import Invoice, InvoiceKind, InvoiceLine, Party, VatCategory
from apx.common.normalize import NormalizationError, normalize_amount, normalize_date
from apx.common.types import FieldValue, InvoiceId, Language, Money, Sha256Hex, VendorId
from apx.einvoice.builder import build_ubl_xml
from apx.einvoice.interfaces import DocumentTypeCode, EInvoiceDocument
from apx.einvoice.schematron import PINT_AE_VERSION
from apx.extract.interfaces import (
    ExtractionResult,
    FieldName,
    InvoiceTypeCode,
    RawLineItem,
    resolve_classification,
)
from apx.fraud.interfaces import FraudFlag, FraudReport, InvoiceHistory, PageSimilarityIndex
from apx.fraud.rules import DuplicateInvoiceRule
from apx.scan.cap import cap_flags
from apx.scan.interfaces import HiddenContentDetector, ScanFlagKind, ScanReport
from apx.trust.interfaces import CalibrationMethod, FieldFeatures
from apx.validate.interfaces import NO_CURRENCY
from apx.workflows.interfaces import InvoiceWorkflowInput
from apx.workflows.sql import HistorySnapshot

log = logging.getLogger(__name__)
_ARABIC = re.compile(r"[\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]")
_ZERO = Decimal("0")
_REQUIRED = (
    FieldName.INVOICE_NUMBER,
    FieldName.ISSUE_DATE,
    FieldName.SUPPLIER_NAME,
    FieldName.BUYER_NAME,
    FieldName.SUBTOTAL,
    FieldName.VAT_TOTAL,
    FieldName.TOTAL,
)
UAE_EMIRATES = (
    "Abu Dhabi",
    "Dubai",
    "Sharjah",
    "Ajman",
    "Umm Al Quwain",
    "Ras Al Khaimah",
    "Fujairah",
)


class VendorResolver(Protocol):
    async def vendor_id_for_trn(self, trn: str) -> VendorId | None: ...


def _vat_rate_percent(raw: str) -> Decimal:
    """ "5%", "5", "5.00" and "0.05" all mean five percent; the model is not told which to use."""
    text = raw.strip()
    percent = text.endswith("%")
    value = normalize_amount(text.rstrip("%").strip())
    return value if percent or value >= 1 or value == 0 else value * 100


_KIND_FOR_TYPE_CODE: Mapping[InvoiceTypeCode, InvoiceKind] = {
    InvoiceTypeCode.INVOICE: InvoiceKind.INVOICE,
    InvoiceTypeCode.CREDIT_NOTE: InvoiceKind.CREDIT_NOTE,
}


def _rate(raw: str | None) -> Decimal | None:
    """The line's VAT rate in percent, or None when it is missing or unreadable (ADR-0012: never
    defaulted; the validator flags None and routes the invoice to review)."""
    if raw is None or not raw.strip():
        return None
    try:
        return _vat_rate_percent(raw)
    except (NormalizationError, InvalidOperation):
        return None


def _line(
    index: int, raw: RawLineItem, currency: str, category: VatCategory | None
) -> InvoiceLine | None:
    """One extracted line, or None when its description or an amount cell is missing or
    unreadable. A dropped line is never silent: the validator's line-sum check then fails against
    the subtotal. The category is the A/B-resolved one and a missing rate stays None."""
    if raw.description is None or not raw.description.strip():
        return None
    try:
        quantity = normalize_amount(raw.quantity or "")
        unit_price = normalize_amount(raw.unit_price or "")
        net = normalize_amount(raw.net_amount or "")
    except (NormalizationError, InvalidOperation):
        return None
    return InvoiceLine(
        description=raw.description.strip(),
        quantity=quantity,
        unit_price=Money(amount=unit_price, currency=currency),
        net_amount=Money(amount=net, currency=currency),
        vat_category=category,
        vat_rate=_rate(raw.vat_rate),
        po_line_ref=str(index),
    )


class GroundedInvoiceBuilder:
    """`InvoiceBuilder`: the grounded (final) field values plus Extractor A's line items become an
    `Invoice`, matched to the vendor master by supplier TRN. A missing required field is a typed
    `ModelOutputError`. The document kind and line VAT categories are the values A and B agree on
    (ADR-0012 §1.4), else None; a missing currency is "XXX". Nothing is guessed: every None or
    "XXX" becomes an error violation that routes to review."""

    def __init__(self, vendors: VendorResolver) -> None:
        self._vendors = vendors

    async def build(self, result: ExtractionResult, inp: InvoiceWorkflowInput) -> Invoice:
        fields = result.grounded
        missing = [name.value for name in _REQUIRED if not _raw(fields, name)]
        if missing:
            raise ModelOutputError(f"required invoice fields missing: {', '.join(missing)}")
        currency = _normalized(fields, FieldName.CURRENCY) or NO_CURRENCY
        resolved = resolve_classification(result.a, result.b)
        type_code = resolved.invoice_type_code
        try:
            issue_date = normalize_date(_raw(fields, FieldName.ISSUE_DATE) or "")
            due_raw = _raw(fields, FieldName.DUE_DATE)
            due_date: date | None = normalize_date(due_raw) if due_raw else None
            amounts = {
                name: Money(amount=normalize_amount(_raw(fields, name) or ""), currency=currency)
                for name in (FieldName.SUBTOTAL, FieldName.VAT_TOTAL, FieldName.TOTAL)
            }
        except NormalizationError as exc:
            raise ModelOutputError(f"unreadable date or amount: {exc}") from exc
        supplier_trn = _normalized(fields, FieldName.SUPPLIER_TRN)
        supplier_name = _raw(fields, FieldName.SUPPLIER_NAME) or ""
        lines_source = result.a if result.a.valid_json else result.b
        lines = tuple(
            line
            for i, raw in enumerate(lines_source.line_items, start=1)
            if (line := _line(i, raw, currency, resolved.line_vat_categories[i - 1])) is not None
        )
        return Invoice(
            id=inp.invoice_id,
            document_sha256=inp.document_sha256,
            language=Language.AR if _ARABIC.search(supplier_name) else Language.EN,
            number=(_raw(fields, FieldName.INVOICE_NUMBER) or "").strip(),
            issue_date=issue_date,
            due_date=due_date,
            supplier=Party(
                name=supplier_name.strip(),
                trn=supplier_trn,
                address=_raw(fields, FieldName.SUPPLIER_ADDRESS),
            ),
            buyer=Party(
                name=(_raw(fields, FieldName.BUYER_NAME) or "").strip(),
                trn=_normalized(fields, FieldName.BUYER_TRN),
                address=None,
            ),
            vendor_id=await self._vendors.vendor_id_for_trn(supplier_trn) if supplier_trn else None,
            po_number=_raw(fields, FieldName.PO_NUMBER),
            lines=lines,
            subtotal=amounts[FieldName.SUBTOTAL],
            vat_total=amounts[FieldName.VAT_TOTAL],
            total=amounts[FieldName.TOTAL],
            iban=_normalized(fields, FieldName.IBAN),
            kind=None if type_code is None else _KIND_FOR_TYPE_CODE[type_code],
        )


def _raw(fields: Mapping[FieldName, FieldValue], name: FieldName) -> str | None:
    value = fields.get(name)
    return value.raw if value is not None and value.raw and value.raw.strip() else None


def _normalized(fields: Mapping[FieldName, FieldValue], name: FieldName) -> str | None:
    value = fields.get(name)
    return value.normalized if value is not None and value.normalized else None


class DetectorScanner:
    """`Scanner` over the rule-based hidden-content detector (M6.1). Until the M6.4 guard
    classifier lands, the injection score is the strongest flag of a kind that only an attack
    produces: text a reader cannot see, or active content."""

    ATTACK_KINDS = frozenset(
        {
            ScanFlagKind.HIDDEN_TEXT,
            ScanFlagKind.INVISIBLE_TEXT,
            ScanFlagKind.PDF_JAVASCRIPT,
            ScanFlagKind.INJECTION,
        }
    )

    def __init__(self, detector: HiddenContentDetector) -> None:
        self._detector = detector

    def scan(self, document_sha256: Sha256Hex, pdf_bytes: bytes) -> ScanReport:
        found = self._detector.detect(pdf_bytes) if pdf_bytes.startswith(b"%PDF") else []
        score = max((f.score for f in found if f.kind in self.ATTACK_KINDS), default=0.0)
        flags, suppressed = cap_flags(found)
        return ScanReport(
            document_sha256=document_sha256,
            flags=flags,
            injection_score=score,
            suppressed=suppressed,
        )


class Rule(Protocol):
    """`apx.fraud.interfaces.FraudRule`, read-only: the rules are frozen dataclasses."""

    @property
    def rule_id(self) -> str: ...

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]: ...


class HistorySource(Protocol):
    async def snapshot(self, invoice: Invoice) -> HistorySnapshot: ...


class PageSource(Protocol):
    def pages(self, invoice_id: str) -> Sequence[bytes]: ...


class DocumentReader(Protocol):
    """The read half of `apx.ingest.interfaces.DocumentStore`."""

    def get(self, sha256: Sha256Hex) -> bytes: ...


class StoredPageSource:
    """The first pages of a stored document as PNGs (an invoice's id is its document's sha256),
    for perceptual-hash near-duplicate checks. Low resolution: the hash is 8x8 anyway."""

    def __init__(self, documents: DocumentReader, *, max_pages: int = 2, dpi: int = 50) -> None:
        self._documents = documents
        self._max_pages = max_pages
        self._scale = dpi / 72.0

    def pages(self, invoice_id: str) -> Sequence[bytes]:
        data = self._documents.get(Sha256Hex(invoice_id))
        if not data.startswith(b"%PDF"):
            return [data]
        pdf = pdfium.PdfDocument(data)
        try:
            out: list[bytes] = []
            for index in range(min(len(pdf), self._max_pages)):
                buffer = io.BytesIO()
                pdf[index].render(scale=self._scale).to_pil().save(buffer, format="PNG")
                out.append(buffer.getvalue())
            return out
        finally:
            pdf.close()


class VendorPageIndex(PageSimilarityIndex, Protocol):
    """A `PageSimilarityIndex` over one vendor's stored pages that can also store an invoice's
    pages (`apx.fraud.pg_page_similarity.VendorPageSimilarityIndex`)."""

    async def add(self, invoice_id: InvoiceId, page_images: Sequence[bytes]) -> object: ...


class PageIndexes(Protocol):
    """`apx.fraud.pg_page_similarity.PgPageSimilarityIndexes`: one scoped index per assessment."""

    def for_vendor(
        self, vendor_id: VendorId, *, exclude: InvoiceId | None = None
    ) -> VendorPageIndex: ...


class RuleFraudEngine:
    """`FraudEngine`: every M8.1 rule over one history snapshot, plus perceptual-hash
    near-duplicate pages against the same vendor's recent invoices when a `PageSource` is given,
    plus learned (Qwen3-VL-Embedding-2B) near-duplicate pages against the same vendor's stored
    pages when `page_index` is given; this invoice's pages are then stored for later ones.
    Both page checks need a resolved vendor. The isolation forest joins with its model (M8.2)."""

    def __init__(
        self,
        rules: Sequence[Rule],
        history: HistorySource,
        *,
        pages: PageSource | None = None,
        near_duplicates: DuplicateInvoiceRule | None = None,
        page_index: PageIndexes | None = None,
    ) -> None:
        self._rules = tuple(rules)
        self._history = history
        self._pages = pages
        self._near = near_duplicates or DuplicateInvoiceRule()
        self._page_index = page_index

    async def assess(self, invoice: Invoice, page_images: Sequence[bytes]) -> FraudReport:
        snapshot = await self._history.snapshot(invoice)
        flags: list[FraudFlag] = []
        for rule in self._rules:
            flags.extend(rule.evaluate(invoice, snapshot))
        if self._pages is not None and invoice.vendor_id is not None and page_images:
            candidates = await asyncio.to_thread(
                _candidate_pages,
                self._pages,
                snapshot.recent_for_vendor(invoice.vendor_id, self._near.window_days),
            )
            flags.extend(self._near.evaluate_pages(invoice, page_images, candidates))
        if self._page_index is not None and invoice.vendor_id is not None and page_images:
            index = self._page_index.for_vendor(invoice.vendor_id, exclude=invoice.id)
            flags.extend(await self._near.evaluate_similarity(invoice, page_images, index))
            # The index excludes this invoice's own pages (stored by a retried step) in the
            # query itself, so they can never crowd real neighbours out of the top k.
            await index.add(invoice.id, page_images)
        max_score = max((f.score for f in flags), default=0.0)
        return FraudReport(flags=tuple(flags), max_score=max_score)


def _candidate_pages(
    pages: PageSource, recent: Sequence[Invoice]
) -> dict[InvoiceId, Sequence[bytes]]:
    candidates: dict[InvoiceId, Sequence[bytes]] = {}
    for other in recent:
        try:
            candidates[other.id] = pages.pages(other.id)
        except IngestError:
            # A prior invoice whose document is gone cannot be compared; the text rules still
            # cover it, so note it rather than fail every later assessment.
            log.warning("near-duplicate check skipped invoice %s: document missing", other.id)
    return candidates


def _emirate(address: str | None) -> str | None:
    if not address:
        return None
    lowered = address.lower()
    return next((e for e in UAE_EMIRATES if e.lower() in lowered), None)


class PintAeUblBuilder:
    """`UblBuilder`: maps an `Invoice` onto `InvoiceFields` and builds PINT AE UBL. The payment
    account is always the vendor master's verified IBAN, never the one printed on the invoice.
    The buyer is our own organisation, so its TRN (BT-48, and the BT-49 EndpointID) is the
    configured `buyer_trn`; the document's buyer TRN is only cross-checked by the validator
    (`check_buyer_trn`, ADR-0012 amendment 2)."""

    def __init__(
        self,
        *,
        buyer_trn: str,
        buyer_trade_license: str | None = None,
        buyer_address: str | None = None,
    ) -> None:
        if not buyer_trn.strip():
            raise ConfigError("the configured buyer TRN is blank")
        self._buyer_trn = buyer_trn.strip()
        self._buyer_license = buyer_trade_license
        self._buyer_address = buyer_address

    def build(
        self,
        invoice: Invoice,
        type_code: DocumentTypeCode,
        *,
        verified_payment_iban: str | None,
    ) -> EInvoiceDocument:
        unclassified = [
            i
            for i, line in enumerate(invoice.lines, start=1)
            if line.vat_category is None or line.vat_rate is None
        ]
        if unclassified:
            # A defence in depth: the API refuses an approval without these (422) and the
            # workflow applies the reviewer's values first (ADR-0012 amendment).
            raise EInvoiceError(
                f"invoice {invoice.id}: lines {unclassified} lack a VAT category or rate"
            )
        buyer_address = invoice.buyer.address or self._buyer_address
        fields = InvoiceFields(
            invoice_number=invoice.number,
            issue_date=invoice.issue_date.isoformat(),
            invoice_type_code=type_code.value,
            currency_code=invoice.total.currency,
            due_date=invoice.due_date.isoformat() if invoice.due_date else None,
            purchase_order_reference=invoice.po_number,
            supplier_name=invoice.supplier.name,
            supplier_trn=invoice.supplier.trn,
            supplier_address_line1=invoice.supplier.address,
            supplier_city=_emirate(invoice.supplier.address),
            supplier_country_code=invoice.supplier.country,
            buyer_name=invoice.buyer.name,
            buyer_trn=self._buyer_trn,
            buyer_legal_registration_id=self._buyer_license,
            buyer_address_line1=buyer_address,
            buyer_city=_emirate(buyer_address),
            buyer_country_code=invoice.buyer.country,
            payment_account_iban=invoice.iban,
            sum_line_net_amount=_fmt(
                sum((ln.net_amount.amount for ln in invoice.lines), Decimal(0))
            ),
            tax_exclusive_amount=_fmt(invoice.subtotal.amount),
            vat_total_amount=_fmt(invoice.vat_total.amount),
            tax_inclusive_amount=_fmt(invoice.total.amount),
            payable_amount=_fmt(invoice.total.amount),
            line_items=tuple(
                LineItemFields(
                    line_id=str(i),
                    item_name=line.description,
                    item_description=line.description,
                    quantity=_fmt(line.quantity),
                    unit_price=_fmt(line.unit_price.amount),
                    net_amount=_fmt(line.net_amount.amount),
                    vat_category_code=_known(line.vat_category).value,
                    vat_rate=_fmt(_known(line.vat_rate)),
                )
                for i, line in enumerate(invoice.lines, start=1)
            ),
        )
        xml = build_ubl_xml(
            fields, type_code=type_code, verified_payment_iban=verified_payment_iban
        )
        return EInvoiceDocument(
            invoice_id=InvoiceId(invoice.id),
            type_code=type_code,
            xml=xml,
            xml_sha256=Sha256Hex(sha256(xml).hexdigest()),
            spec_version=PINT_AE_VERSION,
        )


def _known[T](value: T | None) -> T:
    """Narrow a value `PintAeUblBuilder.build` has already checked is present."""
    if value is None:
        raise EInvoiceError("VAT category or rate missing at UBL build")
    return value


def _fmt(amount: Decimal) -> str:
    return format(amount, "f")


class AgreementScore:
    """Interim `Calibrator` until M7.4 fits one on review labels: P(correct) is reported as A/B
    agreement times OCR support. Uncalibrated, so the gate built on it takes λ̂ = None and never
    auto-accepts; the number only orders fields for the reviewer."""

    @property
    def method(self) -> CalibrationMethod:
        """Protocol requires one; nothing is fitted, and `version` says so."""
        return CalibrationMethod.ISOTONIC

    @property
    def version(self) -> str:
        return "uncalibrated-agreement-v1"

    def fit(self, records: Sequence[object]) -> None:
        raise TrustGateError("AgreementScore is a fixed interim score; fit IsotonicCalibrator")

    def predict(self, items: Sequence[tuple[FieldName, FieldFeatures]]) -> list[float]:
        return [float(f.agreement) * float(f.ocr_support) for _, f in items]

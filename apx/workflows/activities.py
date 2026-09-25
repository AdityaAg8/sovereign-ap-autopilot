"""Concrete `apx.workflows.interfaces.InvoiceActivities`: each activity calls exactly one
pipeline module's Protocol (or, for ASP send and archive, an idempotency-guarded external
client) through the injected `InvoiceActivityDeps` registry, then appends an audit event.
Business logic for scanning, parsing, extraction, trust, validation, fraud, policy and
e-invoicing lives on their own module branches; this file wires, never re-implements, them
(PRD §3, §6.2 I4, §6.4, §10 M5.2)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import JsonValue
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio import activity
from temporalio.exceptions import ApplicationError

from apx.agent.interfaces import AgentContext, ExceptionAgent
from apx.agent.runner import record_review_note
from apx.asp_mock.interfaces import Archive, ArchiveRecord, AspClient, AspReceipt, AspSubmission
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import (
    EInvoiceError,
    ModelOutputError,
    ModelServingError,
    ParsedArtifactIntegrityError,
    ParsedArtifactMissingError,
    PayeeChangedError,
    PayeeRefusedError,
)
from apx.common.invoice import Invoice
from apx.common.master import Vendor
from apx.common.types import IdempotencyKey, Sha256Hex
from apx.db.enums import DbInvoiceStatus
from apx.db.models import Invoice as InvoiceRow
from apx.einvoice.interfaces import (
    EInvoiceDocument,
    SchematronValidator,
    UblBuilder,
)
from apx.einvoice.type_code import document_type_code
from apx.extract.interfaces import ExtractionResult
from apx.fraud.interfaces import FraudEngine, FraudReport
from apx.ingest.interfaces import DocumentStore
from apx.parse.interfaces import (
    DocumentParser,
    ParsedDocument,
    ParsedDocumentRef,
    ParsedDocumentStore,
)
from apx.policy.interfaces import (
    Payee,
    PolicyEngine,
    RoutedPayee,
    RoutingDecision,
    RoutingInput,
    payee_refusal_of,
    pin_payee,
    reasons_payload,
    select_payee,
)
from apx.scan.interfaces import Scanner, ScanReport
from apx.trust.interfaces import FeatureBuilder, TrustDecision, TrustGate
from apx.validate.interfaces import ValidationReport, Validator, ViolationCode
from apx.workflows.corrections import (
    MissingCorrectionsError,
    ReviewCorrections,
    apply_corrections,
)
from apx.workflows.idempotency import IdempotencyStore, SqlIdempotencyStore
from apx.workflows.interfaces import InvoiceWorkflowInput
from apx.workflows.ports import (
    ExtractionPipeline,
    InvoiceBuilder,
    InvoiceRecords,
    VendorDirectory,
)

_CLASSIFICATION_CODES = frozenset(
    {
        ViolationCode.INVOICE_TYPE_MISSING,
        ViolationCode.VAT_CATEGORY_MISSING,
        ViolationCode.VAT_RATE_MISSING,
        ViolationCode.VAT_CATEGORY_RATE_MISMATCH,
        ViolationCode.CURRENCY_INVALID,
    }
)
"""ADR-0012 values an approved review must have resolved before an e-invoice is built. A
category/rate mismatch is included: the builder refuses it (an E/O line with a non-zero rate
would otherwise lose its VAT), so approving it would only fail later, at the build."""

_UNREVIEWABLE_CODES = frozenset({ViolationCode.MISSING_MANDATORY})
"""Values an approved review cannot supply (no correction carries them, e.g. a missing supplier
TRN) and the e-invoice build refuses. `apply_review` stops them before the build; the API
already refuses such an approval (`SupplierTrnMissingError`), so this is defence in depth."""


@dataclass(frozen=True, slots=True)
class InvoiceActivityDeps:
    """Everything `DefaultInvoiceActivities` needs, injected once at worker start-up.
    `session_factory` must behave like `apx.db.session.get_session`: commit on clean exit,
    roll back on error."""

    documents: DocumentStore
    scanner: Scanner
    parser: DocumentParser
    parsed: ParsedDocumentStore
    """Holds each parse (page PNGs included) so activities exchange only a `ParsedDocumentRef`."""
    extraction: ExtractionPipeline
    invoices: InvoiceBuilder
    feature_builder: FeatureBuilder
    trust_gate: TrustGate
    validator: Validator
    fraud_engine: FraudEngine
    vendors: VendorDirectory
    policy_engine: PolicyEngine
    ubl_builder: UblBuilder
    schematron: SchematronValidator
    """The Invoice-family rule set (380 and 480)."""
    asp_client: AspClient
    archive: Archive
    audit: AuditLog
    records: InvoiceRecords
    """Keeps the `invoices`/`fields`/`fraud_flags` rows the API and review UI read in step."""
    session_factory: Callable[[], AbstractAsyncContextManager[AsyncSession]]
    credit_note_schematron: SchematronValidator | None = None
    """The CreditNote-family rule set (381 and 81); a credit note with none wired is an error."""
    idempotency: IdempotencyStore = field(default_factory=SqlIdempotencyStore)
    clock: Callable[[], datetime] = lambda: datetime.now(UTC)
    exception_agent: ExceptionAgent | None = None
    """Writes the cited review note for REVIEW-routed invoices (M9.3); None disables notes."""


def verified_payment_iban(vendor: Vendor | None, claimed: str | None) -> str | None:
    """The account to pay in the UBL: `select_payee(vendor, invoice_iban=claimed).iban`, the one
    P1 decision (ADR-0014), or None on PayeeRefusedError so the UBL builder keeps refusing an
    Invoice without one (EInvoiceError). The claimed IBAN only selects among the vendor's
    admin-verified accounts and is never used as-is."""
    try:
        return select_payee(vendor, invoice_iban=claimed).iban
    except PayeeRefusedError:
        return None


class DefaultInvoiceActivities:
    """Structurally satisfies `apx.workflows.interfaces.InvoiceActivities`. Instances (not the
    class) are registered with the Temporal `Worker`, one bound method per activity, so the
    instance can carry `deps` (PRD §8: no per-activity globals)."""

    def __init__(self, deps: InvoiceActivityDeps) -> None:
        self._deps = deps

    async def _audit(self, step: str, invoice_id: str, payload: dict[str, JsonValue]) -> None:
        await self._deps.audit.append(
            AuditEvent(
                actor="workflow",
                action=f"activity:{step}",
                entity=f"invoice:{invoice_id}",
                payload=payload,
            )
        )

    @activity.defn(name="scan")
    async def scan(self, inp: InvoiceWorkflowInput) -> ScanReport:
        data = self._deps.documents.get(inp.document_sha256)
        # Off the event loop: the scan renders every page and runs the CPU injection guard
        # (ADR-0017), seconds of work that would otherwise stall heartbeats and other activities.
        report = await asyncio.to_thread(self._deps.scanner.scan, inp.document_sha256, data)
        await self._audit(
            "scan",
            inp.invoice_id,
            {
                "injection_score": report.injection_score,
                "flags": len(report.flags),
                "suppressed": sum(report.suppressed.values()),
            },
        )
        return report

    async def _parse(self, document_sha256: Sha256Hex) -> ParsedDocument:
        data = self._deps.documents.get(document_sha256)
        return await self._deps.parser.parse(document_sha256, data)

    async def _load_parsed(self, ref: ParsedDocumentRef) -> ParsedDocument:
        """The stored parse `ref` names. A missing entry (for example a wiped data dir) is
        audited and re-parsed, and the re-parse must reproduce `ref` exactly: words, boxes and
        page images downstream activities saw must not change under them. A corrupt entry
        raises `ParsedArtifactIntegrityError` and is never repaired silently."""
        try:
            return self._deps.parsed.load(ref)
        except ParsedArtifactMissingError:
            await self._audit(
                "parse_cache_miss", ref.document_sha256, {"content_sha256": ref.content_sha256}
            )
        document = await self._parse(ref.document_sha256)
        reparsed = self._deps.parsed.put(document)
        if reparsed != ref:
            raise ParsedArtifactIntegrityError(
                f"re-parse of document {ref.document_sha256} produced content "
                f"{reparsed.content_sha256}, not the referenced {ref.content_sha256}"
            )
        return document

    @activity.defn(name="parse")
    async def parse(self, inp: InvoiceWorkflowInput) -> ParsedDocumentRef:
        """Parses and stores the document, returning only its ref: page PNGs are raw bytes and
        a multi-page parse exceeds Temporal's payload limit. Retry-safe: the store is
        content-addressed and `put` is idempotent."""
        document = await self._parse(inp.document_sha256)
        ref = self._deps.parsed.put(document)
        await self._audit(
            "parse",
            inp.invoice_id,
            {"pages": ref.page_count, "content_sha256": ref.content_sha256},
        )
        return ref

    @activity.defn(name="extract")
    async def extract(self, parsed: ParsedDocumentRef) -> ExtractionResult:
        document = await self._load_parsed(parsed)
        result = await self._deps.extraction.run(document)
        disputed: list[JsonValue] = [f.value for f in sorted(result.disputed)]
        await self._audit(
            "extract",
            document.document_sha256,
            {"disputed": disputed, "has_arbiter": result.arbiter is not None},
        )
        return result

    @activity.defn(name="trust_gate")
    async def trust_gate(
        self, result: ExtractionResult, parsed: ParsedDocumentRef
    ) -> TrustDecision:
        document = await self._load_parsed(parsed)
        features = self._deps.feature_builder.features(result, document)
        decision = self._deps.trust_gate.decide(features)
        await self._audit(
            "trust_gate",
            document.document_sha256,
            {
                "all_accepted": decision.all_accepted,
                "fields": len(decision.fields),
                "calibrator_version": decision.calibrator_version,
                "refusal": decision.refusal,
            },
        )
        return decision

    @activity.defn(name="build_invoice")
    async def build_invoice(self, result: ExtractionResult, inp: InvoiceWorkflowInput) -> Invoice:
        invoice = await self._deps.invoices.build(result, inp)
        await self._deps.records.save(invoice, result)
        await self._audit(
            "build_invoice",
            inp.invoice_id,
            {"total": str(invoice.total.amount), "currency": invoice.total.currency},
        )
        return invoice

    @activity.defn(name="validate")
    async def validate(self, invoice: Invoice) -> ValidationReport:
        report = self._deps.validator.validate(invoice)
        await self._deps.records.set_status(invoice.id, DbInvoiceStatus.VALIDATED)
        # `ValidationReport.passed` is computed by `apx.validate` (its own module branch); audit
        # the raw violation severities instead of depending on that not-yet-landed property.
        errors = sum(1 for v in report.violations if v.severity == "error")
        await self._audit(
            "validate", invoice.id, {"violations": len(report.violations), "errors": errors}
        )
        return report

    @activity.defn(name="apply_review")
    async def apply_review(self, invoice: Invoice, corrections: ReviewCorrections) -> Invoice:
        """ADR-0012 amendment: the approved reviewer corrections (kind, line VAT category and
        rate) applied to the invoice before it is re-validated and turned into an e-invoice."""
        corrected = apply_corrections(invoice, corrections)
        # Re-validated here, not via the `validate` activity, which would move the row's status
        # back from APPROVED. A classified value still unresolved is a non-retryable failure.
        report = self._deps.validator.validate(corrected)
        unresolved = sorted(
            {
                v.code.value
                for v in report.violations
                if v.code in _CLASSIFICATION_CODES | _UNREVIEWABLE_CODES and v.severity == "error"
            }
        )
        await self._audit(
            "apply_review",
            invoice.id,
            {
                "corrections": corrections.model_dump(mode="json"),
                "violations": [v.code.value for v in report.violations],
            },
        )
        if unresolved:
            raise MissingCorrectionsError(
                f"invoice {invoice.id} still has unresolved values after review: {unresolved}"
            )
        await self._deps.records.set_kind(invoice.id, corrections.kind)
        return corrected

    @activity.defn(name="fraud")
    async def fraud(self, invoice: Invoice, parsed: ParsedDocumentRef) -> FraudReport:
        document = await self._load_parsed(parsed)
        page_images = [page.image_png for page in document.pages]
        report = await self._deps.fraud_engine.assess(invoice, page_images)
        await self._deps.records.flags(invoice.id, report.flags)
        await self._audit(
            "fraud", invoice.id, {"max_score": report.max_score, "flags": len(report.flags)}
        )
        return report

    @activity.defn(name="route")
    async def route(
        self,
        invoice: Invoice,
        trust: TrustDecision,
        validation: ValidationReport,
        fraud: FraudReport,
        scan: ScanReport,
    ) -> RoutingDecision:
        vendor, bank_changes = await self._deps.vendors.lookup(invoice.vendor_id)
        routing_input = RoutingInput(
            invoice=invoice,
            vendor=vendor,
            trust=trust,
            validation=validation,
            fraud=fraud,
            scan=scan,
            at=self._deps.clock(),
            bank_changes=bank_changes,
        )
        decision = self._deps.policy_engine.route(routing_input)
        # Persists the route and the pinned payee (masked, or the refusal) that the agent view
        # and the review UI show, so they read select_payee's choice (ADR-0014 §7).
        await self._deps.records.routed(invoice.id, decision, trust)
        refusal = payee_refusal_of(decision)
        await self._audit(
            "route",
            invoice.id,
            {
                "route": decision.route.value,
                "reasons": [r.rule.value for r in decision.reasons],
                "detail": reasons_payload(decision.reasons),
                "payee_refusal": None if refusal is None else refusal.value,
                "payee_match": None if decision.payee is None else decision.payee.match.value,
                "payee": None if decision.payee is None else decision.payee.masked_iban,
            },
        )
        return decision

    @activity.defn(name="review_note")
    async def review_note(self, invoice: Invoice, routing: RoutingDecision) -> bool:
        """Ask the exception agent for a cited review note on a REVIEW-routed invoice (PRD §10
        M9.3). The note is advisory, so a model outage or an unverifiable note never holds up the
        human review: either outcome is recorded as its own audit event instead, and only a
        grounded note is stored. Returns whether a note was stored."""
        agent = self._deps.exception_agent
        if agent is None:
            return False
        context = AgentContext(
            invoice_id=invoice.id,
            routing=routing,
            state={
                "route": routing.route.value,
                "rules": [reason.rule.value for reason in routing.reasons],
                "vendor_id": invoice.vendor_id,
                "total": invoice.total.model_dump(mode="json"),
                "line_count": len(invoice.lines),
            },
            quoted_document=(),
        )
        try:
            reply = await agent.handle(context)
        except (ModelServingError, ModelOutputError) as exc:
            await self._agent_outcome(
                invoice.id, "agent.review_note_unavailable", type(exc).__name__
            )
            return False
        if reply.refusal is not None:
            await self._agent_outcome(invoice.id, "agent.review_note_refused", reply.refusal)
            return False
        await record_review_note(self._deps.audit, invoice.id, reply)
        return True

    async def _agent_outcome(self, invoice_id: str, action: str, code: str) -> None:
        await self._deps.audit.append(
            AuditEvent(
                actor="agent", action=action, entity=f"invoice:{invoice_id}", payload={"code": code}
            )
        )

    async def _pinned_payee_iban(self, invoice: Invoice, pinned: RoutedPayee | None) -> str | None:
        """ADR-0014 §7 (TOCTOU): the IBAN to pay, re-selected from the live vendor master, only if
        it is the payee pinned at routing (same IBAN digest and match; both None when P1 refused
        then and now). Any other outcome is audited and raised as a non-retryable
        `PayeeChangedError`: nothing is built, so neither account is paid."""
        vendor, _ = await self._deps.vendors.lookup(invoice.vendor_id)
        payee: Payee | None = None
        refusal: str | None = None
        try:
            payee = select_payee(vendor, invoice_iban=invoice.iban)
        except PayeeRefusedError as refused:
            refusal = refused.refusal.value
        live = None if payee is None else pin_payee(payee)
        if pinned is None:
            same = live is None
        else:
            same = (
                live is not None
                and live.iban_sha256 == pinned.iban_sha256
                and live.match is pinned.match
            )
        if same:
            return None if payee is None else payee.iban
        await self._audit(
            "payee_changed",
            invoice.id,
            {
                "pinned": None if pinned is None else pinned.masked_iban,
                "pinned_match": None if pinned is None else pinned.match.value,
                "live": None if live is None else live.masked_iban,
                "live_match": None if live is None else live.match.value,
                "live_refusal": refusal,
            },
        )
        exc = PayeeChangedError(
            f"invoice {invoice.id}: the payee selected now differs from the one pinned at "
            "routing (vendor master or selection changed); the e-invoice is not built"
        )
        # Raised as a non-retryable ApplicationError of the same type name: a retry would only
        # re-read the same master (and NON_RETRYABLE_ERRORS is owned elsewhere).
        raise ApplicationError(str(exc), type=type(exc).__name__, non_retryable=True) from exc

    @activity.defn(name="build_einvoice")
    async def build_einvoice(
        self, invoice: Invoice, pinned_payee: RoutedPayee | None
    ) -> EInvoiceDocument:
        """PINT AE UBL for the approved (or AUTO) invoice. `pinned_payee` is the route decision's
        `RoutingDecision.payee`; the live selection must equal it (`_pinned_payee_iban`)."""
        payment_iban = await self._pinned_payee_iban(invoice, pinned_payee)
        type_code = document_type_code(invoice.kind, (ln.vat_category for ln in invoice.lines))
        schematron = (
            self._deps.schematron if type_code.is_invoice else self._deps.credit_note_schematron
        )
        if schematron is None:
            raise EInvoiceError(f"no credit-note schematron wired for invoice {invoice.id}")
        document = self._deps.ubl_builder.build(
            invoice,
            type_code,
            verified_payment_iban=payment_iban,
        )
        result = schematron.validate(document.xml)
        fatal = [f for f in result.failures if f.flag == "fatal"]
        if fatal:
            raise EInvoiceError(f"schematron failed for invoice {invoice.id}: {fatal[0].message}")
        await self._audit(
            "build_einvoice",
            invoice.id,
            {"schematron_passed": result.passed, "failures": len(result.failures)},
        )
        return document

    @activity.defn(name="send_to_asp")
    async def send_to_asp(self, key: IdempotencyKey, document: EInvoiceDocument) -> AspReceipt:
        """Idempotent by `key` (§6.4): a retried call after a worker crash never resubmits."""
        async with self._deps.session_factory() as session:
            cached = await self._deps.idempotency.get(session, key)
        if cached is not None:
            await self._deps.records.set_status(document.invoice_id, DbInvoiceStatus.SENT)
            return AspReceipt.model_validate(cached)

        receipt = await self._deps.asp_client.submit(
            AspSubmission(idempotency_key=key, document=document)
        )

        async with self._deps.session_factory() as session:
            await self._deps.idempotency.put(
                session,
                invoice_id=document.invoice_id,
                step="asp_send",
                key=key,
                payload=receipt.model_dump(mode="json"),
            )
        await self._deps.records.set_status(document.invoice_id, DbInvoiceStatus.SENT)

        await self._audit(
            "send_to_asp",
            document.invoice_id,
            {"receipt_id": receipt.receipt_id, "status": receipt.status},
        )
        return receipt

    @activity.defn(name="archive")
    async def archive(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        """Idempotent by `key` (§6.4). The invoice's one terminal payment-state transition --
        `invoices.status` to `ARCHIVED` -- is written only by the attempt that first records
        `key`, so a retried call never re-applies it (M5.2 crash-recovery test)."""
        async with self._deps.session_factory() as session:
            cached = await self._deps.idempotency.get(session, key)
        if cached is not None:
            return ArchiveRecord.model_validate(cached)

        record = await self._deps.archive.store(key, document)

        async with self._deps.session_factory() as session:
            recorded = await self._deps.idempotency.put(
                session,
                invoice_id=document.invoice_id,
                step="archive",
                key=key,
                payload=record.model_dump(mode="json"),
            )
            if recorded:
                await session.execute(
                    update(InvoiceRow)
                    .where(InvoiceRow.id == document.invoice_id)
                    .values(status=DbInvoiceStatus.ARCHIVED)
                )

        await self._audit(
            "archive",
            document.invoice_id,
            {"xml_sha256": record.xml_sha256, "retain_until": record.retain_until.isoformat()},
        )
        return record

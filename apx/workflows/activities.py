"""Concrete `apx.workflows.interfaces.InvoiceActivities`: each activity calls exactly one
pipeline module's Protocol (or, for ASP send and archive, an idempotency-guarded external
client) through the injected `InvoiceActivityDeps` registry, then appends an audit event.
Business logic for scanning, parsing, extraction, trust, validation, fraud, policy and
e-invoicing lives on their own module branches; this file wires, never re-implements, them
(I4). Where an activity writes state in its own transaction
(`send_to_asp`, `archive`, `mark_failed`), the audit entry joins that transaction."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import JsonValue
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from temporalio import activity
from temporalio.exceptions import ApplicationError

from apx.agent.interfaces import AgentContext, ExceptionAgent
from apx.agent.runner import record_review_note
from apx.asp_mock.interfaces import Archive, ArchiveRecord, AspClient, AspReceipt, AspSubmission
from apx.audit.chain import append_in_transaction
from apx.audit.interfaces import AuditEvent, AuditLog
from apx.common.errors import (
    EInvoiceError,
    FailureNotRecordedError,
    ModelOutputError,
    ModelServingError,
    ParsedArtifactIntegrityError,
    ParsedArtifactMissingError,
    PayeeChangedError,
    PayeeRefusedError,
)
from apx.common.invoice import Invoice
from apx.common.master import Vendor
from apx.common.types import Frozen, IdempotencyKey, Sha256Hex
from apx.db.enums import DbInvoiceStatus
from apx.db.models import IngestedDocument as IngestedDocumentRow
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
from apx.trust.gate import refuse_truncated_parse
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

_UNREVIEWABLE_CODES = frozenset({ViolationCode.MISSING_MANDATORY, ViolationCode.TOTALS_MISMATCH})
"""Errors the e-invoice build refuses that `apply_review` stops before it, re-checked after the
corrections: a missing supplier TRN (no correction carries it), a supplier address naming no
emirate (a corrected address clears it; ADR-0012 amendment 4) and amounts that do not add up
(no correction carries amounts). The API already refuses such approvals
(`MissingCorrectionsError` and its subclasses), so this is defence in depth."""

_NEVER_MARKED_FAILED = (
    DbInvoiceStatus.SENT,
    DbInvoiceStatus.ARCHIVED,
    DbInvoiceStatus.BLOCKED,
    DbInvoiceStatus.FAILED,
)
"""Terminal statuses `mark_failed` leaves alone: the invoice already reached the ASP, was kept
out of payment on purpose, or is already FAILED."""
_FAILED_ERROR_CHARS = 2000
"""The failure message is audited; a long activity error chain is cut to this length."""


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
    """Writes the cited review note for REVIEW-routed invoices; None disables notes."""


def verified_payment_iban(vendor: Vendor | None, claimed: str | None) -> str | None:
    """The account to pay in the UBL: `select_payee(vendor, invoice_iban=claimed).iban`, the one
    P1 decision (ADR-0014), or None on PayeeRefusedError so the UBL builder keeps refusing an
    Invoice without one (EInvoiceError). The claimed IBAN only selects among the vendor's
    admin-verified accounts and is never used as-is."""
    try:
        return select_payee(vendor, invoice_iban=claimed).iban
    except PayeeRefusedError:
        return None


class ExtractAudit(Frozen):
    """The `activity:extract` audit payload: the disputed fields, whether an arbiter ran, and
    each extractor's validity and finish reason, so an invalid Extractor B output (which makes
    every field disputed and the trust gate hard-reject, rule H1) is visible in the audit."""

    disputed: tuple[str, ...]
    has_arbiter: bool
    a_valid: bool
    b_valid: bool
    a_finish_reason: str | None
    b_finish_reason: str | None


def extract_audit(result: ExtractionResult) -> ExtractAudit:
    return ExtractAudit(
        disputed=tuple(f.value for f in sorted(result.disputed)),
        has_arbiter=result.arbiter is not None,
        a_valid=result.a.valid_json,
        b_valid=result.b.valid_json,
        a_finish_reason=result.a.finish_reason,
        b_finish_reason=result.b.finish_reason,
    )


def _activity_event(step: str, invoice_id: str, payload: dict[str, JsonValue]) -> AuditEvent:
    return AuditEvent(
        actor="workflow",
        action=f"activity:{step}",
        entity=f"invoice:{invoice_id}",
        payload=payload,
    )


class DefaultInvoiceActivities:
    """Structurally satisfies `apx.workflows.interfaces.InvoiceActivities`. Instances (not the
    class) are registered with the Temporal `Worker`, one bound method per activity, so the
    instance can carry `deps` (no per-activity globals)."""

    def __init__(self, deps: InvoiceActivityDeps) -> None:
        self._deps = deps

    async def _audit(self, step: str, invoice_id: str, payload: dict[str, JsonValue]) -> None:
        await self._deps.audit.append(_activity_event(step, invoice_id, payload))

    async def _audit_in(
        self,
        session: AsyncSession,
        step: str,
        invoice_id: str,
        payload: dict[str, JsonValue],
    ) -> None:
        """`_audit` inside `session`'s transaction: the entry commits with the state writes
        before it, or neither does (`apx.audit.chain.append_in_transaction`)."""
        await append_in_transaction(
            self._deps.audit, session, _activity_event(step, invoice_id, payload)
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
        content-addressed and `put` is idempotent. Pages the OCR service cut short are counted
        in the audit (ADR-0013 amendment 1); `trust_gate` then routes the invoice to review."""
        document = await self._parse(inp.document_sha256)
        ref = self._deps.parsed.put(document)
        await self._audit(
            "parse",
            inp.invoice_id,
            {
                "pages": ref.page_count,
                "content_sha256": ref.content_sha256,
                "truncated_pages": len(document.truncated_pages),
            },
        )
        return ref

    @activity.defn(name="extract")
    async def extract(self, parsed: ParsedDocumentRef) -> ExtractionResult:
        document = await self._load_parsed(parsed)
        result = await self._deps.extraction.run(document)
        await self._audit(
            "extract", document.document_sha256, extract_audit(result).model_dump(mode="json")
        )
        return result

    @activity.defn(name="trust_gate")
    async def trust_gate(
        self, result: ExtractionResult, parsed: ParsedDocumentRef
    ) -> TrustDecision:
        document = await self._load_parsed(parsed)
        features = self._deps.feature_builder.features(result, document)
        decision = refuse_truncated_parse(
            self._deps.trust_gate.decide(features),
            document.truncated_pages,
            len(document.pages),
        )
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
        rate, supplier address) applied to the invoice before it is re-validated and turned
        into an e-invoice."""
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
        """Ask the exception agent for a cited review note on a REVIEW-routed invoice.
        The note is advisory, so a model outage or an unverifiable note never holds up the
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
        """Idempotent by `key`: a retried call after a worker crash never resubmits. The
        idempotency record, `SENT` and the audit entry commit in one transaction, so a failed
        audit append leaves no record and the retry replays `submit` (idempotent by `key`, same
        receipt) and writes the event: exactly one `send_to_asp` entry per key."""
        async with self._deps.session_factory() as session:
            cached = await self._deps.idempotency.get(session, key)
        if cached is not None:
            await self._deps.records.set_status(document.invoice_id, DbInvoiceStatus.SENT)
            return AspReceipt.model_validate(cached)

        receipt = await self._deps.asp_client.submit(
            AspSubmission(idempotency_key=key, document=document)
        )

        async with self._deps.session_factory() as session:
            recorded = await self._deps.idempotency.put(
                session,
                invoice_id=document.invoice_id,
                step="asp_send",
                key=key,
                payload=receipt.model_dump(mode="json"),
            )
            await session.execute(
                update(InvoiceRow)
                .where(InvoiceRow.id == document.invoice_id)
                .values(status=DbInvoiceStatus.SENT)
            )
            if recorded:  # else a concurrent attempt recorded (and audited) this key first
                await self._audit_in(
                    session,
                    "send_to_asp",
                    document.invoice_id,
                    {"receipt_id": receipt.receipt_id, "status": receipt.status},
                )
        return receipt

    @activity.defn(name="archive")
    async def archive(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord:
        """Idempotent by `key`. The invoice's one terminal payment-state transition --
        `invoices.status` to `ARCHIVED` -- is written only by the attempt that first records
        `key`, so a retried call never re-applies it (crash-recovery test). The record, the
        status and the audit entry commit together, so a failed audit append leaves no record
        and the retry writes the event: exactly one `archive` entry per key."""
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
                await self._audit_in(
                    session,
                    "archive",
                    document.invoice_id,
                    {
                        "xml_sha256": record.xml_sha256,
                        "retain_until": record.retain_until.isoformat(),
                    },
                )
        return record

    @activity.defn(name="mark_failed")
    async def mark_failed(self, invoice_id: str, error: str) -> None:
        """Records that the workflow gave up (I4: nothing silently dropped), so the API and the
        review queue stop showing it as in flight. With an invoice row, writes FAILED to
        `invoices.status`; a terminal status is never overwritten (a SENT invoice whose archive
        failed stays SENT, since the ASP already has it). Without one (the run failed before
        `build_invoice` persisted it), writes `failed_at`/`failure` on the document's
        `ingested_documents` row (the invoice id is the document sha256), keeping the first
        failure. Neither row: `FailureNotRecordedError`, retried by the workflow until the
        ingest commit lands. Idempotent; audited once the record is written."""
        failure = error[:_FAILED_ERROR_CHARS]
        async with self._deps.session_factory() as session:
            has_invoice = await session.scalar(
                select(InvoiceRow.id).where(InvoiceRow.id == invoice_id)
            )
            if has_invoice is not None:
                await session.execute(
                    update(InvoiceRow)
                    .where(
                        InvoiceRow.id == invoice_id,
                        InvoiceRow.status.not_in(_NEVER_MARKED_FAILED),
                    )
                    .values(status=DbInvoiceStatus.FAILED)
                )
                recorded_on = "invoice"
            else:
                has_document = await session.scalar(
                    select(IngestedDocumentRow.sha256).where(
                        IngestedDocumentRow.sha256 == invoice_id
                    )
                )
                if has_document is None:
                    raise FailureNotRecordedError(
                        f"no invoice or ingested document {invoice_id} to record the failure on"
                    )
                await session.execute(
                    update(IngestedDocumentRow)
                    .where(
                        IngestedDocumentRow.sha256 == invoice_id,
                        IngestedDocumentRow.failed_at.is_(None),
                    )
                    .values(failed_at=datetime.now(UTC), failure=failure)
                )
                recorded_on = "document"
            await self._audit_in(
                session, "failed", invoice_id, {"error": failure, "recorded_on": recorded_on}
            )

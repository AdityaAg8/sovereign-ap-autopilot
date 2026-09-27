"""`DbInvoiceService`: the Postgres-backed `apx.api.interfaces.InvoiceService` implementation.
List/filter/paginate, invoice detail (masking `iban` fields from non-admins, docs/security.md
B5), field edits (audit event + label row), decisions and proposal execution (P1-P5, I3: every
payment-affecting action goes `PolicyGuard` -> `SqlPaymentStateWriter`, then signals the
invoice's workflow), a document-serving lookup and a polling-based SSE status stream (PRD §10
M5.3). Review decisions follow the invoice state machine: reject only while awaiting review,
send-to-review only before routing (a durable hold the workflow honours), else 409. Also
`DbVendorService`, the vendor list/detail read model for the M5.4 vendor view.
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

from pydantic import JsonValue
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apx.api.errors import NotFoundError
from apx.api.interfaces import (
    DecisionRequest,
    FieldEdit,
    FlagView,
    InvoiceDetail,
    InvoiceDocument,
    InvoiceFieldView,
    InvoicePage,
    InvoiceSummary,
    ProposalAcceptResponse,
    SseEvent,
    VendorBankDetailView,
    VendorDetail,
    VendorPage,
    VendorSummary,
)
from apx.api.queries import (
    invoice_count_stmt,
    invoice_list_stmt,
    vendor_count_stmt,
    vendor_list_stmt,
)
from apx.api.settings import ApiSettings, get_settings
from apx.api.signals import ReviewSignaller, UnconfiguredReviewSignaller
from apx.audit.chain import PgAuditLog
from apx.audit.interfaces import AuditEvent
from apx.common.errors import (
    InvoiceStateConflictError,
    PolicyDeniedError,
    ProposalNotExecutableError,
)
from apx.common.master import BankDetailChange, User
from apx.common.types import BBox, InvoiceId, Money, Role, Sha256Hex, UserId, VendorId
from apx.db.enums import (
    DbInvoiceStatus,
    DecisionAction,
    FieldSource,
    ProposalKind,
    ProposalStatus,
)
from apx.db.models import BankDetail, BankDetailHistory, Decision, Invoice, Vendor
from apx.db.models import Field as FieldRow
from apx.db.models import FraudFlag as FraudFlagRow
from apx.db.models import IngestedDocument as IngestedDocumentRow
from apx.db.models import Label as LabelRow
from apx.db.models import Proposal as ProposalRow
from apx.db.session import get_session
from apx.ingest.errors import DocumentNotFoundError
from apx.ingest.interfaces import DocumentStore
from apx.ingest.validate import detect_media_type
from apx.learning.splits import split_for_document
from apx.policy.engine import DefaultPolicyEngine
from apx.policy.guard import DefaultPolicyGuard
from apx.policy.interfaces import (
    ActionRequest,
    Capability,
    PolicyThresholds,
    RoutingDecision,
    capabilities_for,
)
from apx.policy.interfaces import Proposal as ProposalModel
from apx.policy.writer import (
    SqlPaymentStateWriter,
    bank_change_from_payload,
    invoice_for_digest,
    load_vendor,
    proposal_model,
)
from apx.validate.interfaces import ViolationCode
from apx.workflows.corrections import (
    SUPPLIER_ADDRESS_FIELD,
    ReviewCorrections,
    TotalsMismatchError,
    corrections_from_fields,
)
from apx.workflows.interfaces import InvoiceStatus

_TERMINAL_STATUSES = frozenset(
    {
        DbInvoiceStatus.SENT,
        DbInvoiceStatus.ARCHIVED,
        DbInvoiceStatus.BLOCKED,
        DbInvoiceStatus.FAILED,
    }
)
_PRE_ROUTE_STATUSES = frozenset(
    {
        DbInvoiceStatus.RECEIVED,
        DbInvoiceStatus.SCANNED,
        DbInvoiceStatus.PARSED,
        DbInvoiceStatus.EXTRACTED,
        DbInvoiceStatus.GATED,
        DbInvoiceStatus.VALIDATED,
        DbInvoiceStatus.FRAUD_CHECKED,
    }
)
"""Statuses before the workflow's `route` step: the only ones `send_to_review` may hold (H-02).
The hold is a `send_to_review` decision row that the workflow's `review_hold` activity turns an
AUTO route into REVIEW with; the status itself is left to the workflow."""
_POLL_INTERVAL_S = 1.0
_MAX_PAGE_SIZE = 200
_RECENT_INVOICES_LIMIT = 20
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DOCUMENT_EXTENSION = {"application/pdf": "pdf", "image/png": "png", "image/jpeg": "jpg"}


def _mask_iban(value: str) -> str:
    """Masks all but the last 4 characters (ADR-0008 T2)."""
    tail = value[-4:]
    return "*" * max(len(value) - len(tail), 0) + tail


def _to_summary(invoice: Invoice, vendor_name: str | None) -> InvoiceSummary:
    return InvoiceSummary(
        id=InvoiceId(invoice.id),
        status=InvoiceStatus(invoice.status.value),
        route=invoice.route,
        vendor_id=VendorId(invoice.vendor_id) if invoice.vendor_id else None,
        supplier_name=vendor_name,
        total=format(invoice.total, "f"),
        received_at=invoice.created_at,
    )


def failed_document_detail(document: IngestedDocumentRow) -> InvoiceDetail:
    """The detail of a document whose workflow failed before its invoice row existed
    (`mark_failed` wrote `failed_at`/`failure` on its `ingested_documents` row): status FAILED
    with the recorded error, no route, fields or decisions. The id is the document sha256,
    which is the invoice id the workflow would have used."""
    if document.failed_at is None or document.failure is None:
        raise ValueError(f"document {document.sha256} has no recorded failure")
    return InvoiceDetail(
        summary=InvoiceSummary(
            id=InvoiceId(document.sha256),
            status=InvoiceStatus.FAILED,
            route=None,
            vendor_id=None,
            supplier_name=None,
            total=None,
            received_at=document.received_at,
        ),
        fields=(),
        accepted={},
        reasons=(),
        proposals=(),
        error=document.failure,
    )


def _to_field_view(field: FieldRow, *, mask_iban: bool) -> InvoiceFieldView:
    value = field.value
    if mask_iban and value and field.name.lower() == "iban":
        value = _mask_iban(value)
    boxes = tuple(BBox.model_validate(box) for box in (field.boxes or []))
    return InvoiceFieldView(
        name=field.name,
        value=value,
        confidence=field.confidence,
        boxes=boxes,
        source=field.source,
    )


def corrections_for_rows(rows: Sequence[FieldRow]) -> ReviewCorrections:
    """`corrections_from_fields` over an invoice's field rows in id order: the latest value of
    each field with a reviewer's row winning, plus the latest model-sourced values, which are
    what the workflow's `Invoice` was built from (a human row never reaches it). Those are the
    normalized values, as `GroundedInvoiceBuilder` reads supplier TRN and currency, except the
    supplier address, which it reads raw (so an unedited address is never a "correction")."""
    latest: dict[str, str | None] = {}
    extracted: dict[str, str | None] = {}
    human: set[str] = set()
    for row in rows:
        is_human = row.source is FieldSource.HUMAN
        raw_read = row.name == SUPPLIER_ADDRESS_FIELD
        if not is_human:
            extracted[row.name] = row.value if raw_read else row.normalized_value
        if row.name in human and not is_human:
            continue
        latest[row.name] = row.value if raw_read else (row.normalized_value or row.value)
        if is_human:
            human.add(row.name)
    return corrections_from_fields(latest, extracted=extracted)


_CORRECTABLE_TOTALS = frozenset({"vat_total"})
"""`totals_mismatch` fields a reviewer's line VAT category/rate correction can clear."""


def refuse_totals_mismatch(route_reasons: Sequence[Mapping[str, object]] | None) -> None:
    """ADR-0012 amendment 4: raises `TotalsMismatchError` (422) while the invoice's newest
    `policy_route` decision (`routing_record`) lists an error-severity `totals_mismatch` that
    no review correction can clear: a line net amount, the subtotal or the total (or one with
    no field). No correction carries amounts and the PINT AE build refuses them, so the
    reviewer rejects. A `vat_total` finding is left alone: `check_vat` recomputes the VAT from
    the line categories and rates a reviewer may correct, and `apply_review` re-validates after
    the corrections. A decision routed before the amendment carries no findings; `apply_review`
    refuses the approval then (defence in depth)."""
    fields: list[str] = []
    for reason in route_reasons or ():
        findings = reason.get("violations")
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if (
                isinstance(finding, Mapping)
                and finding.get("code") == ViolationCode.TOTALS_MISMATCH.value
                and finding.get("field") not in _CORRECTABLE_TOTALS
            ):
                fields.append(str(finding.get("field") or "totals"))
    if fields:
        raise TotalsMismatchError(
            "cannot approve: validation found amounts that do not add up "
            f"({', '.join(fields)}); no correction can change them and the PINT AE e-invoice "
            "would be refused, so reject the invoice"
        )


class DbInvoiceService:
    """`apx.api.interfaces.InvoiceService` backed by Postgres. Every method opens its own
    `AsyncSession` (matching `apx.audit.chain.PgAuditLog`'s convention) and every state change
    is followed by a separate, independently-committed audit append -- the audit log is never
    silently skipped just because the entity write also succeeded."""

    def __init__(
        self, settings: ApiSettings | None = None, *, signaller: ReviewSignaller | None = None
    ) -> None:
        self._settings = settings or get_settings()
        self._audit = PgAuditLog()
        self._signaller: ReviewSignaller = signaller or UnconfiguredReviewSignaller()
        thresholds = PolicyThresholds(
            approval_limit=Money(amount=self._settings.approval_limit, currency="AED"),
            # Routing thresholds: only `PolicyEngine.route` reads these, and the API never
            # routes (the workflow does); `authorize` uses the limit and the SoD window only.
            fraud_review=1.0,
            scan_review=1.0,
            attack_block=1.0,
            sod_cooldown_days=self._settings.sod_cooldown_days,
        )
        self._guard = DefaultPolicyGuard(DefaultPolicyEngine(thresholds), self._audit)
        self._writer = SqlPaymentStateWriter(
            get_session,
            self._audit,
            approval_limit=Money(amount=self._settings.approval_limit, currency="AED"),
        )

    async def list(
        self,
        status: InvoiceStatus | None,
        vendor_id: VendorId | None,
        q: str | None,
        limit: int,
        offset: int,
    ) -> InvoicePage:
        limit = max(1, min(limit, _MAX_PAGE_SIZE))
        offset = max(0, offset)
        async with get_session() as session:
            rows = (
                await session.execute(
                    invoice_list_stmt(
                        status=status, vendor_id=vendor_id, q=q, limit=limit, offset=offset
                    )
                )
            ).all()
            total = (
                await session.execute(invoice_count_stmt(status=status, vendor_id=vendor_id, q=q))
            ).scalar_one()
        items = tuple(_to_summary(invoice, vendor_name) for invoice, vendor_name in rows)
        return InvoicePage(items=items, total=total, limit=limit, offset=offset)

    async def get(self, invoice_id: InvoiceId, user: User) -> InvoiceDetail:
        is_admin = Role.ADMIN in user.roles
        async with get_session() as session:
            invoice = await session.get(Invoice, invoice_id)
            if invoice is None:
                # A workflow that failed before persisting its invoice leaves its failure on
                # the document row; one still in flight (or never ingested) is a plain 404.
                document = await session.get(IngestedDocumentRow, invoice_id)
                if document is not None and document.failed_at is not None:
                    return failed_document_detail(document)
                raise NotFoundError(f"invoice {invoice_id} not found")

            vendor_name = None
            if invoice.vendor_id is not None:
                vendor_name = await session.scalar(
                    select(Vendor.name).where(Vendor.id == invoice.vendor_id)
                )

            field_rows = (
                (
                    await session.execute(
                        select(FieldRow)
                        .where(FieldRow.invoice_id == invoice_id)
                        .order_by(FieldRow.name, FieldRow.id)
                    )
                )
                .scalars()
                .all()
            )
            latest_decision = (
                await session.execute(
                    select(Decision)
                    .where(Decision.invoice_id == invoice_id)
                    .order_by(Decision.created_at.desc(), Decision.id.desc())
                    .limit(1)
                )
            ).scalar_one_or_none()
            proposal_rows = (
                (
                    await session.execute(
                        select(ProposalRow).where(ProposalRow.invoice_id == invoice_id)
                    )
                )
                .scalars()
                .all()
            )
            flag_rows = (
                (
                    await session.execute(
                        select(FraudFlagRow)
                        .where(FraudFlagRow.invoice_id == invoice_id)
                        .order_by(FraudFlagRow.score.desc(), FraudFlagRow.id)
                    )
                )
                .scalars()
                .all()
            )

        fields = tuple(_to_field_view(row, mask_iban=not is_admin) for row in field_rows)
        # Not stored per-field yet (M5.1 schema has no trust-decision table): a field that has
        # not been corrected by a human is treated as accepted from the automated pipeline.
        accepted = {row.name: row.source != FieldSource.HUMAN for row in field_rows}
        reasons: tuple[str, ...] = ()
        if latest_decision is not None and latest_decision.reasons:
            # A `policy_route` row also carries one `{"payee": ...}` element (masked payee or
            # refusal, `apx.policy.interfaces.routing_record`); it is not a reason.
            reasons = tuple(
                str(reason.get("message", reason))
                for reason in latest_decision.reasons
                if "payee" not in reason
            )

        proposals = tuple(
            ProposalModel(
                id=row.id,
                invoice_id=InvoiceId(row.invoice_id) if row.invoice_id else None,
                kind=row.kind.value,
                payload=row.payload,
                created_by=UserId(row.created_by),
                created_at=row.created_at,
                status=row.status.value,
            )
            for row in proposal_rows
        )

        return InvoiceDetail(
            summary=_to_summary(invoice, vendor_name),
            fields=fields,
            accepted=accepted,
            reasons=reasons,
            proposals=proposals,
            flags=tuple(
                FlagView(
                    id=row.id,
                    rule_id=row.rule_id,
                    typology=row.typology,
                    severity=cast(Literal["low", "medium", "high"], row.severity),
                    score=row.score,
                    explanation=row.explanation,
                )
                for row in flag_rows
            ),
        )

    async def edit_field(self, invoice_id: InvoiceId, edit: FieldEdit, user: User) -> None:
        """Applies a reviewer's field fix: appends the new (human-sourced) `fields` row and a
        `labels` row that snapshots the *previous* value/box/source/model-revision alongside
        the correction, then audits both the before and after value (PRD §10 M10.1: every
        correction is a complete, self-contained record -- invoice, field, before/after value,
        box, reviewer, timestamp and the model/revision that produced the original)."""
        async with get_session() as session:
            previous = await self._write_field_edit(session, invoice_id, edit, user)
        await self._audit_field_edit(invoice_id, edit, user, previous)

    async def _write_field_edit(
        self, session: AsyncSession, invoice_id: InvoiceId, edit: FieldEdit, user: User
    ) -> FieldRow | None:
        """Adds the human `fields` row and its `labels` row to `session`; returns the field row
        it supersedes. Shared by a direct edit and an accepted `edit_field` proposal (M-06)."""
        invoice = await session.get(Invoice, invoice_id)
        if invoice is None:
            raise NotFoundError(f"invoice {invoice_id} not found")
        previous = (
            await session.execute(
                select(FieldRow)
                .where(FieldRow.invoice_id == invoice_id, FieldRow.name == edit.field)
                .order_by(FieldRow.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        session.add(
            FieldRow(
                invoice_id=invoice_id,
                name=edit.field,
                value=edit.value,
                normalized_value=edit.value,
                source=FieldSource.HUMAN,
            )
        )
        session.add(
            LabelRow(
                invoice_id=invoice_id,
                field_name=edit.field,
                corrected_value=edit.value,
                corrected_normalized=edit.value,
                previous_value=previous.value if previous else None,
                previous_boxes=previous.boxes if previous else None,
                previous_source=previous.source if previous else None,
                previous_model_revision=previous.model_revision if previous else None,
                source="review",
                # I5: a held-out eval document's correction stays unassigned and is never
                # exported for training; anything else trains (M10.1).
                split=split_for_document(invoice.document_sha256),
                labeled_by=user.id,
                notes=edit.reason,
            )
        )
        return previous

    async def _audit_field_edit(
        self, invoice_id: InvoiceId, edit: FieldEdit, user: User, previous: FieldRow | None
    ) -> None:
        await self._audit.append(
            AuditEvent(
                actor=str(user.id),
                action="field_edit",
                entity=f"invoice:{invoice_id}:field:{edit.field}",
                payload={
                    "value": edit.value,
                    "previous_value": previous.value if previous else None,
                    "boxes": cast(JsonValue, previous.boxes) if previous else None,
                    "model_revision": previous.model_revision if previous else None,
                    "reason": edit.reason,
                },
            )
        )

    async def _action_request(
        self,
        session: AsyncSession,
        user: User,
        action: Capability,
        invoice: Invoice | None,
        *,
        vendor_id: str | None,
        proposal: ProposalModel | None = None,
        bank_change: BankDetailChange | None = None,
    ) -> ActionRequest:
        """`user`'s `ActionRequest` over the *current* database state. The invoice projection
        and vendor come from the same helpers `SqlPaymentStateWriter` uses to recompute the
        grant's digest, so the two always agree unless reality moved in between (STALE_GRANT).
        P2's bank-change history is read from `bank_detail_history`; routing is the stored
        route, and a missing route makes the engine fail closed (ADR-0008 K5)."""
        now = datetime.now(UTC)
        routing = None
        if invoice is not None and invoice.route is not None:
            routing = RoutingDecision(route=invoice.route, reasons=())
        bank_changes: tuple[BankDetailChange, ...] = ()
        if vendor_id is not None:
            cutoff = now - timedelta(days=self._settings.sod_cooldown_days)
            history = (
                (
                    await session.execute(
                        select(BankDetailHistory).where(
                            BankDetailHistory.vendor_id == vendor_id,
                            BankDetailHistory.changed_at >= cutoff,
                        )
                    )
                )
                .scalars()
                .all()
            )
            bank_changes = tuple(
                BankDetailChange(
                    vendor_id=VendorId(row.vendor_id),
                    changed_by=UserId(row.changed_by),
                    changed_at=row.changed_at,
                    old_iban=None,
                    new_iban=row.iban,
                )
                for row in history
            )
        return ActionRequest(
            actor=user,
            action=action,
            invoice=invoice_for_digest(invoice) if invoice is not None else None,
            vendor=await load_vendor(session, vendor_id),
            at=now,
            routing=routing,
            bank_changes=bank_changes,
            bank_change=bank_change,
            proposal=proposal,
        )

    async def _approve(self, invoice_id: InvoiceId, user: User) -> ReviewCorrections:
        """I3: `PolicyGuard.require` issues a digest-bound grant, and `SqlPaymentStateWriter`
        -- the only mutator of payment state (ADR-0008) -- re-checks it, enforces the
        AWAITING_REVIEW precondition under a row lock and is idempotent on replay."""
        async with get_session() as session:
            invoice = await session.get(Invoice, invoice_id)
            if invoice is None:
                raise NotFoundError(f"invoice {invoice_id} not found")
            already_approved = invoice.status is DbInvoiceStatus.APPROVED
            route = invoice.route
            action_request = await self._action_request(
                session, user, Capability.APPROVE_PAYMENT, invoice, vendor_id=invoice.vendor_id
            )

        grant = await self._guard.require(action_request)
        # After authorization, before anything is recorded: an approval without every ADR-0012
        # value is a 422 and never reaches the payment state or the workflow.
        corrections = await self._review_corrections(invoice_id)
        await self._writer.record_approval(invoice_id, grant)
        if not already_approved:
            async with get_session() as session:
                session.add(
                    Decision(
                        invoice_id=invoice_id,
                        actor_user_id=user.id,
                        action=DecisionAction.APPROVE,
                        route=route,
                    )
                )
        return corrections

    async def _review_corrections(self, invoice_id: InvoiceId) -> ReviewCorrections:
        """ADR-0012 amendment: the kind, currency and line VAT categories/rates an approval
        carries to the workflow, read from the invoice's field rows (a reviewer's fix-field row
        wins over the model's). Raises `MissingCorrectionsError` (422) while any is missing, so
        an approval can never reach the e-invoice step with an unresolved value. Its subclasses
        `TotalsMismatchError` (read from the newest routing decision), `SupplierTrnMissingError`
        and `CurrencyRelabelError` refuse what no review can fix."""
        async with get_session() as session:
            route_reasons = await session.scalar(
                select(Decision.reasons)
                .where(
                    Decision.invoice_id == invoice_id,
                    Decision.action == DecisionAction.POLICY_ROUTE,
                )
                .order_by(Decision.created_at.desc(), Decision.id.desc())
                .limit(1)
            )
            rows = (
                (
                    await session.execute(
                        select(FieldRow)
                        .where(FieldRow.invoice_id == invoice_id)
                        .order_by(FieldRow.id)
                    )
                )
                .scalars()
                .all()
            )
        refuse_totals_mismatch(route_reasons)
        return corrections_for_rows(rows)

    async def _approving_proposal_invoice(self, proposal_id: str) -> InvoiceId | None:
        """The invoice an approve_payment proposal would approve, else None."""
        async with get_session() as session:
            row = await session.get(ProposalRow, proposal_id)
        if row is None or ProposalKind(row.kind) is not ProposalKind.APPROVE_PAYMENT:
            return None
        return InvoiceId(row.invoice_id) if row.invoice_id else None

    async def decide(self, invoice_id: InvoiceId, request: DecisionRequest, user: User) -> None:
        signal: bool | None = None
        corrections: ReviewCorrections | None = None
        if request.action == "approve":
            corrections = await self._approve(invoice_id, user)
            signal = True
        elif request.action == "execute_proposal":
            if not request.proposal_id:
                raise PolicyDeniedError("execute_proposal requires proposal_id")
            target = await self._approving_proposal_invoice(request.proposal_id)
            if target is not None:
                corrections = await self._review_corrections(target)
            kind, _ = await self._execute_proposal(request.proposal_id, user)
            signal = True if kind is ProposalKind.APPROVE_PAYMENT else None
        else:
            if Capability.ROUTE_EXCEPTION not in capabilities_for(user.roles):
                raise PolicyDeniedError("actor lacks the route_exception capability")
            async with get_session() as session:
                # Row lock: serialises with an approval (the writer locks the same row) and with
                # the workflow's `routed` status write, so the status checked here is the one
                # the decision lands on.
                invoice = await session.get(Invoice, invoice_id, with_for_update=True)
                if invoice is None:
                    raise NotFoundError(f"invoice {invoice_id} not found")
                if request.action == "reject":
                    if invoice.status is not DbInvoiceStatus.AWAITING_REVIEW:
                        # C-01: the workflow only hears a rejection while it waits for review,
                        # so accepting one earlier would be lost and the invoice paid anyway.
                        raise InvoiceStateConflictError(
                            f"invoice {invoice_id} is {invoice.status.value}; only an invoice "
                            "awaiting review can be rejected"
                        )
                    # H-01: the terminal status and the decision commit together, so a later
                    # approval (the writer also refuses any invoice with a REJECT decision) or
                    # a reopen can never override the rejection.
                    invoice.status = DbInvoiceStatus.BLOCKED
                    session.add(
                        Decision(
                            invoice_id=invoice_id,
                            actor_user_id=user.id,
                            action=DecisionAction.REJECT,
                            route=invoice.route,
                        )
                    )
                    signal = False
                else:
                    if invoice.status not in _PRE_ROUTE_STATUSES:
                        # H-02: never reopens a routed, approved, paid or ended invoice.
                        raise InvoiceStateConflictError(
                            f"invoice {invoice_id} is {invoice.status.value}; only an invoice "
                            "not yet routed can be sent to review"
                        )
                    session.add(
                        Decision(
                            invoice_id=invoice_id,
                            actor_user_id=user.id,
                            action=DecisionAction.SEND_TO_REVIEW,
                        )
                    )

        await self._audit.append(
            AuditEvent(
                actor=str(user.id),
                action=f"decision.{request.action}",
                entity=f"invoice:{invoice_id}",
                payload={
                    "comment": request.comment,
                    "proposal_id": request.proposal_id,
                    "corrections": (
                        None if corrections is None else corrections.model_dump(mode="json")
                    ),
                },
            )
        )
        # Last, after the decision is durable and audited: a failed signal surfaces as a typed
        # 503, and a retry of the same decision is an idempotent no-op plus a re-signal.
        if signal is not None:
            await self._signaller.signal_review(
                invoice_id, approved=signal, corrections=corrections if signal else None
            )

    async def _execute_proposal(
        self, proposal_id: str, user: User
    ) -> tuple[ProposalKind, InvoiceId | None]:
        """P5/K4: authorizes EXECUTE_PROPOSAL (plus the kind's capability) through
        `PolicyGuard`. Payment-affecting kinds (approve_payment, change_bank) are applied only
        by `SqlPaymentStateWriter.execute_proposal`; an `edit_field` proposal is applied through
        the same path as a direct field edit, in the transaction that marks it executed. `route`
        and `request_info` have no executable effect here and are refused (M-06), so nothing is
        ever marked executed without having happened."""
        async with get_session() as session:
            row = await session.get(ProposalRow, proposal_id)
            if row is None:
                raise NotFoundError(f"proposal {proposal_id} not found")
            if row.status != ProposalStatus.PENDING:
                raise PolicyDeniedError(f"proposal {proposal_id} is not pending")
            kind = ProposalKind(row.kind)
            invoice_id = InvoiceId(row.invoice_id) if row.invoice_id else None
            if kind in (ProposalKind.ROUTE, ProposalKind.REQUEST_INFO):
                raise ProposalNotExecutableError(
                    f"a {kind.value} proposal has no action to execute; use the invoice's "
                    "review actions, or send the drafted request yourself, and dismiss it"
                )
            edit = _proposed_edit(row) if kind is ProposalKind.EDIT_FIELD else None

            invoice: Invoice | None = None
            bank_change: BankDetailChange | None = None
            vendor_id: str | None = None
            if kind is ProposalKind.CHANGE_BANK:
                # The writer digests a bank change against the vendor only (no invoice) and the
                # *actual* current IBAN, so the request here is built the same way.
                requested = bank_change_from_payload(row.payload, user.id)
                current_iban = await session.scalar(
                    select(BankDetail.iban).where(
                        BankDetail.vendor_id == requested.vendor_id,
                        BankDetail.is_current.is_(True),
                    )
                )
                bank_change = requested.model_copy(update={"old_iban": current_iban})
                vendor_id = requested.vendor_id
            elif invoice_id is not None:
                invoice = await session.get(Invoice, invoice_id)
                if invoice is None:
                    raise NotFoundError(f"invoice {invoice_id} not found")
                vendor_id = invoice.vendor_id
            action_request = await self._action_request(
                session,
                user,
                Capability.EXECUTE_PROPOSAL,
                invoice,
                vendor_id=vendor_id,
                proposal=proposal_model(row),
                bank_change=bank_change,
            )

        grant = await self._guard.require(action_request)
        if edit is None:
            await self._writer.execute_proposal(proposal_id, grant)
        else:
            if invoice_id is None:
                raise ProposalNotExecutableError(
                    f"edit_field proposal {proposal_id} names no invoice"
                )
            async with get_session() as session:
                locked = await session.get(ProposalRow, proposal_id, with_for_update=True)
                if locked is None or locked.status != ProposalStatus.PENDING:
                    raise PolicyDeniedError(f"proposal {proposal_id} is not pending")
                previous = await self._write_field_edit(session, invoice_id, edit, user)
                locked.status = ProposalStatus.EXECUTED
            await self._audit_field_edit(invoice_id, edit, user, previous)

        # `decisions` rows are per invoice; a vendor-level proposal (change_bank) has none and is
        # recorded by the writer's `payment_state:*` audit events instead.
        if invoice_id is not None:
            async with get_session() as session:
                session.add(
                    Decision(
                        invoice_id=invoice_id,
                        actor_user_id=user.id,
                        action=DecisionAction.EXECUTE_PROPOSAL,
                        proposal_id=proposal_id,
                    )
                )
        return kind, invoice_id

    async def accept_proposal(self, proposal_id: str, user: User) -> ProposalAcceptResponse:
        """`POST /proposals/{id}/accept` (P5, K4): executes a pending proposal if the actor
        holds `EXECUTE_PROPOSAL` plus the capability its kind requires."""
        target = await self._approving_proposal_invoice(proposal_id)
        corrections = await self._review_corrections(target) if target is not None else None
        kind, invoice_id = await self._execute_proposal(proposal_id, user)

        await self._audit.append(
            AuditEvent(
                actor=str(user.id),
                action="proposal.accept",
                entity=f"proposal:{proposal_id}",
                payload={
                    "kind": kind.value,
                    "corrections": (
                        None if corrections is None else corrections.model_dump(mode="json")
                    ),
                },
            )
        )
        if kind is ProposalKind.APPROVE_PAYMENT and invoice_id is not None:
            await self._signaller.signal_review(invoice_id, approved=True, corrections=corrections)
        return ProposalAcceptResponse(id=proposal_id, status="executed")

    async def get_document(self, invoice_id: InvoiceId, store: DocumentStore) -> InvoiceDocument:
        """`GET /invoices/{id}/document` (PRD §10 M5.3/M5.4, docs/security.md B1): resolves the
        invoice's `document_sha256` and reads it back from `store` (the shared, content-addressed
        ingest store under `INGEST_STORE_DIR`, `apx.api.deps.get_document_store`) -- never from a
        path built out of `invoice_id`, a filename, or any other caller-supplied value. The
        digest is re-validated against the sha256 shape here, on top of the
        `document_sha256_is_sha256` CHECK constraint, so a corrupted or otherwise-unexpected row
        can never reach `store.get` with a value that is not exactly 64 lowercase hex characters
        -- the only shape `FileDocumentStore` ever turns into an on-disk path. The media type is
        sniffed from the retrieved bytes (`detect_media_type`), never trusted from a stale
        column, so the response's `Content-Type` always matches what is actually served."""
        async with get_session() as session:
            invoice = await session.get(Invoice, invoice_id)
            if invoice is None:
                raise NotFoundError(f"invoice {invoice_id} not found")
            digest = invoice.document_sha256

        if not _SHA256_RE.fullmatch(digest):
            raise NotFoundError(f"invoice {invoice_id} has no valid stored document reference")

        data = store.get(Sha256Hex(digest))
        media_type = detect_media_type(data)
        if media_type is None:
            raise DocumentNotFoundError(
                f"stored document for invoice {invoice_id} is not a recognised PDF/PNG/JPEG"
            )
        extension = _DOCUMENT_EXTENSION[media_type]
        return InvoiceDocument(data=data, media_type=media_type, filename=f"{digest}.{extension}")

    async def events(self, invoice_id: InvoiceId | None) -> AsyncIterator[SseEvent]:
        """Polls `invoices.status` every second and yields a `status` event on every change,
        stopping once the invoice reaches a terminal status. A real pub/sub (Postgres LISTEN/
        NOTIFY or a Temporal query) is a reasonable follow-up; polling is simple and correct
        for M5.3's single-invoice status stream."""
        if invoice_id is None:
            yield SseEvent(event="error", data={"message": "invoice_id is required"})
            return

        last_status: str | None = None
        while True:
            async with get_session() as session:
                invoice = await session.get(Invoice, invoice_id)
            if invoice is None:
                yield SseEvent(event="error", data={"message": f"invoice {invoice_id} not found"})
                return
            if invoice.status.value != last_status:
                last_status = invoice.status.value
                yield SseEvent(
                    event="status",
                    data={
                        "invoice_id": invoice_id,
                        "status": last_status,
                        "route": invoice.route.value if invoice.route else None,
                    },
                )
            if invoice.status in _TERMINAL_STATUSES:
                return
            await asyncio.sleep(_POLL_INTERVAL_S)


def _proposed_edit(row: ProposalRow) -> FieldEdit:
    """The `FieldEdit` an `edit_field` proposal carries: payload `{"field": str, "value": str,
    "reason"?: str}`. A malformed payload is refused, never partly applied."""
    payload = row.payload
    field, value, reason = payload.get("field"), payload.get("value"), payload.get("reason")
    if not isinstance(field, str) or not isinstance(value, str):
        raise ProposalNotExecutableError(
            f"edit_field proposal {row.id} needs string 'field' and 'value' in its payload"
        )
    try:
        return FieldEdit(
            field=field,
            value=value,
            reason=reason if isinstance(reason, str) else f"accepted proposal {row.id}",
        )
    except ValueError as exc:
        raise ProposalNotExecutableError(f"edit_field proposal {row.id}: {exc}") from exc


class DbVendorService:
    """`GET /vendors` and `GET /vendors/{id}` (PRD §10 M5.4 vendor view): search/paginated list
    and a detail view with bank details always masked to the last 4 IBAN characters (ADR-0008
    line 85), each one's G-3 `verified` generated column, and the vendor's most recent invoices.
    Like `DbInvoiceService`, every method opens its own `AsyncSession`; reads are not audited,
    matching every other read in this API (only field edits, decisions and proposal execution
    are)."""

    async def list(self, *, q: str | None, limit: int, offset: int) -> VendorPage:
        limit = max(1, min(limit, _MAX_PAGE_SIZE))
        offset = max(0, offset)
        async with get_session() as session:
            rows = (await session.execute(vendor_list_stmt(q=q, limit=limit, offset=offset))).all()
            total = (await session.execute(vendor_count_stmt(q=q))).scalar_one()
        items = tuple(
            VendorSummary(
                id=VendorId(vendor.id), name=vendor.name, trn=vendor.trn, verified=bool(verified)
            )
            for vendor, verified in rows
        )
        return VendorPage(items=items, total=total, limit=limit, offset=offset)

    async def get(self, vendor_id: VendorId) -> VendorDetail:
        async with get_session() as session:
            vendor = await session.get(Vendor, vendor_id)
            if vendor is None:
                raise NotFoundError(f"vendor {vendor_id} not found")
            bank_rows = (
                (
                    await session.execute(
                        select(BankDetail)
                        .where(BankDetail.vendor_id == vendor_id)
                        .options(selectinload(BankDetail.verification))
                        .order_by(BankDetail.changed_at.desc(), BankDetail.id.desc())
                    )
                )
                .scalars()
                .all()
            )
            invoice_rows = (
                await session.execute(
                    invoice_list_stmt(
                        status=None,
                        vendor_id=vendor_id,
                        q=None,
                        limit=_RECENT_INVOICES_LIMIT,
                        offset=0,
                    )
                )
            ).all()

        bank_details = tuple(
            VendorBankDetailView(
                id=row.id,
                iban_masked=_mask_iban(row.iban),
                verified=row.verified,
                verified_at=row.verification.verified_at if row.verification else None,
                source=row.source.value,
                is_current=row.is_current,
                changed_at=row.changed_at,
            )
            for row in bank_rows
        )
        recent_invoices = tuple(
            _to_summary(invoice, vendor_name) for invoice, vendor_name in invoice_rows
        )
        return VendorDetail(
            id=VendorId(vendor.id),
            name=vendor.name,
            trn=vendor.trn,
            bank_details=bank_details,
            recent_invoices=recent_invoices,
        )

"""Read-only database facts for the M6.6 attack run: the vendor bank-master
snapshot taken when the run starts, the status/route of in-flight invoices (and `failed` for a
document whose workflow gave up before its invoice existed), every bank-master write
since the snapshot, and one settled invoice's full `CaseState` (payee pinned at routing, reasons,
fraud flags, scan audit, approvals, proposals, audit actions). Raw IBANs are hashed on read."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any, cast

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from apx.db.enums import DecisionAction
from apx.db.models import (
    Approval,
    AuditLog,
    BankDetail,
    BankDetailHistory,
    Decision,
    FraudFlag,
    IngestedDocument,
    Invoice,
    Proposal,
    User,
)
from apx.db.session import get_database_url
from evals.attacks_e2e.outcome import (
    ApprovalRow,
    CaseState,
    IbanRef,
    MasterChange,
    MasterSnapshot,
    PinnedPayee,
    ProposalRow,
    Reason,
    iban_ref,
)

MASTER_AUDIT_ACTIONS = ("payment_state:bank_change", "payment_state:verify_bank")


def with_failed_ingests(
    invoices: dict[str, float], failed_documents: dict[str, float]
) -> dict[str, float]:
    """Invoice creation times plus, for a document that failed before its invoice existed, its
    ingest time; an invoice row, when there is one, is the authority."""
    return {**failed_documents, **invoices}


def with_pre_invoice_failures(
    invoices: dict[str, tuple[str, str | None]], failed_documents: Sequence[str]
) -> dict[str, tuple[str, str | None]]:
    """`invoices` plus `failed` (no route) for each document whose workflow gave up before its
    invoice row existed (`mark_failed` writes `ingested_documents.failed_at` then); an invoice
    row, when there is one, is the authority."""
    merged = dict(invoices)
    for sha in failed_documents:
        merged.setdefault(sha, ("failed", None))
    return merged


def split_routing_record(
    items: Sequence[dict[str, Any]] | None,
) -> tuple[tuple[Reason, ...], PinnedPayee | None]:
    """A `policy_route` decision's `reasons` JSON (`apx.policy.interfaces.routing_record`): the
    reasons, then one `{"payee": ...}` element with the pinned payee or the P1 refusal."""
    reasons: list[Reason] = []
    payee: PinnedPayee | None = None
    for item in items or ():
        if "payee" in item:
            payee = PinnedPayee.model_validate(item["payee"] or {})
        else:
            reasons.append(
                Reason(
                    rule=str(item.get("rule", "")),
                    route=str(item.get("route", "")),
                    message=str(item.get("message", "")),
                )
            )
    return tuple(reasons), payee


def _enum(value: object) -> str | None:
    return None if value is None else str(getattr(value, "value", value))


class AttackDb:
    """Owns its own async engine (the runner is a host process, not the API)."""

    def __init__(self, url: str | None = None) -> None:
        self._engine: AsyncEngine = create_async_engine(url or get_database_url())
        self._sessions = async_sessionmaker(self._engine, expire_on_commit=False)

    async def close(self) -> None:
        await self._engine.dispose()

    async def user_id(self, email: str) -> str | None:
        async with self._sessions() as s:
            return (await s.execute(select(User.id).where(User.email == email))).scalar()

    async def snapshot(self) -> MasterSnapshot:
        async with self._sessions() as s:
            rows = (await s.execute(select(BankDetail))).scalars().all()
            history_max = (await s.execute(select(func.max(BankDetailHistory.id)))).scalar()
            audit_max = (await s.execute(select(func.max(AuditLog.id)))).scalar()
        verified: dict[str, list[IbanRef]] = {}
        for row in rows:
            if row.verified and row.is_current:
                verified.setdefault(row.vendor_id, []).append(iban_ref(row.iban))
        return MasterSnapshot(
            taken_at=time.time(),
            verified={vendor: tuple(refs) for vendor, refs in verified.items()},
            bank_rows={
                str(r.id): (r.vendor_id, iban_ref(r.iban).sha256, r.verified, r.is_current)
                for r in rows
            },
            history_max_id=history_max or 0,
            audit_max_id=audit_max or 0,
        )

    async def statuses(self, shas: Sequence[str]) -> dict[str, tuple[str, str | None]]:
        """`document_sha256 -> (status, route)` for the invoices that exist so far, and `failed`
        for documents that failed before theirs existed (`with_pre_invoice_failures`)."""
        if not shas:
            return {}
        async with self._sessions() as s:
            rows = (
                await s.execute(
                    select(Invoice.document_sha256, Invoice.status, Invoice.route).where(
                        Invoice.document_sha256.in_(list(shas))
                    )
                )
            ).all()
            failed = (
                await s.execute(
                    select(IngestedDocument.sha256).where(
                        IngestedDocument.sha256.in_(list(shas)),
                        IngestedDocument.failed_at.is_not(None),
                    )
                )
            ).scalars()
            invoices = {
                sha: (cast(str, _enum(status)), _enum(route)) for sha, status, route in rows
            }
            return with_pre_invoice_failures(invoices, list(failed))

    async def created(self, shas: Sequence[str]) -> dict[str, float]:
        """`document_sha256 -> invoice created_at` (epoch seconds) for invoices that exist, and
        the ingest time of documents that failed before theirs did (`with_failed_ingests`), so
        a document an earlier run already took through this database counts as stale."""
        if not shas:
            return {}
        async with self._sessions() as s:
            rows = (
                await s.execute(
                    select(Invoice.document_sha256, Invoice.created_at).where(
                        Invoice.document_sha256.in_(list(shas))
                    )
                )
            ).all()
            failed = (
                await s.execute(
                    select(IngestedDocument.sha256, IngestedDocument.received_at).where(
                        IngestedDocument.sha256.in_(list(shas)),
                        IngestedDocument.failed_at.is_not(None),
                    )
                )
            ).all()
        return with_failed_ingests(
            {sha: created.timestamp() for sha, created in rows},
            {sha: received.timestamp() for sha, received in failed},
        )

    async def master_changes(self, before: MasterSnapshot) -> tuple[MasterChange, ...]:
        changes: list[MasterChange] = []
        async with self._sessions() as s:
            for row in (await s.execute(select(BankDetail))).scalars():
                ref = iban_ref(row.iban)
                now = (row.vendor_id, ref.sha256, row.verified, row.is_current)
                if tuple(before.bank_rows.get(str(row.id), ())) != now:
                    changes.append(
                        MasterChange(source="bank_details", vendor_id=row.vendor_id, iban=ref)
                    )
            history = await s.execute(
                select(BankDetailHistory).where(BankDetailHistory.id > before.history_max_id)
            )
            for h in history.scalars():
                changes.append(
                    MasterChange(
                        source=f"bank_details_history:{h.operation}",
                        vendor_id=h.vendor_id,
                        iban=iban_ref(h.iban),
                    )
                )
            audit = await s.execute(
                select(AuditLog).where(
                    AuditLog.id > before.audit_max_id, AuditLog.action.in_(MASTER_AUDIT_ACTIONS)
                )
            )
            for event in audit.scalars():
                payload: dict[str, Any] = event.payload_json or {}
                iban = payload.get("new_iban") or payload.get("iban")
                changes.append(
                    MasterChange(
                        source=f"audit:{event.action}",
                        vendor_id=event.entity.removeprefix("vendor:"),
                        iban=iban_ref(iban) if isinstance(iban, str) else None,
                    )
                )
        return tuple(changes)

    async def case_state(self, sha: str, before: MasterSnapshot) -> CaseState:
        async with self._sessions() as s:
            invoice = (
                await s.execute(select(Invoice).where(Invoice.document_sha256 == sha))
            ).scalar_one_or_none()
            if invoice is None:
                failed_at = await s.scalar(
                    select(IngestedDocument.failed_at).where(IngestedDocument.sha256 == sha)
                )
                return CaseState(
                    invoice_found=False, status=None if failed_at is None else "failed"
                )
            state = await self._invoice_facts(s, invoice)
        return state.model_copy(update={"master_changes": await self.master_changes(before)})

    async def _invoice_facts(self, s: AsyncSession, invoice: Invoice) -> CaseState:
        entity = f"invoice:{invoice.id}"
        route_row = (
            await s.execute(
                select(Decision)
                .where(
                    Decision.invoice_id == invoice.id,
                    Decision.action == DecisionAction.POLICY_ROUTE,
                )
                .order_by(Decision.id.desc())
                .limit(1)
            )
        ).scalar_one_or_none()
        reasons, payee = split_routing_record(route_row.reasons if route_row else None)
        verified_now: tuple[IbanRef, ...] = ()
        if invoice.vendor_id is not None:
            accounts = await s.execute(
                select(BankDetail.iban).where(
                    BankDetail.vendor_id == invoice.vendor_id,
                    BankDetail.is_current.is_(True),
                    BankDetail.verified.is_(True),
                )
            )
            verified_now = tuple(iban_ref(iban) for iban in accounts.scalars())
        fraud = await s.execute(select(FraudFlag.rule_id).where(FraudFlag.invoice_id == invoice.id))
        approvals = await s.execute(select(Approval).where(Approval.invoice_id == invoice.id))
        proposals = await s.execute(select(Proposal).where(Proposal.invoice_id == invoice.id))
        audit = (
            (
                await s.execute(
                    select(AuditLog).where(AuditLog.entity == entity).order_by(AuditLog.id)
                )
            )
            .scalars()
            .all()
        )
        scan = next((e.payload_json for e in reversed(audit) if e.action == "activity:scan"), None)
        return CaseState(
            invoice_found=True,
            status=_enum(invoice.status),
            route=_enum(invoice.route),
            vendor_id=invoice.vendor_id,
            invoice_iban=iban_ref(invoice.iban) if invoice.iban else None,
            total=str(invoice.total),
            currency=invoice.currency,
            payee=payee,
            vendor_verified_now=verified_now,
            reasons=reasons,
            fraud_rules=tuple(sorted(set(fraud.scalars()))),
            scan_flags=None if scan is None else int(scan.get("flags", 0)),
            injection_score=None if scan is None else float(scan.get("injection_score", 0.0)),
            approvals=tuple(
                ApprovalRow(approver_id=a.approver_id, amount=str(a.amount))
                for a in approvals.scalars()
            ),
            proposals=tuple(
                ProposalRow(
                    kind=str(_enum(p.kind)), status=str(_enum(p.status)), created_by=p.created_by
                )
                for p in proposals.scalars()
            ),
            audit_actions=tuple(e.action for e in audit),
            created_at=invoice.created_at.timestamp(),
        )

"""Postgres adapters for the invoice workflow's ports (PRD §10 M5.6): the vendor directory P1/P2
routing reads, the fraud rules' history snapshot, and `SqlInvoiceRecords`, which keeps the
`invoices`, `fields` and `fraud_flags` rows the review UI and API read in step with the workflow."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from typing import Any, cast

from sqlalchemy import delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from apx.common.invoice import Invoice, InvoiceKind
from apx.common.master import BankDetailChange, Vendor
from apx.common.purchasing import PoLine, PurchaseOrder
from apx.common.types import InvoiceId, Money, Route, UserId, VendorId
from apx.db.enums import DbInvoiceStatus, DecisionAction, FieldSource
from apx.db.models import BankDetail as BankDetailRow
from apx.db.models import BankDetailHistory, Decision
from apx.db.models import Field as FieldRow
from apx.db.models import FraudFlag as FraudFlagRow
from apx.db.models import Invoice as InvoiceRow
from apx.db.models import PurchaseOrder as PoRow
from apx.db.models import Vendor as VendorRow
from apx.extract.interfaces import ExtractionResult
from apx.fraud.interfaces import FraudFlag
from apx.policy.interfaces import RoutingDecision, routing_record
from apx.policy.writer import invoice_for_digest, load_vendor
from apx.trust.interfaces import TrustDecision
from apx.workflows.corrections import (
    KIND_FIELD,
    kind_value,
    line_category_field,
    line_rate_field,
)

SessionFactory = Callable[[], AbstractAsyncContextManager[AsyncSession]]

ROUTE_STATUS: Mapping[Route, DbInvoiceStatus] = {
    Route.AUTO: DbInvoiceStatus.APPROVED,
    Route.REVIEW: DbInvoiceStatus.AWAITING_REVIEW,
    Route.BLOCK: DbInvoiceStatus.BLOCKED,
}


async def bank_changes_since(
    session: AsyncSession, vendor_id: VendorId, since: datetime
) -> tuple[BankDetailChange, ...]:
    rows = (
        await session.execute(
            select(BankDetailHistory)
            .where(BankDetailHistory.vendor_id == vendor_id, BankDetailHistory.changed_at >= since)
            .order_by(BankDetailHistory.changed_at)
        )
    ).scalars()
    return tuple(
        BankDetailChange(
            vendor_id=VendorId(row.vendor_id),
            changed_by=UserId(row.changed_by),
            changed_at=row.changed_at,
            old_iban=None,
            new_iban=row.iban,
        )
        for row in rows
    )


class PgVendorDirectory:
    """`VendorDirectory` over the ERP vendor master: current bank accounts plus the bank-detail
    changes inside the review window (P2)."""

    def __init__(
        self,
        session_factory: SessionFactory,
        *,
        window_days: int = 30,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
    ) -> None:
        self._sessions = session_factory
        self._window = timedelta(days=window_days)
        self._clock = clock

    async def lookup(
        self, vendor_id: VendorId | None
    ) -> tuple[Vendor | None, tuple[BankDetailChange, ...]]:
        if vendor_id is None:
            return None, ()
        async with self._sessions() as session:
            vendor = await load_vendor(session, vendor_id)
            if vendor is None:
                return None, ()
            changes = await bank_changes_since(session, vendor_id, self._clock() - self._window)
        return vendor, changes

    async def vendor_id_for_trn(self, trn: str) -> VendorId | None:
        """The vendor whose master TRN is `trn`, or None for an unknown supplier."""
        async with self._sessions() as session:
            found = await session.scalar(select(VendorRow.id).where(VendorRow.trn == trn))
        return VendorId(found) if found is not None else None


@dataclass(frozen=True)
class HistorySnapshot:
    """`apx.fraud.interfaces.InvoiceHistory` over rows read once, up front, so the synchronous
    fraud rules never touch the database. `recent_for_vendor` excludes the invoice under test and
    windows around its issue date, which is how every rule reads "recent"."""

    subject: Invoice
    invoices: tuple[Invoice, ...]
    changes: tuple[BankDetailChange, ...]
    iban_vendors: tuple[VendorId, ...]
    po: PurchaseOrder | None

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]:
        return [
            other
            for other in self.invoices
            if other.vendor_id == vendor_id
            and other.id != self.subject.id
            and abs((other.issue_date - self.subject.issue_date).days) <= days
        ]

    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]:
        """Every loaded change; the rule windows them against the invoice's issue date."""
        return [c for c in self.changes if c.vendor_id == vendor_id]

    def vendors_using_iban(self, iban: str) -> Sequence[VendorId]:
        return self.iban_vendors if iban == self.subject.iban else ()

    def purchase_order(self, number: str) -> PurchaseOrder | None:
        return self.po if self.po is not None and self.po.number == number else None


class PgInvoiceHistory:
    """Loads the `HistorySnapshot` one invoice's fraud assessment needs."""

    def __init__(self, session_factory: SessionFactory, *, max_window_days: int = 180) -> None:
        self._sessions = session_factory
        self._window = timedelta(days=max_window_days)

    async def snapshot(self, invoice: Invoice) -> HistorySnapshot:
        async with self._sessions() as session:
            invoices: tuple[Invoice, ...] = ()
            changes: tuple[BankDetailChange, ...] = ()
            if invoice.vendor_id is not None:
                rows = (
                    await session.execute(
                        select(InvoiceRow).where(
                            InvoiceRow.vendor_id == invoice.vendor_id,
                            InvoiceRow.issue_date >= invoice.issue_date - self._window,
                            InvoiceRow.issue_date <= invoice.issue_date + self._window,
                        )
                    )
                ).scalars()
                invoices = tuple(invoice_for_digest(row) for row in rows)
                since = datetime.combine(invoice.issue_date, time(), UTC) - self._window
                changes = await bank_changes_since(session, invoice.vendor_id, since)
            iban_vendors: tuple[VendorId, ...] = ()
            if invoice.iban:
                found = await session.scalars(
                    select(BankDetailRow.vendor_id)
                    .where(BankDetailRow.iban == invoice.iban, BankDetailRow.is_current)
                    .distinct()
                )
                iban_vendors = tuple(VendorId(v) for v in found)
            po = await self._purchase_order(session, invoice.po_number)
        return HistorySnapshot(invoice, invoices, changes, iban_vendors, po)

    @staticmethod
    async def _purchase_order(session: AsyncSession, number: str | None) -> PurchaseOrder | None:
        if number is None:
            return None
        row = await session.scalar(
            select(PoRow).where(PoRow.po_number == number).options(selectinload(PoRow.lines))
        )
        if row is None:
            return None
        return PurchaseOrder(
            number=row.po_number,
            vendor_id=VendorId(row.vendor_id),
            lines=tuple(
                PoLine(
                    line_ref=str(line.line_no),
                    description=line.description,
                    quantity=line.quantity,
                    unit_price=Money(amount=line.unit_price, currency=line.currency),
                )
                for line in sorted(row.lines, key=lambda line: line.line_no)
            ),
        )


def _classification_fields(invoice: Invoice) -> list[tuple[str, str | None]]:
    rows: list[tuple[str, str | None]] = [(KIND_FIELD, kind_value(invoice.kind))]
    for i, line in enumerate(invoice.lines, start=1):
        category = None if line.vat_category is None else line.vat_category.value
        rate = None if line.vat_rate is None else format(line.vat_rate, "f")
        rows += [(line_category_field(i), category), (line_rate_field(i), rate)]
    return rows


class SqlInvoiceRecords:
    """Keeps the `invoices` row, its model-sourced `fields` rows and its `fraud_flags` in step
    with the workflow. Every write is idempotent (upsert or replace), since Temporal retries
    activities."""

    def __init__(self, session_factory: SessionFactory) -> None:
        self._sessions = session_factory

    async def save(self, invoice: Invoice, result: ExtractionResult) -> None:
        values = {
            "id": invoice.id,
            "document_sha256": invoice.document_sha256,
            "vendor_id": invoice.vendor_id,
            "language": invoice.language,
            "number": invoice.number,
            "issue_date": invoice.issue_date,
            "due_date": invoice.due_date,
            "po_number": invoice.po_number,
            "currency": invoice.total.currency,
            "subtotal": invoice.subtotal.amount,
            "vat_total": invoice.vat_total.amount,
            "total": invoice.total.amount,
            "iban": invoice.iban,
            "kind": None if invoice.kind is None else invoice.kind.value,
            "status": DbInvoiceStatus.GATED,
        }
        mutable = {k: v for k, v in values.items() if k not in ("id", "document_sha256")}
        arbiter = (
            result.arbiter if result.arbiter is not None and result.arbiter.valid_json else None
        )
        base = result.a if result.a.valid_json or not result.b.valid_json else result.b
        async with self._sessions() as session:
            await session.execute(
                insert(InvoiceRow)
                .values(**values)
                .on_conflict_do_update(index_elements=[InvoiceRow.id], set_=mutable)
            )
            await session.execute(
                delete(FieldRow).where(
                    FieldRow.invoice_id == invoice.id, FieldRow.source != FieldSource.HUMAN
                )
            )
            for name, value in result.grounded.items():
                ruled = arbiter is not None and name in result.disputed
                source_output = arbiter if ruled and arbiter is not None else base
                session.add(
                    FieldRow(
                        invoice_id=invoice.id,
                        name=name.value,
                        value=value.raw,
                        normalized_value=value.normalized,
                        boxes=[box.model_dump(mode="json") for box in value.boxes] or None,
                        source=_SOURCE[source_output.extractor.value],
                        model_revision=source_output.model,
                    )
                )
            # ADR-0012: the classified values are field rows too, so a reviewer fixes them with
            # the ordinary field edit and the approval reads them back (NULL = unresolved).
            for field_name, raw in _classification_fields(invoice):
                session.add(
                    FieldRow(
                        invoice_id=invoice.id,
                        name=field_name,
                        value=raw,
                        normalized_value=raw,
                        source=_SOURCE[base.extractor.value],
                        model_revision=base.model,
                    )
                )

    async def set_kind(self, invoice_id: InvoiceId, kind: InvoiceKind) -> None:
        """Records the reviewer-corrected document kind (ADR-0012 amendment)."""
        async with self._sessions() as session:
            await session.execute(
                update(InvoiceRow).where(InvoiceRow.id == invoice_id).values(kind=kind.value)
            )

    async def set_status(self, invoice_id: InvoiceId, status: DbInvoiceStatus) -> None:
        async with self._sessions() as session:
            await session.execute(
                update(InvoiceRow).where(InvoiceRow.id == invoice_id).values(status=status)
            )

    async def routed(
        self, invoice_id: InvoiceId, routing: RoutingDecision, trust: TrustDecision
    ) -> None:
        """Stores the route, its status, each field's calibrated P(correct) and a `policy_route`
        decision whose reasons end with the pinned payee (masked IBAN and match, or the refusal;
        `routing_record`). `v_agent_invoices` reads the newest such row (migration 0008), so the
        agent and reviewer see `select_payee`'s choice. A retry appends another row; the newest
        wins."""
        route = routing.route
        async with self._sessions() as session:
            await session.execute(
                update(InvoiceRow)
                .where(InvoiceRow.id == invoice_id)
                .values(route=route, status=ROUTE_STATUS[route])
            )
            session.add(
                Decision(
                    invoice_id=invoice_id,
                    action=DecisionAction.POLICY_ROUTE,
                    route=route,
                    reasons=cast(list[dict[str, Any]], routing_record(routing)),
                )
            )
            for decision in trust.fields:
                await session.execute(
                    update(FieldRow)
                    .where(
                        FieldRow.invoice_id == invoice_id,
                        FieldRow.name == decision.field.value,
                        FieldRow.source != FieldSource.HUMAN,
                    )
                    .values(confidence=decision.p_correct)
                )

    async def flags(self, invoice_id: InvoiceId, flags: Sequence[FraudFlag]) -> None:
        """Replaces the invoice's fraud flags and marks it fraud-checked."""
        async with self._sessions() as session:
            await session.execute(delete(FraudFlagRow).where(FraudFlagRow.invoice_id == invoice_id))
            session.add_all(
                FraudFlagRow(
                    invoice_id=invoice_id,
                    rule_id=flag.rule_id,
                    typology=flag.typology.value,
                    severity=flag.severity.value,
                    score=flag.score,
                    evidence=flag.evidence,
                    explanation=flag.explanation,
                )
                for flag in flags
            )
            await session.execute(
                update(InvoiceRow)
                .where(InvoiceRow.id == invoice_id)
                .values(status=DbInvoiceStatus.FRAUD_CHECKED)
            )


_SOURCE: Mapping[str, FieldSource] = {
    "a": FieldSource.EXTRACTOR_A,
    "b": FieldSource.EXTRACTOR_B,
    "arbiter": FieldSource.ARBITER,
}

"""Workflow-local seams that are not owned by any single pipeline module: orchestrating
Extractor A/B and the arbiter into one `ExtractionResult`, building the canonical `Invoice`
from grounded fields, and resolving a vendor's current bank state for policy routing. Concrete
implementations land with the extract/validate/fraud modules; `apx.workflows` only calls them
through an injected registry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from apx.common.invoice import Invoice, InvoiceKind
from apx.common.master import BankDetailChange, Vendor
from apx.common.types import InvoiceId, VendorId
from apx.db.enums import DbInvoiceStatus
from apx.extract.interfaces import ExtractionResult
from apx.fraud.interfaces import FraudFlag
from apx.parse.interfaces import ParsedDocument
from apx.policy.interfaces import RoutingDecision
from apx.trust.interfaces import TrustDecision
from apx.workflows.interfaces import InvoiceWorkflowInput


class ExtractionPipeline(Protocol):
    """Runs Extractor A and B, arbitrates disputed fields and grounds the result. A seam for
    `apx.extract`'s future concrete orchestrator (`disputed_fields`, the arbiter call and
    `Grounder.ground`); `apx.workflows` never re-implements dispute detection or grounding."""

    async def run(self, document: ParsedDocument) -> ExtractionResult: ...


class InvoiceBuilder(Protocol):
    """Builds the canonical `Invoice` from grounded extraction fields and vendor resolution. A
    seam for a future `apx.extract`/`apx.validate` concrete builder."""

    async def build(self, result: ExtractionResult, inp: InvoiceWorkflowInput) -> Invoice: ...


class VendorDirectory(Protocol):
    """Resolves a vendor's current bank state for `apx.policy.interfaces.RoutingInput`. A seam
    for the vendor-master lookup that P1/P2 routing needs; `apx.workflows` never re-implements
    bank-detail verification or the segregation-of-duties window itself."""

    async def lookup(
        self, vendor_id: VendorId | None
    ) -> tuple[Vendor | None, tuple[BankDetailChange, ...]]: ...


class InvoiceRecords(Protocol):
    """The `invoices`, `fields` and `fraud_flags` rows the API and review UI read, kept in step
    with the workflow. Every method is idempotent: Temporal retries activities."""

    async def save(self, invoice: Invoice, result: ExtractionResult) -> None: ...
    async def set_status(self, invoice_id: InvoiceId, status: DbInvoiceStatus) -> None: ...
    async def set_kind(self, invoice_id: InvoiceId, kind: InvoiceKind) -> None: ...
    async def routed(
        self, invoice_id: InvoiceId, routing: RoutingDecision, trust: TrustDecision
    ) -> None:
        """Store the route, its status, the fields' P(correct) and a `policy_route` decision
        carrying the reasons and the pinned payee (`apx.policy.interfaces.routing_record`)."""
        ...

    async def flags(self, invoice_id: InvoiceId, flags: Sequence[FraudFlag]) -> None: ...

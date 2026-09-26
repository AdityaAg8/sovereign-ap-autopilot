"""Interfaces for InvoiceWorkflow: statuses, step names used in idempotency keys, workflow
input/state and the activity set the Temporal workflow calls (PRD §3, §6.4, §10 M5.2)."""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from apx.asp_mock.interfaces import ArchiveRecord, AspReceipt
from apx.common.invoice import Invoice
from apx.common.types import Frozen, IdempotencyKey, InvoiceId, Route, Sha256Hex
from apx.einvoice.interfaces import EInvoiceDocument
from apx.extract.interfaces import ExtractionResult
from apx.fraud.interfaces import FraudReport
from apx.parse.interfaces import ParsedDocumentRef
from apx.policy.interfaces import RoutedPayee, RoutingDecision
from apx.scan.interfaces import ScanReport
from apx.trust.interfaces import TrustDecision
from apx.validate.interfaces import ValidationReport


class InvoiceStatus(StrEnum):
    RECEIVED = "received"
    SCANNED = "scanned"
    PARSED = "parsed"
    EXTRACTED = "extracted"
    GATED = "gated"
    VALIDATED = "validated"
    FRAUD_CHECKED = "fraud_checked"
    AWAITING_REVIEW = "awaiting_review"
    APPROVED = "approved"
    SENT = "sent"
    ARCHIVED = "archived"
    BLOCKED = "blocked"
    FAILED = "failed"


class Step(StrEnum):
    """Step names; `f"{invoice_id}:{step}"` is the idempotency key (§6.4)."""

    SCAN = "scan"
    PARSE = "parse"
    EXTRACT = "extract"
    TRUST = "trust"
    VALIDATE = "validate"
    FRAUD = "fraud"
    POLICY = "policy"
    EINVOICE = "einvoice"
    ASP_SEND = "asp_send"
    ARCHIVE = "archive"


class InvoiceWorkflowInput(Frozen):
    invoice_id: InvoiceId
    document_sha256: Sha256Hex
    filename: str
    llm_task_queue: str | None = None
    """Task queue for the llama-swap activities (extract, review_note), served by a worker with
    bounded concurrency (ADR-0013). None runs them on the workflow's own queue (tests)."""


class InvoiceWorkflowState(Frozen):
    invoice_id: InvoiceId
    status: InvoiceStatus
    route: Route | None = None
    error: str | None = None


class InvoiceActivities(Protocol):
    """Activities; each appends to the audit log (I4). Side-effecting ones take a key. Every
    argument and result must be JSON-safe and small (Temporal payloads): the parse travels as a
    `ParsedDocumentRef` into `apx.parse.store`, never as the document with its page PNGs."""

    async def scan(self, inp: InvoiceWorkflowInput) -> ScanReport: ...
    async def parse(self, inp: InvoiceWorkflowInput) -> ParsedDocumentRef: ...
    async def extract(self, parsed: ParsedDocumentRef) -> ExtractionResult: ...
    async def trust_gate(
        self, result: ExtractionResult, parsed: ParsedDocumentRef
    ) -> TrustDecision: ...
    async def build_invoice(
        self, result: ExtractionResult, inp: InvoiceWorkflowInput
    ) -> Invoice: ...
    async def validate(self, invoice: Invoice) -> ValidationReport: ...
    async def fraud(self, invoice: Invoice, parsed: ParsedDocumentRef) -> FraudReport: ...
    async def route(
        self,
        invoice: Invoice,
        trust: TrustDecision,
        validation: ValidationReport,
        fraud: FraudReport,
        scan: ScanReport,
    ) -> RoutingDecision: ...
    async def review_note(self, invoice: Invoice, routing: RoutingDecision) -> bool: ...
    async def build_einvoice(
        self, invoice: Invoice, pinned_payee: RoutedPayee | None
    ) -> EInvoiceDocument: ...
    async def send_to_asp(self, key: IdempotencyKey, document: EInvoiceDocument) -> AspReceipt: ...
    async def archive(self, key: IdempotencyKey, document: EInvoiceDocument) -> ArchiveRecord: ...
    async def mark_failed(self, invoice_id: str, error: str) -> None: ...

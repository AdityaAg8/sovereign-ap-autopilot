"""Interfaces for fraud detection: typology rules (duplicate, split, bank change, ...), the
isolation-forest anomaly score and page-image similarity (PRD §3, §5 fraud gate, §6.3 P4)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Protocol

from pydantic import Field

from apx.common.invoice import Invoice
from apx.common.master import BankDetailChange
from apx.common.purchasing import PurchaseOrder
from apx.common.types import Frozen, InvoiceId, VendorId

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class FraudTypology(StrEnum):
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SPLIT = "split"
    BANK_CHANGE = "bank_change"
    VENDOR_IMPERSONATION = "vendor_impersonation"
    NEW_VENDOR_HIGH_AMOUNT = "new_vendor_high_amount"
    PRICE_ANOMALY = "price_anomaly"
    ANOMALY = "anomaly"


class FraudSeverity(StrEnum):
    """How urgently a flag should be surfaced in review (feeds the M8.3 UI explanation)."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FraudFlag(Frozen):
    rule_id: str
    """Stable id of the rule (or rule mechanism) that raised this flag, e.g. "duplicate_invoice"."""
    typology: FraudTypology
    severity: FraudSeverity
    score: Score
    evidence: str
    """Concrete facts behind the flag (ids, amounts, dates) for the audit trail."""
    explanation: str
    """Plain-language sentence a reviewer can read without domain training (M8.3)."""
    related_invoices: tuple[InvoiceId, ...] = ()


class FraudReport(Frozen):
    flags: tuple[FraudFlag, ...]
    max_score: Score


class SimilarPage(Frozen):
    invoice_id: InvoiceId
    page: int
    similarity: Score


class InvoiceHistory(Protocol):
    """Data access every rule needs: prior invoices, bank changes, cross-vendor IBAN use and
    the PO to match prices against. One context object so every rule takes (invoice, history)."""

    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]: ...
    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]: ...
    def vendors_using_iban(self, iban: str) -> Sequence[VendorId]: ...
    def purchase_order(self, number: str) -> PurchaseOrder | None: ...


class FraudRule(Protocol):
    @property
    def rule_id(self) -> str: ...

    @property
    def typology(self) -> FraudTypology:
        """Read-only, so frozen-dataclass rules satisfy the protocol."""
        ...

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]: ...


class AnomalyDetector(Protocol):
    def score(self, invoice: Invoice) -> float:
        """Isolation-forest anomaly score mapped to [0, 1]."""
        ...


class PageSimilarityIndex(Protocol):
    async def nearest(self, page_png: bytes, k: int) -> list[SimilarPage]:
        """Qwen3-VL-Embedding page neighbours from pgvector (runs in its own GPU stage)."""
        ...


class FraudEngine(Protocol):
    async def assess(self, invoice: Invoice, page_images: Sequence[bytes]) -> FraudReport: ...

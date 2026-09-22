"""Interfaces for fraud detection: typology rules (duplicate, split, bank change, ...), the
isolation-forest anomaly score and page-image similarity (PRD §3, §5 fraud gate, §6.3 P4)."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Annotated, Protocol

from pydantic import Field

from apx.common.invoice import Invoice
from apx.common.master import BankDetailChange
from apx.common.types import Frozen, InvoiceId, VendorId

Score = Annotated[float, Field(ge=0.0, le=1.0)]


class FraudTypology(StrEnum):
    DUPLICATE = "duplicate"
    NEAR_DUPLICATE = "near_duplicate"
    SPLIT = "split"
    BANK_CHANGE = "bank_change"
    VENDOR_IMPERSONATION = "vendor_impersonation"
    PRICE_ANOMALY = "price_anomaly"
    ANOMALY = "anomaly"


class FraudFlag(Frozen):
    typology: FraudTypology
    score: Score
    evidence: str
    related_invoices: tuple[InvoiceId, ...] = ()


class FraudReport(Frozen):
    flags: tuple[FraudFlag, ...]
    max_score: Score


class SimilarPage(Frozen):
    invoice_id: InvoiceId
    page: int
    similarity: Score


class InvoiceHistory(Protocol):
    def recent_for_vendor(self, vendor_id: VendorId, days: int) -> Sequence[Invoice]: ...
    def bank_changes(self, vendor_id: VendorId, days: int) -> Sequence[BankDetailChange]: ...


class FraudRule(Protocol):
    typology: FraudTypology

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

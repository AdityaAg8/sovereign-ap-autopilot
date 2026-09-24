"""Typed fraud rules: duplicate/near-duplicate invoices, split invoices, recent bank-detail
change, new vendor with a high amount, shared IBAN across vendors and PO price variance (PRD
§3 apx/fraud, §10 M8.1). Each rule returns `FraudFlag`s carrying a plain-language explanation.

Credit notes are `Invoice`s with `kind is InvoiceKind.CREDIT_NOTE` and amounts as printed
(ADR-0012), never a negative total. The duplicate and split rules compare only documents of the
same kind, so a credit note is never a duplicate of, or a split part of, the invoice it reverses.
A None kind (unresolved, already routed to review) compares with either kind.
"""

from __future__ import annotations

import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

import imagehash
from PIL import Image, UnidentifiedImageError

from apx.common.errors import FraudError
from apx.common.invoice import Invoice
from apx.common.types import InvoiceId, Money
from apx.fraud.interfaces import (
    FraudFlag,
    FraudSeverity,
    FraudTypology,
    InvoiceHistory,
    PageSimilarityIndex,
    Score,
)
from apx.fraud.normalize import normalize_iban, normalize_invoice_number

RULE_ID_DUPLICATE_INVOICE = "duplicate_invoice"
RULE_ID_NEAR_DUPLICATE_PAGE = "near_duplicate_page"
RULE_ID_NEAR_DUPLICATE_EMBEDDING = "near_duplicate_embedding"


def same_kind(a: Invoice, b: Invoice) -> bool:
    """True when both documents are the same kind, or either kind is unresolved (None): an
    unknown kind must not hide a duplicate, so it is compared conservatively."""
    return a.kind is None or b.kind is None or a.kind is b.kind


def _clamp_score(value: float) -> Score:
    return max(0.0, min(1.0, value))


def _recency_score(days_since: int, window_days: int) -> Score:
    """1.0 for a change on the invoice date itself, fading linearly to 0.0 at the window edge."""
    if window_days <= 0:
        return 1.0
    return _clamp_score(1.0 - (days_since / window_days))


@dataclass(frozen=True, slots=True)
class DuplicateInvoiceRule:
    """Exact-key duplicate detection, plus perceptual-hash and (optional) learned-embedding
    near-duplicate detection over page images (PRD §3: "duplicates ... perceptual hash ...
    Qwen3-VL-Embedding-2B page similarity"). The embedding-backed index lives in
    `apx/fraud/page_similarity.py`; this rule only depends on the `PageSimilarityIndex` Protocol.
    """

    window_days: int = 14
    max_hamming_distance: int = 6
    """imagehash.phash uses a 64-bit hash; distances at or below this are "near-duplicate"."""
    rule_id: str = RULE_ID_DUPLICATE_INVOICE
    typology: FraudTypology = FraudTypology.DUPLICATE

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]:
        """Flag an exact match on normalized number + vendor + amount within the date window.

        A different vendor using the same invoice number is not a duplicate (the vendor check
        below is explicit, independent of what `history.recent_for_vendor` happens to return).
        """
        if invoice.vendor_id is None:
            return []
        key_number = normalize_invoice_number(invoice.number)
        flags: list[FraudFlag] = []
        for other in history.recent_for_vendor(invoice.vendor_id, self.window_days):
            if other.id == invoice.id or other.vendor_id != invoice.vendor_id:
                continue
            if not same_kind(other, invoice):
                continue
            if normalize_invoice_number(other.number) != key_number:
                continue
            if other.total.currency != invoice.total.currency:
                continue
            if other.total.amount != invoice.total.amount:
                continue
            if abs((other.issue_date - invoice.issue_date).days) > self.window_days:
                continue
            flags.append(
                FraudFlag(
                    rule_id=self.rule_id,
                    typology=self.typology,
                    severity=FraudSeverity.HIGH,
                    score=1.0,
                    evidence=(
                        f"Invoice {invoice.number!r} matches {other.number!r} for vendor "
                        f"{invoice.vendor_id}: same amount {invoice.total.amount} "
                        f"{invoice.total.currency}, issue dates within {self.window_days} days."
                    ),
                    explanation=(
                        "This invoice has the same number, vendor and amount as a previous "
                        "invoice submitted within a few days, so it looks like the same invoice "
                        "was submitted twice rather than a new charge."
                    ),
                    related_invoices=(other.id,),
                )
            )
        return flags

    def evaluate_pages(
        self,
        invoice: Invoice,
        page_images: Sequence[bytes],
        candidates: Mapping[InvoiceId, Sequence[bytes]],
    ) -> list[FraudFlag]:
        """Flag near-duplicate pages by perceptual hash against candidate invoices' pages.

        `candidates` is supplied by the caller (typically recent invoices, any vendor) since
        page images are not part of `InvoiceHistory`.
        """
        if not page_images:
            return []
        our_hashes = _page_hashes(page_images)
        flags: list[FraudFlag] = []
        for other_id, other_images in candidates.items():
            if other_id == invoice.id:
                continue
            other_hashes = _page_hashes(other_images)
            distance = _min_hamming_distance(our_hashes, other_hashes)
            if distance is None or distance > self.max_hamming_distance:
                continue
            score = _clamp_score(1.0 - distance / 64.0)
            is_close = distance <= self.max_hamming_distance // 2
            severity = FraudSeverity.HIGH if is_close else FraudSeverity.MEDIUM
            flags.append(
                FraudFlag(
                    rule_id=RULE_ID_NEAR_DUPLICATE_PAGE,
                    typology=FraudTypology.NEAR_DUPLICATE,
                    severity=severity,
                    score=round(score, 4),
                    evidence=(
                        f"Page image hash distance {distance} to invoice {other_id} "
                        f"(threshold {self.max_hamming_distance})."
                    ),
                    explanation=(
                        "A page of this invoice looks visually almost identical to a page of a "
                        "previously seen invoice, which can indicate a rescanned or reprinted "
                        "duplicate even where the extracted text differs."
                    ),
                    related_invoices=(other_id,),
                )
            )
        return flags

    async def evaluate_similarity(
        self,
        invoice: Invoice,
        page_images: Sequence[bytes],
        similarity_index: PageSimilarityIndex,
        min_similarity: Score = 0.9,
        k: int = 5,
    ) -> list[FraudFlag]:
        """Flag near-duplicate pages using the learned page-embedding index, when one is wired
        in. Optional: callers without a `PageSimilarityIndex` simply never call this method.
        """
        flags: list[FraudFlag] = []
        for page_num, page_png in enumerate(page_images):
            for neighbour in await similarity_index.nearest(page_png, k):
                if neighbour.invoice_id == invoice.id or neighbour.similarity < min_similarity:
                    continue
                flags.append(
                    FraudFlag(
                        rule_id=RULE_ID_NEAR_DUPLICATE_EMBEDDING,
                        typology=FraudTypology.NEAR_DUPLICATE,
                        severity=FraudSeverity.MEDIUM,
                        score=neighbour.similarity,
                        evidence=(
                            f"Page {page_num} embeds at similarity {neighbour.similarity:.3f} to "
                            f"invoice {neighbour.invoice_id} page {neighbour.page}."
                        ),
                        explanation=(
                            "A page of this invoice is nearly identical, by learned visual "
                            "similarity, to a page of another invoice already on file."
                        ),
                        related_invoices=(neighbour.invoice_id,),
                    )
                )
        return flags


def _page_hashes(images: Sequence[bytes]) -> tuple[imagehash.ImageHash, ...]:
    hashes = []
    for blob in images:
        try:
            with Image.open(io.BytesIO(blob)) as img:
                hashes.append(imagehash.phash(img))
        except (UnidentifiedImageError, OSError) as exc:
            raise FraudError("page image could not be decoded for near-duplicate hashing") from exc
    return tuple(hashes)


def _min_hamming_distance(
    a: Sequence[imagehash.ImageHash], b: Sequence[imagehash.ImageHash]
) -> int | None:
    if not a or not b:
        return None
    return min(int(x - y) for x in a for y in b)


@dataclass(frozen=True, slots=True)
class SplitInvoiceRule:
    """Flags several invoices from one vendor, each individually under the approval limit,
    whose sum within a short window reaches or exceeds it (PRD §3: "split invoices under the
    approval limit within N days")."""

    approval_limit: Money
    window_days: int = 7
    rule_id: str = "split_invoice"
    typology: FraudTypology = FraudTypology.SPLIT

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]:
        if invoice.vendor_id is None:
            return []
        if invoice.total.currency != self.approval_limit.currency:
            return []
        if invoice.total.amount >= self.approval_limit.amount:
            return []
        siblings = [
            other
            for other in history.recent_for_vendor(invoice.vendor_id, self.window_days)
            if other.id != invoice.id
            and other.vendor_id == invoice.vendor_id
            and same_kind(other, invoice)
            and other.total.currency == self.approval_limit.currency
            and other.total.amount < self.approval_limit.amount
            and abs((other.issue_date - invoice.issue_date).days) <= self.window_days
        ]
        if not siblings:
            return []
        total = invoice.total.amount + sum((s.total.amount for s in siblings), Decimal("0"))
        if total < self.approval_limit.amount:
            return []
        return [
            FraudFlag(
                rule_id=self.rule_id,
                typology=self.typology,
                severity=FraudSeverity.HIGH,
                score=1.0,
                evidence=(
                    f"{len(siblings) + 1} invoices from vendor {invoice.vendor_id} total "
                    f"{total} {self.approval_limit.currency} within {self.window_days} days, "
                    f"each individually under the {self.approval_limit.amount} approval limit."
                ),
                explanation=(
                    "Several smaller invoices from this vendor, each under the approval limit "
                    "on its own, add up to more than the limit within a short window - a common "
                    "way to avoid the approval that the combined amount would require."
                ),
                related_invoices=tuple(s.id for s in siblings),
            )
        ]


@dataclass(frozen=True, slots=True)
class BankDetailChangeRule:
    """Flags an invoice paid to an IBAN that was changed for this vendor shortly before it was
    issued (PRD §3: "recent bank-detail change")."""

    window_days: int = 30
    rule_id: str = "bank_detail_change"
    typology: FraudTypology = FraudTypology.BANK_CHANGE

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]:
        if invoice.vendor_id is None or invoice.iban is None:
            return []
        invoice_iban = normalize_iban(invoice.iban)
        flags: list[FraudFlag] = []
        for change in history.bank_changes(invoice.vendor_id, self.window_days):
            if normalize_iban(change.new_iban) != invoice_iban:
                continue
            changed_date = change.changed_at.date()
            if changed_date > invoice.issue_date:
                continue
            days_since = (invoice.issue_date - changed_date).days
            if days_since > self.window_days:
                continue
            score = _recency_score(days_since, self.window_days)
            flags.append(
                FraudFlag(
                    rule_id=self.rule_id,
                    typology=self.typology,
                    severity=FraudSeverity.HIGH,
                    score=round(score, 4),
                    evidence=(
                        f"Vendor {invoice.vendor_id}'s bank details changed to {invoice_iban} "
                        f"{days_since} day(s) before this invoice's issue date "
                        f"({invoice.issue_date.isoformat()})."
                    ),
                    explanation=(
                        "This invoice pays into an account that was only just added for this "
                        "vendor. Paying immediately after a bank-detail change is a classic sign "
                        "of a compromised vendor or impersonation, so it is worth a callback to "
                        "the vendor's known contact before paying."
                    ),
                    related_invoices=(),
                )
            )
        return flags


@dataclass(frozen=True, slots=True)
class NewVendorHighAmountRule:
    """Flags a high-value invoice from a vendor with no invoice history in the lookback window
    (PRD §3: "new vendor with a high amount")."""

    high_amount_threshold: Money
    new_vendor_window_days: int = 180
    rule_id: str = "new_vendor_high_amount"
    typology: FraudTypology = FraudTypology.NEW_VENDOR_HIGH_AMOUNT

    def __post_init__(self) -> None:
        if self.high_amount_threshold.amount <= 0:
            raise FraudError(
                "high_amount_threshold must be positive, got "
                f"{self.high_amount_threshold.amount} {self.high_amount_threshold.currency}"
            )

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]:
        if invoice.vendor_id is None:
            return []
        if invoice.total.currency != self.high_amount_threshold.currency:
            return []
        if invoice.total.amount < self.high_amount_threshold.amount:
            return []
        prior = [
            other
            for other in history.recent_for_vendor(invoice.vendor_id, self.new_vendor_window_days)
            if other.id != invoice.id and other.vendor_id == invoice.vendor_id
        ]
        if prior:
            return []
        ratio = float(invoice.total.amount / self.high_amount_threshold.amount)
        score = _clamp_score(0.5 * ratio)
        return [
            FraudFlag(
                rule_id=self.rule_id,
                typology=self.typology,
                severity=FraudSeverity.HIGH,
                score=round(score, 4),
                evidence=(
                    f"Vendor {invoice.vendor_id} has no invoices in the last "
                    f"{self.new_vendor_window_days} days, but this invoice totals "
                    f"{invoice.total.amount} {invoice.total.currency}, at or above the "
                    f"{self.high_amount_threshold.amount} high-amount threshold."
                ),
                explanation=(
                    "This vendor has no recent invoice history, and a brand-new vendor issuing "
                    "a high-value invoice is a common fraud pattern that is worth a second look "
                    "before payment."
                ),
                related_invoices=(),
            )
        ]


@dataclass(frozen=True, slots=True)
class SharedIbanRule:
    """Flags an IBAN also used to pay a different vendor (PRD §3: "the same IBAN shared by
    different vendors"), a common shell-vendor / impersonation pattern."""

    rule_id: str = "shared_iban"
    typology: FraudTypology = FraudTypology.VENDOR_IMPERSONATION

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]:
        if invoice.vendor_id is None or invoice.iban is None:
            return []
        iban = normalize_iban(invoice.iban)
        other_vendors = sorted(
            {v for v in history.vendors_using_iban(iban) if v != invoice.vendor_id}
        )
        if not other_vendors:
            return []
        return [
            FraudFlag(
                rule_id=self.rule_id,
                typology=self.typology,
                severity=FraudSeverity.HIGH,
                score=1.0,
                evidence=(
                    f"IBAN {iban} on this invoice is also used by vendor(s) "
                    f"{', '.join(other_vendors)}, not just {invoice.vendor_id}."
                ),
                explanation=(
                    "The bank account on this invoice is shared with one or more other vendors "
                    "in the system, which can mean two different suppliers are being paid into "
                    "the same account - a common sign of vendor impersonation or a fraudulent "
                    "shell vendor."
                ),
                related_invoices=(),
            )
        ]


@dataclass(frozen=True, slots=True)
class PriceVarianceRule:
    """Flags invoice lines whose unit price differs from the matching PO line by more than a
    tolerance (PRD §3: "price variance against the PO")."""

    tolerance_pct: Decimal = Decimal("0.10")
    rule_id: str = "price_variance_po"
    typology: FraudTypology = FraudTypology.PRICE_ANOMALY

    def evaluate(self, invoice: Invoice, history: InvoiceHistory) -> list[FraudFlag]:
        if invoice.po_number is None:
            return []
        po = history.purchase_order(invoice.po_number)
        if po is None:
            return []
        po_lines_by_ref = {line.line_ref: line for line in po.lines}
        flags: list[FraudFlag] = []
        for line in invoice.lines:
            if line.po_line_ref is None:
                continue
            po_line = po_lines_by_ref.get(line.po_line_ref)
            if po_line is None:
                continue
            if po_line.unit_price.currency != line.unit_price.currency:
                continue
            po_price = po_line.unit_price.amount
            if po_price == 0:
                continue
            variance = abs(line.unit_price.amount - po_price) / po_price
            if variance <= self.tolerance_pct:
                continue
            severity = (
                FraudSeverity.HIGH if variance > self.tolerance_pct * 2 else FraudSeverity.MEDIUM
            )
            flags.append(
                FraudFlag(
                    rule_id=self.rule_id,
                    typology=self.typology,
                    severity=severity,
                    # Same scale as the other rules: 0.5 just over tolerance, 1.0 at twice it.
                    score=round(_clamp_score(float(variance / (2 * self.tolerance_pct))), 4),
                    evidence=(
                        f"Line {line.description!r} bills {line.unit_price.amount} "
                        f"{line.unit_price.currency} vs PO {po.number} line {line.po_line_ref} "
                        f"price {po_price} ({variance:.1%} variance, tolerance "
                        f"{self.tolerance_pct:.0%})."
                    ),
                    explanation=(
                        "The unit price billed on this line differs from the purchase order "
                        "price by more than the allowed tolerance, which can mean the invoice "
                        "was altered or the vendor is overcharging against the agreed price."
                    ),
                    related_invoices=(),
                )
            )
        return flags
